applyWorkerGeminiKey("ORACLE_GEMINI_API_KEY");
import { requireEnv, requireAnyLLMKey, applyWorkerGeminiKey, createThrottle } from "../../lib/agent-bootstrap";
import { kellyFraction } from "../../lib/kelly";
import { type VerdictPayload } from "../../lib/verdict";
import { parseLLMVerdictWithRetry, VERDICT_LLM_SCHEMA, VERDICT_RETRY_SUFFIX } from "../../lib/verdict-parser";
import { INJECTION_GUARD, fenceUntrusted } from "../../lib/prompt-safety";
import { callLLM, activeLLMProvider, activeLLMModel, activeLLMKeyFingerprint, pickGeminiModel, extractJson } from "../../lib/llm";
import { BPS_DIVISOR, challengeClaim, getClaimCount, readClaimRaw, resolveClaim, type ClaimData } from "../../lib/contract";
import { getOracleWallet, readAgentBalances } from "../../lib/agent-wallets";
import { evidenceCommitmentHash, type CouncilCommitment } from "../../lib/evidence-commitment";
import { STELLAR_NETWORK, getExplorerTxUrl, requireMarketContractId } from "../../lib/stellar";
import { fetchWithBudget, payingWalletFor } from "../../lib/x402/buyer";
import { reportingPoll } from "../../lib/ops/heartbeat";
import { unitsToUsdc, usdcToUnits, formatAtomicUsdc } from "../../lib/usdc";
import { fetchEvidence as fetchEvidenceShared, EvidenceFetchError, type EvidenceFetcherKind, type EvidencePayment } from "../../lib/server/evidence-fetcher";
import { gatherCouncilVerdict, scoreCouncilVotes, payCouncilBonuses, parseCouncilBonusPool, isConfirmedCouncilSettlement, verdictToProbability, Q_PRIOR, type CouncilVote } from "./council-vote";
import { normalizeQuorum } from "../../lib/council/quorum";
import { RiskManager, DEFAULT_RISK_CONFIG, validateRiskConfig, isEvidenceStale } from "../../lib/oracle-risk";

const POLL_INTERVAL_MS = Number(process.env.ORACLE_POLL_INTERVAL_MS?? "60000");
const MAX_CONTENT_CHARS = 8_000;
const CONTRACT_ID = requireMarketContractId();
const AUTO_CHALLENGE = process.env.AUTO_CHALLENGE === "1";
const CHALLENGE_STAKE_USDC = Number(process.env.CHALLENGE_STAKE_USDC?? "2");
const CHALLENGE_CONFIDENCE = Number(process.env.CHALLENGE_CONFIDENCE?? "80");
const LLM_THROTTLE_MS = Number(process.env.ORACLE_LLM_THROTTLE_MS?? "8000");
const PAY_EVIDENCE = process.env.PAY_EVIDENCE!== "0";
const EVIDENCE_POOL_BPS = Number(process.env.EVIDENCE_POOL_BPS?? "50");
const EVIDENCE_MAX_USDC = Number(process.env.EVIDENCE_MAX_USDC?? "0.05");
const EVIDENCE_MIN_USDC = Number(process.env.EVIDENCE_MIN_USDC?? "0.001");
const COUNCIL_SETTLEMENT = process.env.COUNCIL_SETTLEMENT === "1";
const COUNCIL_BASE_URL = process.env.MIMIR_BASE_URL?? "http://localhost:3000";
const COUNCIL_QUORUM = normalizeQuorum(process.env.COUNCIL_QUORUM?? "3");
const COUNCIL_VOTE_CAP = Number(process.env.COUNCIL_VOTE_CAP_USDC?? "0.005");
const COUNCIL_SELF_RESOLVING = COUNCIL_SETTLEMENT && process.env.COUNCIL_SELF_RESOLVING === "1";
const COUNCIL_ALPHA = Number(process.env.COUNCIL_ALPHA?? "0.25");
const COUNCIL_BONUS_ATOMIC = parseCouncilBonusPool(process.env.COUNCIL_BONUS_USDC?? "0.01");
const SETTLEMENT_DELAY_MS = Number(process.env.ORACLE_SETTLEMENT_DELAY_MS?? "900000");

const llmGate = createThrottle(LLM_THROTTLE_MS);
async function throttledLLM(...args: Parameters<typeof callLLM>): Promise<string> { await llmGate(); return callLLM(...args); }

const riskManager = new RiskManager(DEFAULT_RISK_CONFIG);
const riskCfgErrs = validateRiskConfig(DEFAULT_RISK_CONFIG);
if (riskCfgErrs.length) throw new Error(`Risk config invalid: ${riskCfgErrs.join(", ")}`);

requireEnv(["ORACLE_SECRET"]); requireAnyLLMKey();
const ORACLE = getOracleWallet(); const ORACLE_ADDR = ORACLE.address; const ORACLE_PAYER = payingWalletFor(ORACLE);
type ClaimOnChain = ClaimData; type OracleVerdict = VerdictPayload;

async function fetchClaim(claimId: number): Promise<ClaimOnChain | null> {
  try { return await readClaimRaw(claimId); } catch { riskManager.recordFailure("dependency"); return null; }
}
interface EvidenceResult { text: string; fetcher: EvidenceFetcherKind | "none"; sourceUrl?: string; fetchedAt?: number; payment?: EvidencePayment; }
function evidenceBudgetUsdc(claim: ClaimOnChain): number { const potUsdc = claim.total_pot; const fraction = (potUsdc * EVIDENCE_POOL_BPS) / BPS_DIVISOR; return Math.min(EVIDENCE_MAX_USDC, Math.max(EVIDENCE_MIN_USDC, fraction)); }
async function fetchEvidence(claim: ClaimOnChain): Promise<EvidenceResult> {
  const url = claim.resolution_url; if (!url?.startsWith("http")) return { text: "(No resolution URL provided)", fetcher: "none" };
  const budgetUsdc = evidenceBudgetUsdc(claim); const maxUnits = usdcToUnits(budgetUsdc);
  const paidFetch = PAY_EVIDENCE? async (u: string, init?: RequestInit) => { const r = await fetchWithBudget(u, ORACLE_PAYER, maxUnits, init); return { response: r.response, payment: r.payment? { priceUnits: r.payment.priceUnits.toString(), txHash: r.payment.txHash } : null }; } : undefined;
  try { const snap = await fetchEvidenceShared(url, { maxChars: MAX_CONTENT_CHARS, userAgent: "Mimir-Oracle/1.0", paidFetch }); return { text: snap.text, fetcher: snap.fetcher, sourceUrl: snap.sourceUrl, fetchedAt: snap.fetchedAt, payment: snap.payment }; } catch (err: any) { const msg = err instanceof EvidenceFetchError? err.message : (err?.message?? "unknown"); return { text: `(Failed to fetch: ${msg})`, fetcher: "none" }; }
}
async function evaluateClaim(claim: ClaimOnChain, evidence: string, jurorHistory: string[] = []): Promise<OracleVerdict> {
  const deadlineDate = new Date(claim.deadline * 1000).toISOString(); const nowDate = new Date().toISOString(); const potUsdc = claim.total_pot;
  const jurySection = jurorHistory.length > 0? `\n## Council juror reports\n${fenceUntrusted("juror-reports", jurorHistory.map((r, i) => `${i + 1}. ${r}`).join("\n"))}\n` : "";
  const claimBlock = fenceUntrusted("claim", [`Question: ${claim.question}`, `Creator position (Side A): ${claim.creator_position}`, `Challenger position (Side B): ${claim.counter_position}`, `Category: ${claim.category}`, `Market type: ${claim.market_type}`, claim.handicap_line? `Handicap: ${claim.handicap_line}` : null, `Settlement rule: ${claim.settlement_rule || "Use the linked source"}`, `Resolution URL: ${claim.resolution_url}`].filter(Boolean).join("\n"));
  const prompt = `You are Mimir, impartial AI oracle.\n${INJECTION_GUARD}\n## Time\n- Current UTC: ${nowDate}\n- Deadline: ${deadlineDate}\n- Pot: ${potUsdc.toFixed(2)} USDC\n## Claim\n${claimBlock}\n## Evidence\n${fenceUntrusted("web-evidence", evidence)}\n${jurySection}\nReturn JSON only: {"verdict":"CREATOR_WINS"|"CHALLENGERS_WIN"|"DRAW"|"UNRESOLVABLE","confidence":<0-100>,"explanation":"<one paragraph>"}`;
  const { result, lastRawText, attempts } = await parseLLMVerdictWithRetry({ extractor: extractJson, buildPrompt: (attempt) => attempt === 1? prompt : `${prompt}${VERDICT_RETRY_SUFFIX}`, callLLMFn: (p) => throttledLLM(p, { maxTokens: 1024, jsonOnly: true, model: pickGeminiModel("oracle"), jsonSchema: VERDICT_LLM_SCHEMA }), });
  if (!result.ok) throw new Error(`Oracle verdict ${result.reason} after ${attempts} attempts: ${result.detail} — raw: ${lastRawText.slice(0, 200)}`); return result.payload;
}
function verdictToSide(v: OracleVerdict["verdict"]): "creator" | "challengers" | "draw" | "unresolvable" { switch (v) { case "CREATOR_WINS": return "creator"; case "CHALLENGERS_WIN": return "challengers"; case "DRAW": return "draw"; case "UNRESOLVABLE": return "unresolvable"; } }
const KELLY_CAP = 0.25; const CONFIDENCE_HIGH_MIN = 80; const CONFIDENCE_MED_MIN = 60;
function tierVerdict(verdict: OracleVerdict): OracleVerdict { if (verdict.verdict === "UNRESOLVABLE" || verdict.verdict === "DRAW") return verdict; if (verdict.confidence >= CONFIDENCE_HIGH_MIN) return verdict; if (verdict.confidence >= CONFIDENCE_MED_MIN) return {...verdict, explanation: `[CONTESTED] ${verdict.explanation}`.slice(0, 500) }; return { verdict: "UNRESOLVABLE", confidence: verdict.confidence, explanation: `[LOW CONFIDENCE — refunded] ${verdict.explanation}`.slice(0, 500) }; }
const MAX_CONFIDENCE_NON_API = 75;
function applyFetcherTrust(verdict: OracleVerdict, fetcher: EvidenceFetcherKind | "none"): OracleVerdict { if (fetcher === "coingecko-api") return verdict; if (verdict.verdict === "UNRESOLVABLE") return verdict; const cappedConfidence = Math.min(verdict.confidence, MAX_CONFIDENCE_NON_API); const tag = fetcher === "jina"? "[via-jina]" : fetcher === "direct"? "[via-scrape]" : "[no-fetch]"; return {...verdict, confidence: cappedConfidence, explanation: `${tag} ${verdict.explanation}`.slice(0, 500) }; }
const SPORTS_SETTLE_GRACE_SECS = Math.max(1, Number(process.env.SPORTS_SETTLE_GRACE_HOURS?? 12)) * 3600;
async function isSportsEventFinal(claim: ClaimOnChain, evidenceText: string): Promise<boolean> {
  const prompt = `Determine if match DEFINITIVELY CONCLUDED.\n${INJECTION_GUARD}\n## Claim\n${fenceUntrusted("claim", `Question: ${claim.question}`)}\n## Evidence\n${fenceUntrusted("web-evidence", evidenceText)}\nReply JSON only: { "final": true | false }`;
  try { const text = await throttledLLM(prompt, { maxTokens: 64, jsonOnly: true, model: pickGeminiModel("oracle"), jsonSchema: { type: "object", properties: { final: { type: "boolean" } }, required: ["final"] } }); const parsed = JSON.parse(extractJson(text)?? "{}"); return parsed.final === true; } catch { return false; }
}
async function settle(claim: ClaimOnChain): Promise<boolean> {
  console.log(`\n[settle] Claim #${claim.id}: "${claim.question.slice(0, 60)}..."`); const evidence = await fetchEvidence(claim); console.log(`[settle] Evidence fetcher: ${evidence.fetcher}`);
  if (claim.category.toLowerCase() === "sports") { const now = Math.floor(Date.now() / 1000); const pastGrace = now > claim.deadline + SPORTS_SETTLE_GRACE_SECS; if (!pastGrace &&!(await isSportsEventFinal(claim, evidence.text))) { console.log(`[settle] Claim #${claim.id}: match not final yet — deferring.`); return false; } }
  if (evidence.payment) { const paid = unitsToUsdc(BigInt(evidence.payment.priceUnits)); console.log(`[settle] Paid ${paid.toFixed(6)} USDC for evidence`); }
  let rawVerdict: OracleVerdict; let councilCommitment: CouncilCommitment | null = null; let bonusVotes: CouncilVote[] | null = null;
  if (COUNCIL_SETTLEMENT) {
    const council = await gatherCouncilVerdict({ claimId: claim.id, category: claim.category, baseUrl: COUNCIL_BASE_URL, payer: ORACLE_PAYER, capUsdc: COUNCIL_VOTE_CAP, quorum: COUNCIL_QUORUM, claimState: claim.state,...(COUNCIL_SELF_RESOLVING? { selfResolving: { alpha: COUNCIL_ALPHA, minVotes: COUNCIL_QUORUM } } : {}) }).catch((err) => { console.warn(`[settle] council failed`, err); return null; });
    if (council && COUNCIL_SELF_RESOLVING) { const reference = await evaluateClaim(claim, evidence.text, council.reports?? []); const referenceQ = verdictToProbability(reference.verdict, reference.confidence, Q_PRIOR); council.votes = scoreCouncilVotes(council.votes, referenceQ); rawVerdict = reference; councilCommitment = { tally: council.tally, qChain: council.qHistory?? [], referenceQ: Number(referenceQ.toFixed(4)), scores: council.votes.filter(v => v.probability!== undefined).map(v => Number((v.score?? 0).toFixed(4))) }; bonusVotes = council.votes; }
    else if (council) { rawVerdict = { verdict: council.verdict, confidence: council.confidence, explanation: council.explanation }; councilCommitment = { tally: council.tally }; }
    else { rawVerdict = await evaluateClaim(claim, evidence.text); }
  } else { rawVerdict = await evaluateClaim(claim, evidence.text); }
  const evidenceHash = evidenceCommitmentHash({ evidence: evidence.text, fetcher: evidence.fetcher, sourceUrl: evidence.sourceUrl, fetchedAt: evidence.fetchedAt, now: Date.now(), council: councilCommitment });
  const trusted = applyFetcherTrust(rawVerdict, evidence.fetcher); const verdict = tierVerdict(trusted);
  console.log(`[settle] Verdict: ${verdict.verdict} (${verdict.confidence}%)`);
  const settled = await resolveClaim(ORACLE.signer, claim.id, { winner_side: verdictToSide(verdict.verdict), summary: verdict.explanation, confidence: verdict.confidence, evidence_hash: evidenceHash });
  console.log(`[settle] Resolved — ${settled.explorerUrl?? settled.txHash}`);
  if (bonusVotes && COUNCIL_BONUS_ATOMIC > 0n) {
    try {
      const confirmed = settled.pending? null : await fetchClaim(claim.id);
      if (!isConfirmedCouncilSettlement(confirmed, verdictToSide(verdict.verdict), evidenceHash, Boolean(settled.pending))) { console.warn(`[settle] Bonus withheld: not confirmed`); }
      else { const receipts = await payCouncilBonuses({ votes: bonusVotes, poolAtomic: COUNCIL_BONUS_ATOMIC, payerWallet: ORACLE, claimId: claim.id, contractId: CONTRACT_ID, settlementTxHash: settled.txHash }); for (const r of receipts) console.log(`[settle] Bonus ${formatAtomicUsdc(r.amountAtomic)} USDC to ${r.slug}: ${r.status}`); }
    } catch { console.warn(`[settle] Bonus withheld for manual reconciliation`); }
  }
  return true;
}
async function challengeIfMispriced(claim: ClaimOnChain): Promise<void> {
  if (!AUTO_CHALLENGE) return;
  const validation = riskManager.validateClaim(claim);
  if (!validation.ok) { console.log(`[challenge] Skip #${claim?.id}: ${validation.reason} — ${validation.detail}`); if (validation.reason === "malformed" || validation.reason === "dependency") riskManager.recordFailure(validation.reason); if (claim) riskManager.recordEvaluated(claim.id); return; }
  if (claim.is_private) { riskManager.recordEvaluated(claim.id); return; }
  if (claim.creator === ORACLE_ADDR) { riskManager.recordEvaluated(claim.id); return; }
  if ((claim.challenger_addresses?? []).includes(ORACLE_ADDR)) { riskManager.recordEvaluated(claim.id); return; }
  if (claim.max_challengers > 0 && claim.challenger_count >= claim.max_challengers) { riskManager.recordEvaluated(claim.id); return; }
  if (riskManager.isInCooldown()) { console.log(`[risk] Cooldown active —`, riskManager.getStats()); return; }
  let balances; try { balances = await readAgentBalances(ORACLE_ADDR); } catch { riskManager.recordFailure("dependency"); return; }
  if (balances.usdc === null) { console.log(`[challenge] No USDC trustline`); return; }
  if (balances.usdc < CHALLENGE_STAKE_USDC) { console.log(`[challenge] Insufficient USDC ${balances.usdc.toFixed(2)}`); return; }
  console.log(`\n[challenge] Evaluating claim #${claim.id}: "${claim.question.slice(0, 60)}..."`);
  const evidence = await fetchEvidence(claim);
  if (evidence.fetcher === "none") { console.log(`[challenge] No evidence — dependency failure`); riskManager.recordFailure("dependency"); riskManager.recordEvaluated(claim.id); return; }
  if (isEvidenceStale(evidence.fetchedAt)) { console.log(`[challenge] Stale evidence #${claim.id} — skip`); riskManager.recordEvaluated(claim.id); return; }
  let rawVerdict: OracleVerdict; try { rawVerdict = await evaluateClaim(claim, evidence.text); } catch (e: any) { console.warn(`[challenge] LLM dependency failure #${claim.id}:`, e.message); riskManager.recordFailure("dependency"); return; }
  const verdict = applyFetcherTrust(rawVerdict, evidence.fetcher);
  console.log(`[challenge] Early verdict: ${verdict.verdict} (${verdict.confidence}%)`);
  if (verdict.verdict!== "CHALLENGERS_WIN" || verdict.confidence < CHALLENGE_CONFIDENCE) { console.log(`[challenge] Not confident enough — skip`); riskManager.recordEvaluated(claim.id); return; }
  const kelly = kellyFraction(verdict.confidence, KELLY_CAP); const bankroll = balances.usdc; let kellyStake = Math.max(CHALLENGE_STAKE_USDC, Math.min(bankroll * kelly, bankroll * 0.1)); kellyStake = Math.min(kellyStake, riskManager.config.maxStakePerClaimUsdc); const stakeUsdc = Math.round(kellyStake * 100) / 100;
  const can = riskManager.canChallenge(stakeUsdc); if (!can.ok) { console.log(`[challenge] Blocked #${claim.id}: ${can.reason} — ${can.detail}`, riskManager.getStats()); riskManager.recordEvaluated(claim.id); return; }
  console.log(`[challenge] Kelly: ${(kelly * 100).toFixed(1)}% → ${stakeUsdc} USDC | before:`, riskManager.getStats());
  try { const staked = await challengeClaim(ORACLE.signer, claim.id, stakeUsdc); riskManager.recordChallenge(claim.id, stakeUsdc); riskManager.resetFailures(); console.log(`[challenge] ✓ Staked ${stakeUsdc} USDC — ${staked.explorerUrl?? staked.txHash} | exposure now ${riskManager.getStats().exposure}`); } catch (e: any) { console.error(`[challenge] Tx failed #${claim.id}:`, e.message); riskManager.recordFailure("dependency"); }
}
async function poll(): Promise<void> {
  const now = Math.floor(Date.now() / 1000); let total: number; try { total = await getClaimCount(); } catch (err) { console.warn("[oracle] Failed claim count", err); riskManager.recordFailure("dependency"); return; }
  console.log(`\n[oracle] Poll at ${new Date().toISOString()} — ${total} claims | risk:`, riskManager.getStats());
  const settled: number[] = []; const challenged: number[] = []; const expiredActive: ClaimOnChain[] = [];
  for (let id = 1; id <= total; id++) { const claim = await fetchClaim(id); if (!claim) continue; if (claim.state === "active" && claim.deadline <= now) { expiredActive.push(claim); continue; } try { if ((claim.state === "open" || claim.state === "active") && claim.deadline > now) { const before = riskManager.getStats().concurrent; await challengeIfMispriced(claim); if (riskManager.getStats().concurrent > before) challenged.push(id); } } catch (err) { console.error(`[oracle] Error on claim ${id}:`, err); } }
  expiredActive.sort((a, b) => a.deadline - b.deadline);
  for (let i = 0; i < expiredActive.length; i++) { const claim = expiredActive[i]; try { const resolved = await settle(claim); if (!resolved) continue; settled.push(claim.id); riskManager.recordSettled(claim.total_challenger_stake); if (i < expiredActive.length - 1 && SETTLEMENT_DELAY_MS > 0) { console.log(`[oracle] Cooling down ${(SETTLEMENT_DELAY_MS / 60000).toFixed(1)} min...`); await new Promise(r => setTimeout(r, SETTLEMENT_DELAY_MS)); } } catch (err) { console.error(`[oracle] Error settling claim ${claim.id}:`, err); } }
  const summary = [settled.length? `Settled: [${settled.join(", ")}]` : null, challenged.length? `Challenged: [${challenged.join(", ")}]` : null].filter(Boolean).join(" | "); console.log(summary? `[oracle] ${summary}` : "[oracle] Nothing to do this round.");
}
async function main(): Promise<void> {
  const balances = await readAgentBalances(ORACLE_ADDR); if (!balances.exists) throw new Error(`oracle account ${ORACLE_ADDR} does not exist — run: npm run agents:fund`);
  console.log("═══════════════════════════════════════════════"); console.log(" Mimir Oracle Agent — BOUNDED RISK (Issue #111)"); console.log(` Contract : ${CONTRACT_ID}`); console.log(` Oracle : ${ORACLE_ADDR}`); console.log(` Bankroll : ${balances.usdc === null? "no trustline" : `${balances.usdc.toFixed(4)} USDC`}`); console.log(` Network : Stellar ${STELLAR_NETWORK}`); console.log(` LLM : ${activeLLMProvider()} / ${activeLLMModel()}`); console.log(` Risk : daily=${DEFAULT_RISK_CONFIG.maxDailyChallengeUsdc} USDC, exposure=${DEFAULT_RISK_CONFIG.maxTotalExposureUsdc} USDC, concurrent=${DEFAULT_RISK_CONFIG.maxConcurrentChallenges}`); console.log(` Auto-challenge: ${AUTO_CHALLENGE? `YES ≥${CHALLENGE_CONFIDENCE}%` : "OFF"}`); console.log("═══════════════════════════════════════════════\n");
  const safePoll = () => reportingPoll("oracle", "oracle", POLL_INTERVAL_MS / 1000, poll); await safePoll(); setInterval(safePoll, POLL_INTERVAL_MS);
}
main().catch((err) => { console.error("[oracle] Fatal:", err); process.exit(1); });
