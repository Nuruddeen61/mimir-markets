/**
 * Mimir Oracle Agent — AI economic actor on Stellar
 * Issue #111: Bound autonomous challenge risk
 *
 * SECURITY HARDENING:
 * - Exposure limits (daily, total, concurrent, per-claim)
 * - Circuit breaker on dependency failures
 * - Explicit handling: malformed, stale, duplicate, cancelled, paused, dependency-failure
 * - Soroban state authoritative (on-chain roster check)
 * - Funded-state safety: never challenges if no trustline / insufficient USDC
 *
 * Rollback: revert this file + delete lib/oracle-risk.ts — existing flows safe
 * Accounting impact: limits USDC outflow, no change to settlement escrow logic
 * Trust impact: prevents autonomous runaway staking, preserves oracle neutrality
 * Operational impact: new env vars, risk logs, cooldown
 */

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

const POLL_INTERVAL_MS = Number(process.env.ORACLE_POLL_INTERVAL_MS ?? "60000");
const MAX_CONTENT_CHARS = 8_000;
const CONTRACT_ID = requireMarketContractId();
const AUTO_CHALLENGE = process.env.AUTO_CHALLENGE === "1";
const CHALLENGE_STAKE_USDC = Number(process.env.CHALLENGE_STAKE_USDC ?? "2");
const CHALLENGE_CONFIDENCE = Number(process.env.CHALLENGE_CONFIDENCE ?? "80");
const LLM_THROTTLE_MS = Number(process.env.ORACLE_LLM_THROTTLE_MS ?? "8000");
const PAY_EVIDENCE = process.env.PAY_EVIDENCE !== "0";
const EVIDENCE_POOL_BPS = Number(process.env.EVIDENCE_POOL_BPS ?? "50");
const EVIDENCE_MAX_USDC = Number(process.env.EVIDENCE_MAX_USDC ?? "0.05");
const EVIDENCE_MIN_USDC = Number(process.env.EVIDENCE_MIN_USDC ?? "0.001");
const COUNCIL_SETTLEMENT = process.env.COUNCIL_SETTLEMENT === "1";
const COUNCIL_BASE_URL = process.env.MIMIR_BASE_URL ?? "http://localhost:3000";
const COUNCIL_QUORUM = normalizeQuorum(process.env.COUNCIL_QUORUM ?? "3");
const COUNCIL_VOTE_CAP = Number(process.env.COUNCIL_VOTE_CAP_USDC ?? "0.005");
const COUNCIL_SELF_RESOLVING = COUNCIL_SETTLEMENT && process.env.COUNCIL_SELF_RESOLVING === "1";
const COUNCIL_ALPHA = Number(process.env.COUNCIL_ALPHA ?? "0.25");
const COUNCIL_BONUS_ATOMIC = parseCouncilBonusPool(process.env.COUNCIL_BONUS_USDC ?? "0.01");
const SETTLEMENT_DELAY_MS = Number(process.env.ORACLE_SETTLEMENT_DELAY_MS ?? "900000");

const llmGate = createThrottle(LLM_THROTTLE_MS);
async function throttledLLM(...args: Parameters<typeof callLLM>): Promise<string> {
  await llmGate();
  return callLLM(...args);
}

// ── Risk Manager (Issue #111) ─────────────────────────────────────────
const riskManager = new RiskManager(DEFAULT_RISK_CONFIG);
const riskCfgErrs = validateRiskConfig(DEFAULT_RISK_CONFIG);
if (riskCfgErrs.length) throw new Error(`Risk config invalid: ${riskCfgErrs.join(", ")}`);

requireEnv(["ORACLE_SECRET"]);
requireAnyLLMKey();

const ORACLE = getOracleWallet();
const ORACLE_ADDR = ORACLE.address;
const ORACLE_PAYER = payingWalletFor(ORACLE);

type ClaimOnChain = ClaimData;
type OracleVerdict = VerdictPayload;

async function fetchClaim(claimId: number): Promise<ClaimOnChain | null> {
  try {
    return await readClaimRaw(claimId);
  } catch (e) {
    // dependency-failure: RPC down
    riskManager.recordFailure("dependency");
    return null;
  }
}

interface EvidenceResult {
  text: string;
  fetcher: EvidenceFetcherKind | "none";
  sourceUrl?: string;
  fetchedAt?: number;
  payment?: EvidencePayment;
}

function evidenceBudgetUsdc(claim: ClaimOnChain): number {
  const potUsdc = claim.total_pot;
  const fraction = (potUsdc * EVIDENCE_POOL_BPS) / BPS_DIVISOR;
  return Math.min(EVIDENCE_MAX_USDC, Math.max(EVIDENCE_MIN_USDC, fraction));
}

async function fetchEvidence(claim: ClaimOnChain): Promise<EvidenceResult> {
  const url = claim.resolution_url;
  if (!url?.startsWith("http")) return { text: "(No resolution URL provided)", fetcher: "none" };
  const budgetUsdc = evidenceBudgetUsdc(claim);
  const maxUnits = usdcToUnits(budgetUsdc);
  const paidFetch = PAY_EVIDENCE ? async (u: string, init?: RequestInit) => {
    const r = await fetchWithBudget(u, ORACLE_PAYER, maxUnits, init);
    return { response: r.response, payment: r.payment ? { priceUnits: r.payment.priceUnits.toString(), txHash: r.payment.txHash } : null };
  } : undefined;
  try {
    const snap = await fetchEvidenceShared(url, { maxChars: MAX_CONTENT_CHARS, userAgent: "Mimir-Oracle/1.0", paidFetch });
    return { text: snap.text, fetcher: snap.fetcher, sourceUrl: snap.sourceUrl, fetchedAt: snap.fetchedAt, payment: snap.payment };
  } catch (err: any) {
    const msg = err instanceof EvidenceFetchError ? err.message : (err?.message ?? "unknown");
    return { text: `(Failed to fetch: ${msg})`, fetcher: "none" };
  }
}

async function evaluateClaim(claim: ClaimOnChain, evidence: string, jurorHistory: string[] = []): Promise<OracleVerdict> {
  const deadlineDate = new Date(claim.deadline * 1000).toISOString();
  const nowDate = new Date().toISOString();
  const potUsdc = claim.total_pot;
  const jurySection = jurorHistory.length > 0 ? `\n## Council juror reports\n${fenceUntrusted("juror-reports", jurorHistory.map((r,i)=>`${i+1}. ${r}`).join("\n"))}\n` : "";
  const claimBlock = fenceUntrusted("claim", [`Question: ${claim.question}`, `Creator position (Side A): ${claim.creator_position}`, `Challenger position (Side B): ${claim.counter_position}`, `Category: ${claim.category}`, `Market type: ${claim.market_type}`, claim.handicap_line ? `Handicap: ${claim.handicap_line}` : null, `Settlement rule: ${claim.settlement_rule || "Use the linked source"}`, `Resolution URL: ${claim.resolution_url}`].filter(Boolean).join("\n"));
  const prompt = `You are Mimir, impartial AI oracle.\n${INJECTION_GUARD}\n## Time\n- Current UTC: ${nowDate}\n- Deadline: ${deadlineDate}\n- Pot: ${potUsdc.toFixed(2)} USDC\n## Claim\n${claimBlock}\n## Evidence\n${fenceUntrusted("web-evidence", evidence)}\n${jurySection}\nReturn JSON only: {"verdict":"CREATOR_WINS"|"CHALLENGERS_WIN"|"DRAW"|"UNRESOLVABLE","confidence":<0-100>,"explanation":"<one paragraph>"}`;
  const { result, lastRawText, attempts } = await parseLLMVerdictWithRetry({
    extractor: extractJson,
    buildPrompt: (attempt) => attempt === 1 ? prompt : `${prompt}${VERDICT_RETRY_SUFFIX}`,
    callLLMFn: (p) => throttledLLM(p, { maxTokens: 1024, jsonOnly: true, model: pickGeminiModel("oracle"), jsonSchema: VERDICT_LLM_SCHEMA }),
  });
  if (!result.ok) throw new Error(`Oracle verdict ${result.reason} after ${attempts} attempts: ${result.detail} — raw: ${lastRawText.slice(0,200)}`);
  return result.payload;
}

function verdictToSide(v: OracleVerdict["verdict"]): "creator" | "challengers" | "draw" | "unresolvable" {
  switch(v){ case "CREATOR_WINS": return "creator"; case "CHALLENGERS_WIN": return "challengers"; case "DRAW": return "draw"; case "UNRESOLVABLE": return "unresolvable"; }
}
const KELLY_CAP = 0.25;
const CONFIDENCE_HIGH_MIN = 80;
const CONFIDENCE_MED_MIN = 60;
function tierVerdict(verdict: OracleVerdict): OracleVerdict {
  if (verdict.verdict === "UNRESOLVABLE" || verdict.verdict === "DRAW") return verdict;
  if (verdict.confidence >= CONFIDENCE_HIGH_MIN) return verdict;
  if (verdict.confidence >= CONFIDENCE_MED_MIN) return { ...verdict, explanation: `[CONTESTED] ${verdict.explanation}`.slice(0,500) };
  return { verdict: "UNRESOLVABLE", confidence: verdict.confidence, explanation: `[LOW CONFIDENCE — refunded] ${verdict.explanation}`.slice(0,500) };
}
const MAX_CONFIDENCE_NON_API = 75;
function applyFetcherTrust(verdict: OracleVerdict, fetcher: EvidenceFetcherKind | "none"): OracleVerdict {
  if (fetcher === "coingecko-api") return verdict;
  if (verdict.verdict === "UNRESOLVABLE") return verdict;
  const cappedConfidence = Math.min(verdict.confidence, MAX_CONFIDENCE_NON_API);
  const tag = fetcher === "jina" ? "[via-jina]" : fetcher === "direct" ? "
