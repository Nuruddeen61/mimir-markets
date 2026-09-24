/**
 * Oracle Risk Manager — bounds autonomous challenge risk
 * Issue #111: feat(agents): bound autonomous challenge risk
 *
 * Goals:
 * - Preserve funded-state safety
 * - Clear operational boundaries
 * - Explicit handling: malformed, stale, duplicate, cancelled, paused, dependency-failure
 */

import type { ClaimData } from "./contract";

// ── Config ──────────────────────────────────────────────────────────
export interface RiskConfig {
  maxDailyChallengeUsdc: number; // e.g. 20 USDC per day
  maxTotalExposureUsdc: number; // e.g. 100 USDC total active stakes
  maxConcurrentChallenges: number; // e.g. 5 open challenges
  maxStakePerClaimUsdc: number; // e.g. 10 USDC
  minStakePerClaimUsdc: number; // e.g. 1 USDC
  cooldownOnFailureMs: number; // e.g. 5min after 3 failures
  maxFailuresBeforeCooldown: number;
  staleThresholdMs: number; // evidence older than this = stale
}

export const DEFAULT_RISK_CONFIG: RiskConfig = {
  maxDailyChallengeUsdc: Number(process.env.ORACLE_MAX_DAILY_USDC?? "20"),
  maxTotalExposureUsdc: Number(process.env.ORACLE_MAX_EXPOSURE_USDC?? "100"),
  maxConcurrentChallenges: Number(process.env.ORACLE_MAX_CONCURRENT?? "10"),
  maxStakePerClaimUsdc: Number(process.env.ORACLE_MAX_STAKE_PER_CLAIM?? "10"),
  minStakePerClaimUsdc: Number(process.env.ORACLE_MIN_STAKE_PER_CLAIM?? "1"),
  cooldownOnFailureMs: Number(process.env.ORACLE_COOLDOWN_MS?? "300000"),
  maxFailuresBeforeCooldown: Number(process.env.ORACLE_MAX_FAILURES?? "3"),
  staleThresholdMs: Number(process.env.ORACLE_STALE_MS?? "3600000"), // 1h
};

export function validateRiskConfig(c: RiskConfig): string[] {
  const errs: string[] = [];
  if (c.maxDailyChallengeUsdc <= 0) errs.push("maxDailyChallengeUsdc must be >0");
  if (c.maxTotalExposureUsdc <= 0) errs.push("maxTotalExposureUsdc must be >0");
  if (c.maxConcurrentChallenges <= 0) errs.push("maxConcurrentChallenges must be >0");
  if (c.maxStakePerClaimUsdc < c.minStakePerClaimUsdc) errs.push("maxStake < minStake");
  if (c.minStakePerClaimUsdc < 1) errs.push("minStake must be >=1 USDC (contract MIN_STAKE)");
  if (c.cooldownOnFailureMs < 0) errs.push("cooldown negative");
  return errs;
}

// ── Failure tracking / circuit breaker ──────────────────────────────
export type FailureReason = "dependency" | "malformed" | "stale" | "duplicate" | "cancelled" | "paused" | "exposure" | "other";

export class RiskManager {
  private dailySpent = 0;
  private dailyResetAt: number;
  private exposure = 0; // active stakes we initiated
  private concurrent = 0;
  private failures = 0;
  private cooldownUntil = 0;
  private challengedIds = new Set<number>();
  private evaluatedIds = new Set<number>();

  constructor(public config: RiskConfig = DEFAULT_RISK_CONFIG) {
    const errs = validateRiskConfig(config);
    if (errs.length) throw new Error(`Invalid risk config: ${errs.join("; ")}`);
    this.dailyResetAt = this.nextMidnight();
  }

  private nextMidnight() {
    const d = new Date();
    d.setUTCDate(d.getUTCDate() + 1);
    d.setUTCHours(0,0,0,0);
    return d.getTime();
  }

  private maybeResetDaily() {
    if (Date.now() >= this.dailyResetAt) {
      this.dailySpent = 0;
      this.dailyResetAt = this.nextMidnight();
      console.log("[risk] Daily budget reset");
    }
  }

  isInCooldown(): boolean {
    return Date.now() < this.cooldownUntil;
  }

  // ── Claim validation ─────────────────────────────────────────────
  validateClaim(claim: ClaimData | null): { ok: boolean; reason?: FailureReason; detail?: string } {
    if (!claim) return { ok: false, reason: "malformed", detail: "null claim" };
    if (!claim.question || claim.question.trim().length < 5) return { ok: false, reason: "malformed", detail: "question too short" };
    if (!claim.resolution_url ||!claim.resolution_url.startsWith("http")) return { ok: false, reason: "malformed", detail: "invalid resolution_url" };
    if (claim.deadline <= 0) return { ok: false, reason: "malformed", detail: "invalid deadline" };
    if (claim.state === "cancelled") return { ok: false, reason: "cancelled", detail: `claim #${claim.id} cancelled` };
    if (claim.state === "resolved") return { ok: false, reason: "cancelled", detail: `claim #${claim.id} already resolved` };
    // paused: treat as cancelled if not open/active (future-proof)
    if ((claim.state as string) === "paused") return { ok: false, reason: "paused", detail: `claim #${claim.id} paused` };
    if (this.challengedIds.has(claim.id)) return { ok: false, reason: "duplicate", detail: `claim #${claim.id} already challenged this session` };
    return { ok: true };
  }

  // ── Exposure checks ──────────────────────────────────────────────
  canChallenge(stakeUsdc: number): { ok: boolean; reason?: FailureReason; detail?: string } {
    this.maybeResetDaily();
    if (this.isInCooldown()) return { ok: false, reason: "other", detail: `cooldown until ${new Date(this.cooldownUntil).toISOString()}` };
    if (stakeUsdc < this.config.minStakePerClaimUsdc) return { ok: false, reason: "malformed", detail: `stake ${stakeUsdc} < min ${this.config.minStakePerClaimUsdc}` };
    if (stakeUsdc > this.config.maxStakePerClaimUsdc) return { ok: false, reason: "exposure", detail: `stake ${stakeUsdc} > max per claim ${this.config.maxStakePerClaimUsdc}` };
    if (this.dailySpent + stakeUsdc > this.config.maxDailyChallengeUsdc) return { ok: false, reason: "exposure", detail: `daily limit: ${this.dailySpent}+${stakeUsdc} > ${this.config.maxDailyChallengeUsdc}` };
    if (this.exposure + stakeUsdc > this.config.maxTotalExposureUsdc) return { ok: false, reason: "exposure", detail: `total exposure: ${this.exposure}+${stakeUsdc} > ${this.config.maxTotalExposureUsdc}` };
    if (this.concurrent >= this.config.maxConcurrentChallenges) return { ok: false, reason: "exposure", detail: `concurrent ${this.concurrent} >= ${this.config.maxConcurrentChallenges}` };
    return { ok: true };
  }

  // ── Bookkeeping ──────────────────────────────────────────────────
  recordChallenge(claimId: number, stakeUsdc: number) {
    this.dailySpent += stakeUsdc;
    this.exposure += stakeUsdc;
    this.concurrent += 1;
    this.challengedIds.add(claimId);
    this.evaluatedIds.add(claimId);
  }

  recordSettled(claimId: number, stakeUsdc: number) {
    // when our challenged claim resolves, reduce exposure
    this.exposure = Math.max(0, this.exposure - stakeUsdc);
    this.concurrent = Math.max(0, this.concurrent - 1);
  }

  recordEvaluated(claimId: number) {
    this.evaluatedIds.add(claimId);
  }

  recordFailure(reason: FailureReason) {
    if (reason === "dependency" || reason === "other") {
      this.failures += 1;
      if (this.failures >= this.config.maxFailuresBeforeCooldown) {
        this.cooldownUntil = Date.now() + this.config.cooldownOnFailureMs;
        console.warn(`[risk] Circuit breaker tripped — cooldown ${this.config.cooldownOnFailureMs}ms (failures=${this.failures})`);
        this.failures = 0;
      }
    }
  }

  resetFailures() { this.failures = 0; }

  isDuplicate(claimId: number): boolean {
    return this.challengedIds.has(claimId) || this.evaluatedIds.has(claimId);
  }

  getStats() {
    return {
      dailySpent: this.dailySpent,
      dailyResetAt: this.dailyResetAt,
      exposure: this.exposure,
      concurrent: this.concurrent,
      inCooldown: this.isInCooldown(),
      cooldownUntil: this.cooldownUntil,
    };
  }
}

// ── Evidence staleness check ───────────────────────────────────────
export function isEvidenceStale(fetchedAt?: number, staleThresholdMs = DEFAULT_RISK_CONFIG.staleThresholdMs): boolean {
  if (!fetchedAt) return true;
  return Date.now() - fetchedAt > staleThresholdMs;
}
