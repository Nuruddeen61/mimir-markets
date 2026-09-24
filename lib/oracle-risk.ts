/**
 * Oracle Risk Manager — Issue #111
 * Bounds autonomous challenge risk, preserves funded-state safety
 */
import type { ClaimData } from "./contract";

export interface RiskConfig {
  maxDailyChallengeUsdc: number;
  maxTotalExposureUsdc: number;
  maxConcurrentChallenges: number;
  maxStakePerClaimUsdc: number;
  minStakePerClaimUsdc: number;
  cooldownOnFailureMs: number;
  maxFailuresBeforeCooldown: number;
  staleThresholdMs: number;
}

export const DEFAULT_RISK_CONFIG: RiskConfig = {
  maxDailyChallengeUsdc: Number(process.env.ORACLE_MAX_DAILY_USDC?? "20"),
  maxTotalExposureUsdc: Number(process.env.ORACLE_MAX_EXPOSURE_USDC?? "100"),
  maxConcurrentChallenges: Number(process.env.ORACLE_MAX_CONCURRENT?? "10"),
  maxStakePerClaimUsdc: Number(process.env.ORACLE_MAX_STAKE_PER_CLAIM?? "10"),
  minStakePerClaimUsdc: Number(process.env.ORACLE_MIN_STAKE_PER_CLAIM?? "1"),
  cooldownOnFailureMs: Number(process.env.ORACLE_COOLDOWN_MS?? "300000"),
  maxFailuresBeforeCooldown: Number(process.env.ORACLE_MAX_FAILURES?? "3"),
  staleThresholdMs: Number(process.env.ORACLE_STALE_MS?? "3600000"),
};

export function validateRiskConfig(c: RiskConfig): string[] {
  const errs: string[] = [];
  if (c.maxDailyChallengeUsdc <= 0) errs.push("maxDailyChallengeUsdc >0");
  if (c.maxTotalExposureUsdc <= 0) errs.push("maxTotalExposureUsdc >0");
  if (c.maxConcurrentChallenges <= 0) errs.push("maxConcurrentChallenges >0");
  if (c.maxStakePerClaimUsdc < c.minStakePerClaimUsdc) errs.push("maxStake < minStake");
  if (c.minStakePerClaimUsdc < 1) errs.push("minStake >=1 USDC");
  if (c.cooldownOnFailureMs < 0) errs.push("cooldown negative");
  return errs;
}

export type FailureReason = "dependency" | "malformed" | "stale" | "duplicate" | "cancelled" | "paused" | "exposure" | "other";

export class RiskManager {
  private dailySpent = 0;
  private dailyResetAt: number;
  private exposure = 0;
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
    const d = new Date(); d.setUTCDate(d.getUTCDate()+1); d.setUTCHours(0,0,0,0);
    return d.getTime();
  }
  private maybeResetDaily() {
    if (Date.now() >= this.dailyResetAt) {
      this.dailySpent = 0;
      this.dailyResetAt = this.nextMidnight();
      console.log("[risk] Daily budget reset");
    }
  }
  isInCooldown(): boolean { return Date.now() < this.cooldownUntil; }

  validateClaim(claim: ClaimData | null): { ok: boolean; reason?: FailureReason; detail?: string } {
    if (!claim) return { ok: false, reason: "malformed", detail: "null claim" };
    if (!claim.question || claim.question.trim().length < 5) return { ok: false, reason: "malformed", detail: "question too short" };
    if (!claim.resolution_url ||!claim.resolution_url.startsWith("http")) return { ok: false, reason: "malformed", detail: "invalid resolution_url" };
    if (claim.deadline <= 0) return { ok: false, reason: "malformed", detail: "invalid deadline" };
    if (claim.state === "cancelled") return { ok: false, reason: "cancelled", detail: `claim #${claim.id} cancelled` };
    if (claim.state === "resolved") return { ok: false, reason: "cancelled", detail: `claim #${claim.id} resolved` };
    if ((claim.state as string) === "paused") return { ok: false, reason: "paused", detail: `claim #${claim.id} paused` };
    if (this.challengedIds.has(claim.id)) return { ok: false, reason: "duplicate", detail: `claim #${claim.id} already challenged` };
    return { ok: true };
  }

  canChallenge(stakeUsdc: number): { ok: boolean; reason?: FailureReason; detail?: string } {
    this.maybeResetDaily();
    if (this.isInCooldown()) return { ok: false, reason: "other", detail: `cooldown until ${new Date(this.cooldownUntil).toISOString()}` };
    if (stakeUsdc < this.config.minStakePerClaimUsdc) return { ok: false, reason: "malformed", detail: `stake ${stakeUsdc} < min` };
    if (stakeUsdc > this.config.maxStakePerClaimUsdc) return { ok: false, reason: "exposure", detail: `stake ${stakeUsdc} > max per claim` };
    if (this.dailySpent + stakeUsdc > this.config.maxDailyChallengeUsdc) return { ok: false, reason: "exposure", detail: `daily limit exceeded` };
    if (this.exposure + stakeUsdc > this.config.maxTotalExposureUsdc) return { ok: false, reason: "exposure", detail: `total exposure exceeded` };
    if (this.concurrent >= this.config.maxConcurrentChallenges) return { ok: false, reason: "exposure", detail: `concurrent limit` };
    return { ok: true };
  }

  recordChallenge(claimId: number, stakeUsdc: number) {
    this.dailySpent += stakeUsdc;
    this.exposure += stakeUsdc;
    this.concurrent += 1;
    this.challengedIds.add(claimId);
    this.evaluatedIds.add(claimId);
  }
  recordSettled(stakeUsdc: number) {
    this.exposure = Math.max(0, this.exposure - stakeUsdc);
    this.concurrent = Math.max(0, this.concurrent - 1);
  }
  recordEvaluated(claimId: number) { this.evaluatedIds.add(claimId); }
  recordFailure(reason: FailureReason) {
    if (reason === "dependency" || reason === "other") {
      this.failures += 1;
      if (this.failures >= this.config.maxFailuresBeforeCooldown) {
        this.cooldownUntil = Date.now() + this.config.cooldownOnFailureMs;
        console.warn(`[risk] Circuit breaker tripped — cooldown ${this.config.cooldownOnFailureMs}ms`);
        this.failures = 0;
      }
    }
  }
  resetFailures() { this.failures = 0; }
  isDuplicate(id: number) { return this.challengedIds.has(id) || this.evaluatedIds.has(id); }
  getStats() {
    return { dailySpent: this.dailySpent, exposure: this.exposure, concurrent: this.concurrent, inCooldown: this.isInCooldown() };
  }
}

export function isEvidenceStale(fetchedAt?: number, thresholdMs = DEFAULT_RISK_CONFIG.staleThresholdMs): boolean {
  if (!fetchedAt) return true;
  return Date.now() - fetchedAt > thresholdMs;
}
