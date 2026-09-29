import { describe, it, expect } from "vitest";
import { RiskManager } from "./oracle-risk";

const limits = {
  maxStakePerClaim: 10,
  dailyLimit: 15,
  totalExposure: 20,
  maxConcurrent: 2,
};

const future = Math.floor(Date.now() / 1000) + 3600;
const claim = (id: number) => ({ id, state: "open", deadline: future });

describe("RiskManager", () => {
  it("rejects malformed, closed and expired claims", () => {
    const r = new RiskManager(limits);
    expect(r.validateClaim(null).ok).toBe(false);
    expect(r.validateClaim({}).ok).toBe(false);
    expect(
      r.validateClaim({ id: 1, state: "resolved", deadline: future }).ok,
    ).toBe(false);
    expect(
      r.validateClaim({ id: 1, state: "open", deadline: 1 }).ok,
    ).toBe(false);
    expect(r.validateClaim(claim(1)).ok).toBe(true);
  });

  it("enforces per-claim cap", () => {
    const r = new RiskManager(limits);
    expect(r.canChallenge(claim(1), 11).ok).toBe(false);
    expect(r.canChallenge(claim(1), 5).ok).toBe(true);
  });

  it("rejects invalid stakes", () => {
    const r = new RiskManager(limits);
    expect(r.canChallenge(claim(1), 0).ok).toBe(false);
    expect(r.canChallenge(claim(1), -1).ok).toBe(false);
    expect(r.canChallenge(claim(1), NaN).ok).toBe(false);
  });

  it("rejects duplicates", () => {
    const r = new RiskManager(limits);
    r.recordChallenge(claim(1), 5);
    expect(r.canChallenge(claim(1), 5).ok).toBe(false);
  });

  it("enforces the concurrent limit", () => {
    const r = new RiskManager(limits);
    r.recordChallenge(claim(1), 5);
    r.recordChallenge(claim(2), 5);
    expect(r.canChallenge(claim(3), 1).ok).toBe(false);
  });

  it("enforces the daily limit even after release", () => {
    const r = new RiskManager(limits);
    r.recordChallenge(claim(1), 5);
    r.recordChallenge(claim(2), 5);
    r.release(1);
    r.release(2);
    r.recordChallenge(claim(4), 5);
    expect(r.canChallenge(claim(5), 6).ok).toBe(false);
  });

  it("frees exposure on release", () => {
    const r = new RiskManager(limits);
    r.recordChallenge(claim(1), 10);
    expect(r.exposure()).toBe(10);
    r.release(1);
    expect(r.exposure()).toBe(0);
  });

  it("enforces total exposure", () => {
    const r = new RiskManager({ ...limits, dailyLimit: 100, maxConcurrent: 5 });
    r.recordChallenge(claim(1), 10);
    r.recordChallenge(claim(2), 10);
    expect(r.canChallenge(claim(3), 1).ok).toBe(false);
  });
});
