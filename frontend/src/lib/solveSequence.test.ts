import { describe, expect, it } from "vitest";
import { isAbort, LatestOnly } from "./solveSequence";

describe("LatestOnly", () => {
  it("marks the older request stale and aborts it when a newer one starts", () => {
    const l = new LatestOnly();
    const a = l.begin();
    expect(l.isCurrent(a.seq)).toBe(true);
    const b = l.begin();
    expect(a.signal.aborted).toBe(true);
    expect(l.isCurrent(a.seq)).toBe(false);
    expect(l.isCurrent(b.seq)).toBe(true);
    expect(b.signal.aborted).toBe(false);
  });

  it("drops the late answer of a cancelled request, and a later one still counts", async () => {
    const l = new LatestOnly();
    const applied: string[] = [];
    const answer = async (name: string, ms: number) => {
      const { seq } = l.begin();
      await new Promise((r) => setTimeout(r, ms));
      if (l.isCurrent(seq)) applied.push(name);
    };
    // the slow first solve answers after the second one was started: only the second is applied
    await Promise.all([answer("old", 20), answer("new", 5)]);
    expect(applied).toEqual(["new"]);
    const { seq } = l.begin();
    l.cancel();
    expect(l.isCurrent(seq)).toBe(false);
  });

  it("recognises an AbortError", () => {
    expect(isAbort(new DOMException("x", "AbortError"))).toBe(true);
    expect(isAbort(new Error("x"))).toBe(false);
  });
});
