import { describe, expect, it } from "vitest";
import { formatDelta, formatValue, lookup, MINUS } from "./format";

describe("formatValue", () => {
  it("scales by the display factor and fixes the decimals", () => {
    expect(formatValue(0.0123456, 1000, 2)).toBe("12.35");
    expect(formatValue(2, 1, 0)).toBe("2");
  });

  it("shows null and non-finite as a dash", () => {
    expect(formatValue(null, 1, 2)).toBe("-");
    expect(formatValue(undefined, 1, 2)).toBe("-");
    expect(formatValue(Number.NaN, 1, 2)).toBe("-");
    expect(formatValue(Number.POSITIVE_INFINITY, 1, 2)).toBe("-");
  });

  it("uses a typographic minus", () => {
    expect(formatValue(-1.5, 1, 1)).toBe(`${MINUS}1.5`);
  });

  it("drops the sign of a value that rounds to zero", () => {
    expect(formatValue(-0.0004, 1, 2)).toBe("0.00");
    expect(formatValue(-0, 1, 1)).toBe("0.0");
    expect(formatValue(-0.04, 1, 0)).toBe("0");
  });
});

describe("formatDelta", () => {
  it("is a plain signed number", () => {
    expect(formatDelta(1.5, 1, 1, 2)).toBe("+0.50");
    expect(formatDelta(1, 1.5, 1, 2)).toBe(`${MINUS}0.50`);
  });

  it("has no sign when the difference rounds to zero", () => {
    expect(formatDelta(1.0001, 1, 1, 2)).toBe("0.00");
    expect(formatDelta(0.9999, 1, 1, 2)).toBe("0.00");
  });

  it("is a dash when either side is missing", () => {
    expect(formatDelta(null, 1, 1, 2)).toBe("-");
    expect(formatDelta(1, null, 1, 2)).toBe("-");
  });
});

describe("lookup", () => {
  const s = { a: { b: 1.5, c: null, d: "x" }, n: Number.NaN };
  it("walks nested paths and only returns finite numbers", () => {
    expect(lookup(s, ["a", "b"])).toBe(1.5);
    expect(lookup(s, ["a", "c"])).toBeNull();
    expect(lookup(s, ["a", "d"])).toBeNull();
    expect(lookup(s, ["n"])).toBeNull();
    expect(lookup(s, ["a", "zzz", "q"])).toBeNull();
    expect(lookup(undefined, ["a"])).toBeNull();
  });
});
