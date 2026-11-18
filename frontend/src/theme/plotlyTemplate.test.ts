import { describe, expect, it } from "vitest";
import { chartLayout, type Tokens } from "./plotlyTemplate";

const T: Tokens = {
  sheet: "#fff",
  sheetRaised: "#fff",
  ink: "#000",
  inkMuted: "#888",
  grid: "#ccc",
  gridMinor: "#eee",
  rule: "#bbb",
  layoutBlue: "#00f",
  danger: "#f00",
  fontUi: "Barlow",
  rev: {},
};

describe("chartLayout hover axis", () => {
  const layout = chartLayout(T, { xTitle: "x", yTitle: "y", legend: false, yZero: false, xHover: { label: "Wheel travel", unit: "mm" } });
  const x2 = layout["xaxis2"] as Record<string, unknown>;

  it("keeps the hidden hover axis locked to the visible x axis", () => {
    expect(x2["matches"]).toBe("x");
    expect(x2["overlaying"]).toBe("x");
    expect(x2["showticklabels"]).toBe(false);
  });

  it("puts label and unit on the hover value and snaps spikes to data (linked hover has no cursor)", () => {
    expect(x2["tickprefix"]).toBe("Wheel travel ");
    expect(x2["ticksuffix"]).toBe(" mm");
    expect(x2["spikesnap"]).toBe("data");
  });

  it("has no hover axis without a heading", () => {
    expect(chartLayout(T, { xTitle: "x", yTitle: "y", legend: false, yZero: false })["xaxis2"]).toBeUndefined();
  });
});
