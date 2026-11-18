import { describe, expect, it } from "vitest";
import { chartLayout, cursorShape, type Tokens } from "./plotlyTemplate";

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
const base = { xTitle: "x", yTitle: "y", legend: false, yZero: false };

describe("chartLayout", () => {
  it("has no hover labels and no hidden hover axis on curve panels", () => {
    const layout = chartLayout(T, base);
    expect(layout["hovermode"]).toBe(false);
    expect(layout["xaxis2"]).toBeUndefined();
    expect((layout["xaxis"] as Record<string, unknown>)["showspikes"]).toBeUndefined();
  });

  it("keeps hover labels for scatter charts", () => {
    expect(chartLayout(T, { ...base, hover: "closest" })["hovermode"]).toBe("closest");
  });
});

describe("cursorShape", () => {
  it("is a dashed muted line while following and a solid blue line when pinned", () => {
    const follow = cursorShape(T, 3, false);
    const pinned = cursorShape(T, 3, true);
    expect(follow["x0"]).toBe(3);
    expect(follow["x1"]).toBe(3);
    expect(follow["yref"]).toBe("paper");
    expect(follow["line"]).toMatchObject({ color: "#888", dash: "dash", width: 1 });
    expect(pinned["line"]).toMatchObject({ color: "#00f", dash: "solid", width: 1 });
  });
});
