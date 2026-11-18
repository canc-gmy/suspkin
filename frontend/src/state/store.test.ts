import { beforeEach, describe, expect, it } from "vitest";
import { useUi, visibleIds } from "./store";

const s = () => useUi.getState();

beforeEach(() => {
  useUi.setState({ order: [], letters: {}, hidden: {}, reference: null, fresh: {}, synced: false, tab: null, mode: "overlay" });
});

describe("syncDatasets", () => {
  it("assigns letters in open order and defaults the reference to the first", () => {
    s().syncDatasets(["x", "y", "z"]);
    expect(s().letters).toEqual({ x: "A", y: "B", z: "C" });
    expect(s().reference).toBe("x");
    expect(s().fresh).toEqual({}); // first load is not highlighted
  });

  it("keeps letters stable on close, reuses a freed one, and highlights new rows", () => {
    s().syncDatasets(["x", "y", "z"]);
    s().syncDatasets(["x", "z"]);
    expect(s().letters).toEqual({ x: "A", z: "C" });
    s().syncDatasets(["x", "z", "w"]);
    expect(s().letters["w"]).toBe("B");
    expect(s().order).toEqual(["x", "z", "w"]);
    expect(s().fresh).toEqual({ w: true });
    s().clearFresh("w");
    expect(s().fresh).toEqual({});
  });

  it("moves the reference when the reference dataset closes", () => {
    s().syncDatasets(["x", "y"]);
    s().setReference("y");
    expect(s().reference).toBe("y");
    s().syncDatasets(["x"]);
    expect(s().reference).toBe("x");
    s().syncDatasets([]);
    expect(s().reference).toBeNull();
  });
});

describe("visibility and selections", () => {
  it("toggles visibility and lists visible ids in open order", () => {
    s().syncDatasets(["x", "y", "z"]);
    s().toggleVisible("y");
    expect(visibleIds(s())).toEqual(["x", "z"]);
    s().toggleVisible("y");
    expect(visibleIds(s())).toEqual(["x", "y", "z"]);
  });

  it("ignores an unknown reference id", () => {
    s().syncDatasets(["x"]);
    s().setReference("nope");
    expect(s().reference).toBe("x");
  });

  it("sets tab, axle, mode and theme", () => {
    s().setTab("roll");
    s().setAxle("rear");
    s().setMode("delta");
    s().setTheme("dark");
    expect([s().tab, s().axle, s().mode, s().theme]).toEqual(["roll", "rear", "delta", "dark"]);
    s().toggleTheme();
    expect(s().theme).toBe("light");
  });
});
