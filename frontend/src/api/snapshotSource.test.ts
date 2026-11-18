import { afterEach, describe, expect, it } from "vitest";
import { pickSource } from ".";
import { SnapshotSource, type Snapshot } from "./snapshotSource";

const snap: Snapshot = {
  health: { status: "ok", version: "1", workspace: "ws" },
  figures: { schema: 1, palette: [], linestyles: [], figures: [], summary_rows: [] },
  datasets: [{ id: "a", label: "A", kind: "vehicle", source: "a.yaml", color: 0, studies: ["roll"], params: {}, steps: 31, resolution_fixed: false }],
  tables: { a: { roll: { columns: { roll: [0, 1] } } } },
  summaries: { a: { x: 1 } },
  histories: {},
  scenes: { a: { data: [], layout: {} } },
};

afterEach(() => {
  delete window.__SUSPKIN_SNAPSHOT__;
});

describe("SnapshotSource", () => {
  it("is picked when a snapshot is embedded, and reads from it", async () => {
    window.__SUSPKIN_SNAPSHOT__ = snap;
    const src = pickSource();
    expect(src).toBeInstanceOf(SnapshotSource);
    expect(src.readOnly).toBe(true);
    expect((await src.listDatasets())[0]?.id).toBe("a");
    expect((await src.getTable("a", "roll")).columns["roll"]).toEqual([0, 1]);
    expect(await src.getSummary("a")).toEqual({ x: 1 });
    expect(await src.getSummary("zzz")).toEqual({});
    expect(await src.getScene("a", { roll: 0.1 })).toEqual({ data: [], layout: {} });
    expect((await src.getHealth()).workspace).toBe("ws");
  });

  it("reports missing data and refuses every change", async () => {
    window.__SUSPKIN_SNAPSHOT__ = snap;
    const src = new SnapshotSource();
    await expect(src.getTable("a", "pitch")).rejects.toThrow(/no pitch table/);
    await expect(src.getHistory("r")).rejects.toThrow(/no optimizer history/);
    for (const call of [
      () => src.openPath(),
      () => src.patchDataset(),
      () => src.closeDataset(),
      () => src.exportSnapshot(),
      () => src.browse(),
    ]) {
      await expect(call()).rejects.toThrow(/read-only/);
    }
  });

  it("fails clearly without embedded data", async () => {
    await expect(new SnapshotSource().listDatasets()).rejects.toThrow(/No snapshot data/);
  });
});
