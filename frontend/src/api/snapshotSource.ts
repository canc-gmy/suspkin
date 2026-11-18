import { ApiError, type DataSource } from "./source";
import type { SpecDoc } from "./spec";
import type { DatasetInfo, ExportResult, Hardpoint, Health, HistoryData, Summary, TableData } from "./types";

/** Shape of the data embedded by the snapshot export (full support arrives with the export). */
export interface Snapshot {
  health: Health;
  figures: SpecDoc;
  datasets: DatasetInfo[];
  tables: Record<string, Record<string, TableData>>;
  summaries: Record<string, Summary>;
  histories?: Record<string, HistoryData>;
  scenes?: Record<string, Record<string, unknown>>;
  hardpoints?: Record<string, Hardpoint[]>;
}

declare global {
  interface Window {
    __SUSPKIN_SNAPSHOT__?: Snapshot;
  }
}

export const READ_ONLY_TIP = "Read-only snapshot";
const READ_ONLY = "This is a read-only snapshot. Open the live app to change revisions.";

/** Read-only source backed by the data embedded in `window.__SUSPKIN_SNAPSHOT__`. */
export class SnapshotSource implements DataSource {
  readonly kind = "snapshot" as const;
  readonly readOnly = true;

  private get snap(): Snapshot {
    const s = window.__SUSPKIN_SNAPSHOT__;
    if (!s) throw new ApiError("No snapshot data is embedded in this page.", 0);
    return s;
  }

  async getHealth(): Promise<Health> {
    return this.snap.health;
  }
  async getFigures(): Promise<SpecDoc> {
    return this.snap.figures;
  }
  async listDatasets(): Promise<DatasetInfo[]> {
    return this.snap.datasets;
  }
  async openPath(): Promise<DatasetInfo[]> {
    throw new ApiError(READ_ONLY, 405);
  }
  async patchDataset(): Promise<DatasetInfo> {
    throw new ApiError(READ_ONLY, 405);
  }
  async closeDataset(): Promise<void> {
    throw new ApiError(READ_ONLY, 405);
  }
  async getTable(id: string, study: string): Promise<TableData> {
    const t = this.snap.tables[id]?.[study];
    if (!t) throw new ApiError(`The snapshot has no ${study} table for this revision.`, 404);
    return t;
  }
  async getSummary(id: string): Promise<Summary> {
    return this.snap.summaries[id] ?? {};
  }
  /** Snapshots hold one static-pose scene per revision; the pose is ignored. */
  async getScene(id: string): Promise<Record<string, unknown>> {
    const s = this.snap.scenes?.[id];
    if (!s) throw new ApiError("The snapshot has no 3D scene for this revision.", 404);
    return s;
  }
  async getHardpoints(id: string): Promise<Hardpoint[]> {
    const h = this.snap.hardpoints?.[id];
    if (!h) throw new ApiError("The snapshot has no hardpoints for this revision.", 404);
    return h;
  }
  async getHistory(runId: string): Promise<HistoryData> {
    const h = this.snap.histories?.[runId];
    if (!h) throw new ApiError("The snapshot has no optimizer history.", 404);
    return h;
  }
  async exportSnapshot(): Promise<ExportResult> {
    throw new ApiError(READ_ONLY, 405);
  }
  async browse(): Promise<never> {
    throw new ApiError(READ_ONLY, 405);
  }
}
