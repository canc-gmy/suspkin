import { ApiError, type DataSource } from "./source";
import type { SpecDoc } from "./spec";
import type { DatasetInfo, Health, HistoryData, Summary, TableData } from "./types";

/** Shape of the data embedded by the snapshot export (full support arrives with the export). */
export interface Snapshot {
  health: Health;
  figures: SpecDoc;
  datasets: DatasetInfo[];
  tables: Record<string, Record<string, TableData>>;
  summaries: Record<string, Summary>;
  histories?: Record<string, HistoryData>;
  scenes?: Record<string, Record<string, unknown>>;
}

declare global {
  interface Window {
    __SUSPKIN_SNAPSHOT__?: Snapshot;
  }
}

const READ_ONLY = "This is a read-only snapshot. Open the live app to change revisions.";

/** Read-only source backed by ``window.__SUSPKIN_SNAPSHOT__`` (stub: reads what is embedded). */
export class SnapshotSource implements DataSource {
  readonly kind = "snapshot" as const;

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
  async getScene(id: string): Promise<Record<string, unknown>> {
    const s = this.snap.scenes?.[id];
    if (!s) throw new ApiError("The snapshot has no 3D scene for this revision.", 404);
    return s;
  }
  async getHistory(runId: string): Promise<HistoryData> {
    const h = this.snap.histories?.[runId];
    if (!h) throw new ApiError("The snapshot has no optimizer history.", 404);
    return h;
  }
  async browse(): Promise<never> {
    throw new ApiError(READ_ONLY, 405);
  }
}
