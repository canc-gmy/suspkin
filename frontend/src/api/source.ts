import type {
  BrowseData,
  DatasetInfo,
  DatasetPatch,
  ExportResult,
  Health,
  HistoryData,
  ScenePose,
  Summary,
  TableData,
} from "./types";
import type { SpecDoc } from "./spec";

/** Everything the UI needs from a backend: a live server or an embedded snapshot. */
export interface DataSource {
  readonly kind: "http" | "snapshot";
  /** True for an embedded snapshot: no opening, renaming, closing, exporting or pose changes. */
  readonly readOnly: boolean;
  getHealth(): Promise<Health>;
  getFigures(): Promise<SpecDoc>;
  listDatasets(): Promise<DatasetInfo[]>;
  openPath(path: string): Promise<DatasetInfo[]>;
  patchDataset(id: string, patch: DatasetPatch): Promise<DatasetInfo>;
  closeDataset(id: string): Promise<void>;
  getTable(id: string, study: string): Promise<TableData>;
  getSummary(id: string): Promise<Summary>;
  getScene(id: string, pose?: ScenePose): Promise<Record<string, unknown>>;
  getHistory(runId: string): Promise<HistoryData>;
  browse(dir: string): Promise<BrowseData>;
  /** Write a snapshot of the open revisions into the workspace. */
  exportSnapshot(out?: string): Promise<ExportResult>;
}

/** An API failure with a message that is safe to show under the control that caused it. */
export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}
