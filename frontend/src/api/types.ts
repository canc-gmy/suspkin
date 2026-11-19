import type { components } from "./schema";

type S = components["schemas"];

export type DatasetInfo = S["DatasetOut"];
export type DatasetPatch = S["DatasetPatch"];
export type Health = S["HealthOut"];
export type TableData = S["TableOut"];
export type HistoryData = S["HistoryOut"];
export type BrowseData = S["BrowseOut"];
export type BrowseEntry = S["BrowseEntry"];
export type PoseInput = S["PoseInput"];
export type Hardpoint = S["HardpointOut"];
export type ExportResult = S["ExportOut"];
export type HardpointEdit = S["HardpointEdit"];
export type DeriveRequest = S["DeriveRequest"];
export type SaveRequest = S["SaveRequest"];
export type SaveResult = S["SaveOut"];
export type Region = S["RegionOut"];
export type DatasetKind = DatasetInfo["kind"];

export type Summary = Record<string, unknown>;

export interface ScenePose {
  heave?: number;
  roll?: number;
  pitch?: number;
  rack?: number;
}
