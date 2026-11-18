import { HttpSource } from "./httpSource";
import { SnapshotSource } from "./snapshotSource";
import type { DataSource } from "./source";

export * from "./source";
export * from "./types";
export * from "./spec";
export { HttpSource } from "./httpSource";
export { SnapshotSource } from "./snapshotSource";

/** Pick the data source at start-up: embedded snapshot if present, else the HTTP API. */
export function pickSource(): DataSource {
  return window.__SUSPKIN_SNAPSHOT__ ? new SnapshotSource() : new HttpSource();
}
