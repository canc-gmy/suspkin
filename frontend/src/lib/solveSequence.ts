/**
 * Keeps only the newest of several overlapping requests: starting one cancels the previous
 * (AbortController) and marks its sequence number stale, so a late answer can be ignored even
 * when the request itself could not be aborted.
 */
export class LatestOnly {
  private seq = 0;
  private ctrl: AbortController | null = null;

  /** Start a request: aborts the previous one. */
  begin(): { seq: number; signal: AbortSignal } {
    this.ctrl?.abort();
    this.ctrl = new AbortController();
    this.seq += 1;
    return { seq: this.seq, signal: this.ctrl.signal };
  }

  /** True while `seq` is the newest request and nothing was cancelled. */
  isCurrent(seq: number): boolean {
    return seq === this.seq && this.ctrl !== null && !this.ctrl.signal.aborted;
  }

  /** Cancel the running request; its answer counts as stale. */
  cancel(): void {
    this.ctrl?.abort();
  }
}

/** A cancelled `fetch` rejects with an AbortError. */
export const isAbort = (e: unknown): boolean => e instanceof DOMException && e.name === "AbortError";
