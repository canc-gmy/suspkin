import { useCallback, useEffect, useMemo, useRef, useState, type Dispatch } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { ApiError } from "../api";
import { useSource } from "../api/hooks";
import { buildEdits, type EditAction, type EditModel, indexBase } from "../lib/hardpointEdit";
import type { Hardpoint } from "../api/types";
import { isAbort, LatestOnly } from "../lib/solveSequence";

/** Quiet time after the last edit before Auto-solve starts a solve. */
export const AUTO_SOLVE_MS = 600;

/** Message under the table for a failed solve: what happened, and what to do for a bad geometry. */
export function solveErrorText(e: unknown): string {
  if (e instanceof ApiError) {
    const hint = e.status === 400 ? " Change the marked values and solve again." : "";
    return `Could not solve: ${e.message.replace(/\.$/, "")}.${hint}`;
  }
  return `Could not solve: ${e instanceof Error ? e.message : "unknown error"}.`;
}

/**
 * Turns the edits of the Hardpoints tab into revisions: "Solve" makes a new one, "Auto-solve"
 * re-solves 600 ms after the last edit and keeps a single auto-solved revision. Only the newest
 * request counts; older answers are aborted or ignored.
 */
export function useHardpointSolver(edit: EditModel, points: readonly Hardpoint[] | null, dispatch: Dispatch<EditAction>) {
  const src = useSource();
  const qc = useQueryClient();
  const [latest] = useState(() => new LatestOnly());
  const autoId = useRef<string | null>(null);
  const [auto, setAuto] = useState(false);
  const [solving, setSolving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const base = useMemo(() => (points ? indexBase(points) : null), [points]);
  const edits = useMemo(() => (base ? buildEdits(edit, base) : []), [edit, base]);
  const sourceId = edit?.sourceId ?? null;

  const run = useCallback(
    async (isAuto: boolean) => {
      if (!sourceId || edits.length === 0) return;
      const { seq, signal } = latest.begin();
      setSolving(true);
      setError(null);
      setStatus(null);
      try {
        const d = await src.deriveDataset(sourceId, { edits }, signal);
        if (!latest.isCurrent(seq)) return;
        const prev = autoId.current;
        autoId.current = isAuto ? d.id : null;
        if (isAuto && prev && prev !== d.id) await src.closeDataset(prev).catch(() => undefined);
        await qc.invalidateQueries({ queryKey: ["datasets"] });
        if (!latest.isCurrent(seq)) return;
        if (isAuto) {
          setStatus(`Solved as ${d.label}`);
        } else {
          setStatus(`Created ${d.label}`);
          dispatch({ type: "discard" });
        }
      } catch (e) {
        if (isAbort(e) || !latest.isCurrent(seq)) return;
        setError(solveErrorText(e));
      } finally {
        if (latest.isCurrent(seq)) setSolving(false);
      }
    },
    [sourceId, edits, latest, src, qc, dispatch],
  );

  // Auto-solve: any edit cancels the running solve and restarts the 600 ms wait
  useEffect(() => {
    if (!auto || !sourceId) return;
    latest.cancel();
    if (edits.length === 0) {
      setSolving(false);
      return;
    }
    setSolving(true);
    const t = setTimeout(() => void run(true), AUTO_SOLVE_MS);
    return () => clearTimeout(t);
  }, [auto, sourceId, edits, run, latest]);

  useEffect(() => () => latest.cancel(), [latest]);

  const discard = useCallback(() => {
    latest.cancel();
    setSolving(false);
    setError(null);
    autoId.current = null;
    dispatch({ type: "discard" });
  }, [latest, dispatch]);

  const changeAuto = (on: boolean) => {
    setAuto(on);
    if (!on) {
      latest.cancel(); // a pending auto solve is dropped; the last result stays as an ordinary revision
      setSolving(false);
      autoId.current = null;
    }
  };

  return { base, edits, auto, setAuto: changeAuto, solving, error, status, solve: () => void run(false), discard };
}

/** Save an edited revision as a YAML file, asking before an existing file is replaced. */
export function useSaveYaml() {
  const src = useSource();
  const [saved, setSaved] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const save = useCallback(
    async (id: string, name: string, overwrite = false): Promise<boolean> => {
      setError(null);
      setSaved(null);
      try {
        const r = await src.saveDataset(id, { path: name, overwrite });
        setSaved(`Saved to ${r.path}`);
        return true;
      } catch (e) {
        const m = e instanceof ApiError && e.status === 409 ? /^already exists: (.+)$/.exec(e.message) : null;
        if (m && !overwrite) {
          return window.confirm(`Replace ${m[1]}?`) ? save(id, name, true) : false;
        }
        setError(`Could not save ${name}: ${e instanceof Error ? e.message : "unknown error"}`);
        return false;
      }
    },
    [src],
  );

  return { save, saved, error, clear: () => (setSaved(null), setError(null)) };
}
