import { createContext, useContext } from "react";
import { keepPreviousData, useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import type { DataSource } from "./source";
import type { DatasetPatch, ScenePose } from "./types";

export const SourceContext = createContext<DataSource | null>(null);

/** The data source chosen at start-up. */
export function useSource(): DataSource {
  const s = useContext(SourceContext);
  if (!s) throw new Error("useSource needs a SourceContext provider");
  return s;
}

const FOREVER = { staleTime: Infinity, gcTime: Infinity } as const;

export function useHealth() {
  const src = useSource();
  return useQuery({ queryKey: ["health"], queryFn: () => src.getHealth(), ...FOREVER });
}

export function useFigures() {
  const src = useSource();
  return useQuery({ queryKey: ["figures"], queryFn: () => src.getFigures(), ...FOREVER });
}

export function useDatasets() {
  const src = useSource();
  return useQuery({ queryKey: ["datasets"], queryFn: () => src.listDatasets(), staleTime: 5_000 });
}

/** One study table per (dataset, study) pair; cached for the session. */
export function useTables(refs: { id: string; study: string }[]) {
  const src = useSource();
  return useQueries({
    queries: refs.map((r) => ({
      queryKey: ["table", r.id, r.study],
      queryFn: () => src.getTable(r.id, r.study),
      ...FOREVER,
    })),
  });
}

export function useSummaries(ids: string[]) {
  const src = useSource();
  return useQueries({
    queries: ids.map((id) => ({
      queryKey: ["summary", id],
      queryFn: () => src.getSummary(id),
      ...FOREVER,
    })),
  });
}

export function useBrowse(dir: string, enabled = true) {
  const src = useSource();
  return useQuery({
    queryKey: ["browse", dir],
    queryFn: () => src.browse(dir),
    enabled,
    gcTime: 0,
  });
}

export function useHistory(runId: string | null) {
  const src = useSource();
  return useQuery({
    queryKey: ["history", runId],
    queryFn: () => src.getHistory(runId as string),
    enabled: runId !== null,
    ...FOREVER,
  });
}

export function useScene(id: string | null, pose: ScenePose) {
  const src = useSource();
  return useQuery({
    queryKey: ["scene", id, pose],
    queryFn: () => src.getScene(id as string, pose),
    enabled: id !== null,
    staleTime: 60_000,
    placeholderData: keepPreviousData,
    retry: false,
  });
}

export function useOpenPath() {
  const src = useSource();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (path: string) => src.openPath(path),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["datasets"] }),
  });
}

export function usePatchDataset() {
  const src = useSource();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (a: { id: string; patch: DatasetPatch }) => src.patchDataset(a.id, a.patch),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["datasets"] }),
  });
}

export function useCloseDataset() {
  const src = useSource();
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => src.closeDataset(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["datasets"] }),
  });
}

export function useExportSnapshot() {
  const src = useSource();
  return useMutation({ mutationFn: (out?: string) => src.exportSnapshot(out) });
}
