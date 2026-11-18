import { create } from "zustand";
import { nextLetter, type Theme } from "../theme/revisions";
import type { AxleChoice } from "../lib/panels";

export type Mode = "overlay" | "delta";

export interface UiState {
  /** Open dataset ids in open order. */
  order: string[];
  /** Revision letter per dataset id (stable for the life of the dataset). */
  letters: Record<string, string>;
  hidden: Record<string, true>;
  reference: string | null;
  /** Ids opened after the first sync (their row highlights once). */
  fresh: Record<string, true>;
  synced: boolean;
  tab: string | null;
  axle: AxleChoice;
  mode: Mode;
  theme: Theme;
  /** Draw small markers at the real solver points (kept for the session in localStorage). */
  showPoints: boolean;
  drawerOpen: boolean;
  openDialog: boolean;

  syncDatasets: (ids: readonly string[]) => void;
  toggleVisible: (id: string) => void;
  setReference: (id: string) => void;
  setTab: (tab: string) => void;
  setAxle: (axle: AxleChoice) => void;
  setMode: (mode: Mode) => void;
  setTheme: (theme: Theme) => void;
  setShowPoints: (show: boolean) => void;
  toggleTheme: () => void;
  setDrawer: (open: boolean) => void;
  setOpenDialog: (open: boolean) => void;
  clearFresh: (id: string) => void;
}

const THEME_KEY = "suspkin.theme";
const POINTS_KEY = "suspkin.showPoints";

function initialShowPoints(): boolean {
  try {
    return localStorage.getItem(POINTS_KEY) === "1";
  } catch {
    return false; // storage unavailable
  }
}

export function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    // storage unavailable
  }
  return typeof matchMedia === "function" && matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

/** Tab named in the URL hash (`#optimization`), so a view can be linked or reloaded. */
function initialTab(): string | null {
  const h = typeof location === "undefined" ? "" : location.hash.replace(/^#/, "");
  return h || null;
}

export const useUi = create<UiState>()((set) => ({
  order: [],
  letters: {},
  hidden: {},
  reference: null,
  fresh: {},
  synced: false,
  tab: initialTab(),
  axle: "front",
  mode: "overlay",
  theme: initialTheme(),
  showPoints: initialShowPoints(),
  drawerOpen: false,
  openDialog: false,

  syncDatasets: (ids) =>
    set((s) => {
      const alive = new Set(ids);
      const order = [...s.order.filter((id) => alive.has(id)), ...ids.filter((id) => !s.order.includes(id))];
      const added = order.filter((id) => !(id in s.letters));
      const letters: Record<string, string> = {};
      for (const id of order) if (id in s.letters) letters[id] = s.letters[id] as string;
      for (const id of added) letters[id] = nextLetter(Object.values(letters));
      const hidden: Record<string, true> = {};
      for (const id of order) if (s.hidden[id]) hidden[id] = true;
      const fresh: Record<string, true> = {};
      for (const id of order) if (s.fresh[id]) fresh[id] = true;
      if (s.synced) for (const id of added) fresh[id] = true;
      const reference = s.reference !== null && alive.has(s.reference) ? s.reference : (order[0] ?? null);
      return { order, letters, hidden, fresh, reference, synced: true };
    }),
  toggleVisible: (id) =>
    set((s) => {
      const hidden = { ...s.hidden };
      if (hidden[id]) delete hidden[id];
      else hidden[id] = true;
      return { hidden };
    }),
  setReference: (id) => set((s) => (s.order.includes(id) ? { reference: id } : s)),
  setTab: (tab) => set({ tab }),
  setAxle: (axle) => set({ axle }),
  setMode: (mode) => set({ mode }),
  setTheme: (theme) => set({ theme }),
  setShowPoints: (showPoints) => {
    try {
      localStorage.setItem(POINTS_KEY, showPoints ? "1" : "0");
    } catch {
      // storage unavailable
    }
    set({ showPoints });
  },
  toggleTheme: () => set((s) => ({ theme: s.theme === "light" ? "dark" : "light" })),
  setDrawer: (drawerOpen) => set({ drawerOpen }),
  setOpenDialog: (openDialog) => set({ openDialog }),
  clearFresh: (id) =>
    set((s) => {
      if (!s.fresh[id]) return s;
      const fresh = { ...s.fresh };
      delete fresh[id];
      return { fresh };
    }),
}));

/** Persist and apply the theme (call once at start-up). */
export function bindTheme(): () => void {
  const apply = (t: Theme) => {
    document.documentElement.dataset.theme = t;
    try {
      localStorage.setItem(THEME_KEY, t);
    } catch {
      // storage unavailable
    }
  };
  apply(useUi.getState().theme);
  return useUi.subscribe((s, prev) => {
    if (s.theme !== prev.theme) apply(s.theme);
  });
}

/** Keep the tab and the URL hash in step: hash changes switch tab, tab changes rewrite the hash. */
export function bindHash(): () => void {
  const fromHash = () => {
    const h = location.hash.replace(/^#/, "");
    if (h && h !== useUi.getState().tab) useUi.getState().setTab(h);
  };
  window.addEventListener("hashchange", fromHash);
  const unsub = useUi.subscribe((s, prev) => {
    if (s.tab !== prev.tab && s.tab !== null && location.hash !== `#${s.tab}`) {
      history.replaceState(null, "", `#${s.tab}`);
    }
  });
  return () => {
    window.removeEventListener("hashchange", fromHash);
    unsub();
  };
}

/** Visible dataset ids in open order. */
export function visibleIds(s: Pick<UiState, "order" | "hidden">): string[] {
  return s.order.filter((id) => !s.hidden[id]);
}
