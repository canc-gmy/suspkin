import { useEffect, useMemo, useRef } from "react";
import { useDatasets, useFigures, useHealth } from "./api/hooks";
import { OpenDialog } from "./components/OpenDialog";
import { SummaryTable } from "./components/SummaryTable";
import { TitleBlock } from "./components/TitleBlock";
import { groupLabel, groupsOf } from "./lib/panels";
import { useUi } from "./state/store";
import { EmptyState } from "./views/EmptyState";
import { FigureView } from "./views/FigureView";
import { Placeholder } from "./views/Placeholder";
import { Tabs, type TabDef } from "./views/Tabs";

const EXTRA_TABS: TabDef[] = [
  { id: "summary", label: "Summary" },
  { id: "scene3d", label: "3D" },
  { id: "optimization", label: "Optimization" },
];

export function App() {
  const topbar = useRef<HTMLElement>(null);
  useEffect(() => {
    const el = topbar.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const set = () => document.documentElement.style.setProperty("--topbar-h", `${el.offsetHeight}px`);
    set();
    const ro = new ResizeObserver(set);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const health = useHealth();
  const figures = useFigures();
  const datasets = useDatasets();
  const theme = useUi((s) => s.theme);
  const toggleTheme = useUi((s) => s.toggleTheme);
  const tab = useUi((s) => s.tab);
  const setTab = useUi((s) => s.setTab);
  const drawerOpen = useUi((s) => s.drawerOpen);
  const setDrawer = useUi((s) => s.setDrawer);
  const openDialog = useUi((s) => s.openDialog);
  const setOpenDialog = useUi((s) => s.setOpenDialog);
  const syncDatasets = useUi((s) => s.syncDatasets);

  const list = datasets.data;
  useEffect(() => {
    if (list) syncDatasets(list.map((d) => d.id));
  }, [list, syncDatasets]);

  const spec = figures.data;
  const tabs = useMemo<TabDef[]>(
    () => [...(spec ? groupsOf(spec.figures).map((g) => ({ id: g, label: groupLabel(g) })) : []), ...EXTRA_TABS],
    [spec],
  );
  const active = tab !== null && tabs.some((t) => t.id === tab) ? tab : (tabs[0]?.id ?? "");
  const empty = list !== undefined && list.length === 0;

  let content: React.ReactNode;
  if (datasets.isError || figures.isError) {
    const msg = (datasets.error ?? figures.error)?.message ?? "unknown error";
    content = (
      <p className="error" role="alert">
        Could not load the results: {msg}
      </p>
    );
  } else if (!spec || !list) {
    content = <p className="muted">Loading.</p>;
  } else if (empty) {
    content = <EmptyState />;
  } else if (active === "summary") {
    content = (
      <section aria-label="Summary">
        <h2 className="view-title">Summary</h2>
        <SummaryTable spec={spec} datasets={list} />
      </section>
    );
  } else if (active === "scene3d") {
    content = <Placeholder title="3D" />;
  } else if (active === "optimization") {
    content = <Placeholder title="Optimization" />;
  } else {
    content = <FigureView group={active} spec={spec} datasets={list} />;
  }

  return (
    <div className="app">
      <header className="topbar" ref={topbar}>
        <span className="product">suspkin</span>
        <span className="workspace" title={health.data?.workspace}>
          {health.data ? `workspace: ${health.data.workspace}` : ""}
        </span>
        <button
          type="button"
          className="btn secondary drawer-toggle"
          aria-expanded={drawerOpen}
          aria-controls="titleblock"
          onClick={() => setDrawer(!drawerOpen)}
        >
          Revisions
        </button>
        <button type="button" className="btn secondary" onClick={() => setOpenDialog(true)}>
          Open files
        </button>
        <span title="Comes in a later version">
          <button type="button" className="btn secondary" disabled>
            Export snapshot
          </button>
        </span>
        <button
          type="button"
          className="icon-btn"
          aria-label={theme === "light" ? "Switch to the blueprint (dark) theme" : "Switch to the drafting film (light) theme"}
          onClick={toggleTheme}
        >
          {"◐"}
        </button>
      </header>
      <div className="body">
        <aside id="titleblock" className={drawerOpen ? "titleblock open" : "titleblock"} aria-label="Title block">
          {spec && list ? (
            <TitleBlock datasets={list} palette={spec.palette} theme={theme} />
          ) : (
            <p className="muted">Loading.</p>
          )}
        </aside>
        <main className="main">
          {!empty && spec && list && <Tabs tabs={tabs} active={active} onChange={setTab} />}
          <div id="tab-panel" role="tabpanel" aria-labelledby={`tab-${active}`} className="tab-panel">
            {content}
          </div>
        </main>
      </div>
      {openDialog && <OpenDialog />}
    </div>
  );
}
