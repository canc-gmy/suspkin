import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import type { BrowseEntry } from "../api";
import { useBrowse, useOpenPath } from "../api/hooks";
import { useUi } from "../state/store";

const KIND_LABEL: Record<string, string> = {
  vehicle: "Vehicle",
  study: "Study",
  bundle: "Saved bundle",
  report: "Report",
  run: "Optimizer run",
};

const openable = (e: BrowseEntry) => e.kind in KIND_LABEL;

/** Pick workspace files or folders (vehicle YAML, bundle, report, optimizer run) and open them. */
export function OpenDialog() {
  const close = useUi((s) => s.setOpenDialog);
  const [dir, setDir] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const browse = useBrowse(dir);
  const open = useOpenPath();
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    box.current?.focus();
  }, []);

  const name = (p: string) => p.split("/").pop() || p;
  const toggle = (p: string) =>
    setSelected((s) => (s.includes(p) ? s.filter((x) => x !== p) : [...s, p]));

  const submit = async () => {
    setBusy(true);
    const next: Record<string, string> = {};
    const left: string[] = [];
    for (const p of selected) {
      try {
        await open.mutateAsync(p);
      } catch (e) {
        next[p] = `Could not open ${name(p)}: ${e instanceof Error ? e.message : "unknown error"}`;
        left.push(p);
      }
    }
    setBusy(false);
    setErrors(next);
    setSelected(left);
    if (left.length === 0) close(false);
  };

  const onKey = (e: KeyboardEvent) => {
    if (e.key === "Escape") {
      close(false);
      return;
    }
    if (e.key !== "Tab" || !box.current) return;
    const items = box.current.querySelectorAll<HTMLElement>("button:not(:disabled), input:not(:disabled)");
    const first = items[0];
    const last = items[items.length - 1];
    if (!first || !last) return;
    if (e.shiftKey && (document.activeElement === first || document.activeElement === box.current)) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && document.activeElement === last) {
      e.preventDefault();
      first.focus();
    }
  };

  const data = browse.data;
  const crumbs = dir ? dir.split("/") : [];

  return (
    <div className="scrim" onMouseDown={(e) => e.target === e.currentTarget && close(false)}>
      <div
        className="dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="open-title"
        ref={box}
        tabIndex={-1}
        onKeyDown={onKey}
      >
        <h2 id="open-title" className="view-title">
          Open files
        </h2>
        <nav className="crumbs" aria-label="Folder">
          <button type="button" className="link-btn" onClick={() => setDir("")} disabled={dir === ""}>
            Workspace
          </button>
          {crumbs.map((c, i) => (
            <span key={i}>
              {" / "}
              <button
                type="button"
                className="link-btn"
                onClick={() => setDir(crumbs.slice(0, i + 1).join("/"))}
                disabled={i === crumbs.length - 1}
              >
                {c}
              </button>
            </span>
          ))}
        </nav>

        {browse.isError && (
          <p className="error" role="alert">
            Could not list this folder: {browse.error.message}
          </p>
        )}
        {browse.isPending && <p className="muted">Loading folder.</p>}
        {data && (
          <ul className="file-list">
            {data.parent !== null && (
              <li>
                <button type="button" className="file-row" onClick={() => setDir(data.parent ?? "")}>
                  <span className="file-name">..</span>
                  <span className="muted">Parent folder</span>
                </button>
              </li>
            )}
            {data.entries
              .filter((e) => e.kind === "dir" || openable(e))
              .map((e) => (
                <li key={e.path}>
                  {e.kind === "dir" ? (
                    <button type="button" className="file-row" onClick={() => setDir(e.path)}>
                      <span className="file-name">{e.name}</span>
                      <span className="muted">Folder</span>
                    </button>
                  ) : (
                    <>
                      <label className="file-row">
                        <input
                          type="checkbox"
                          checked={selected.includes(e.path)}
                          onChange={() => toggle(e.path)}
                        />
                        <span className="file-name">{e.name}</span>
                        <span className="muted">{KIND_LABEL[e.kind]}</span>
                      </label>
                      {errors[e.path] && (
                        <p className="error" role="alert">
                          {errors[e.path]}
                        </p>
                      )}
                    </>
                  )}
                </li>
              ))}
            {data.entries.length === 0 && <li className="muted">This folder is empty.</li>}
          </ul>
        )}
        {Object.entries(errors)
          .filter(([p]) => !data?.entries.some((e) => e.path === p))
          .map(([p, m]) => (
            <p key={p} className="error" role="alert">
              {m}
            </p>
          ))}

        <div className="dialog-actions">
          <button type="button" className="btn secondary" onClick={() => close(false)}>
            Cancel
          </button>
          <button type="button" className="btn primary" disabled={selected.length === 0 || busy} onClick={() => void submit()}>
            {busy ? "Opening" : selected.length > 1 ? `Open ${selected.length} files` : "Open files"}
          </button>
        </div>
      </div>
    </div>
  );
}
