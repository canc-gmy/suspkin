import { useEffect, useRef, type KeyboardEvent } from "react";

export interface TabDef {
  id: string;
  label: string;
}

interface Props {
  tabs: TabDef[];
  active: string;
  onChange: (id: string) => void;
}

/** Text-only tabs; the active one has a 2 px layout-blue underline. */
export function Tabs({ tabs, active, onChange }: Props) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  useEffect(() => {
    const el = refs.current[tabs.findIndex((t) => t.id === active)];
    el?.scrollIntoView?.({ block: "nearest", inline: "nearest" });
  }, [active, tabs]);

  const onKey = (e: KeyboardEvent, i: number) => {
    const n = tabs.length;
    const target: Record<string, number> = {
      ArrowRight: (i + 1) % n,
      ArrowLeft: (i - 1 + n) % n,
      Home: 0,
      End: n - 1,
    };
    const j = target[e.key];
    if (j === undefined) return;
    e.preventDefault();
    const t = tabs[j];
    if (t) {
      onChange(t.id);
      refs.current[j]?.focus();
    }
  };

  return (
    <div className="tabs" role="tablist" aria-label="Results">
      {tabs.map((t, i) => (
        <button
          key={t.id}
          ref={(el) => {
            refs.current[i] = el;
          }}
          type="button"
          role="tab"
          id={`tab-${t.id}`}
          aria-selected={t.id === active}
          aria-controls="tab-panel"
          tabIndex={t.id === active ? 0 : -1}
          className={t.id === active ? "tab active" : "tab"}
          onClick={() => onChange(t.id)}
          onKeyDown={(e) => onKey(e, i)}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
