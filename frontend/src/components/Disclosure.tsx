import { useEffect, useId, useRef, useState, type ReactNode } from "react";

/**
 * A secondary button that opens a small panel under it. Esc closes it and returns focus to the
 * button; so does a click outside. The panel floats, so opening it never moves the layout.
 */
export function Disclosure({
  label,
  className,
  children,
}: {
  label: string;
  className?: string;
  children: (close: () => void) => ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const panelId = useId();

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      setOpen(false);
      button.current?.focus();
    };
    const onDown = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onDown);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onDown);
    };
  }, [open]);

  return (
    <div className="centres-menu" ref={root}>
      <button
        ref={button}
        type="button"
        className="btn secondary"
        aria-expanded={open}
        aria-controls={open ? panelId : undefined}
        onClick={() => setOpen(!open)}
      >
        {label}
      </button>
      {open && (
        <div id={panelId} className={className ? `centres-panel ${className}` : "centres-panel"}>
          {children(() => setOpen(false))}
        </div>
      )}
    </div>
  );
}
