import { useEffect, useRef, useState } from "react";

interface Props {
  /** Rows of the panel's solved points (header first); built on click, never on render. */
  getRows: () => string[][];
  filename: string;
  toCsv: (rows: string[][]) => string;
  toTsv: (rows: string[][]) => string;
}

const CONFIRM_MS = 1500;

/** Copy text with the async clipboard, or a hidden textarea when it is not available. */
export async function copyText(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // fall through to the textarea
  }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  } catch {
    return false;
  }
}

/** Save text as a file (UTF-8 with a byte-order mark, so spreadsheets read the Δ). */
export function downloadText(text: string, filename: string): void {
  const url = URL.createObjectURL(new Blob(["﻿", text], { type: "text/csv;charset=utf-8" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/** "Copy data" and "Download CSV" text buttons for a panel header, with an inline confirmation. */
export function PanelActions({ getRows, filename, toCsv, toTsv }: Props) {
  const [status, setStatus] = useState<"copied" | "failed" | null>(null);
  const timer = useRef(0);
  useEffect(() => () => window.clearTimeout(timer.current), []);

  const say = (s: "copied" | "failed") => {
    setStatus(s);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setStatus(null), CONFIRM_MS);
  };

  return (
    <div className="panel-actions">
      <button
        type="button"
        className="text-btn"
        onClick={() => {
          void copyText(toTsv(getRows())).then((ok) => say(ok ? "copied" : "failed"));
        }}
      >
        Copy data
      </button>
      <button type="button" className="text-btn" onClick={() => downloadText(toCsv(getRows()), filename)}>
        Download CSV
      </button>
      <span className={status === "failed" ? "panel-status error" : "panel-status"} aria-live="polite">
        {status === "copied" ? "Copied" : status === "failed" ? "Could not copy, use Download CSV" : ""}
      </span>
    </div>
  );
}
