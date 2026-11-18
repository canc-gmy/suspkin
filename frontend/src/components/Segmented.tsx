import { useRef, type KeyboardEvent } from "react";

interface Props<T extends string> {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  disabled?: boolean;
}

/** Segmented control (radio group): 1 px rule, active segment filled with the layout blue. */
export function Segmented<T extends string>({ label, value, options, onChange, disabled = false }: Props<T>) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);

  const onKey = (e: KeyboardEvent, i: number) => {
    const step = e.key === "ArrowRight" || e.key === "ArrowDown" ? 1 : e.key === "ArrowLeft" || e.key === "ArrowUp" ? -1 : 0;
    if (!step) return;
    e.preventDefault();
    const j = (i + step + options.length) % options.length;
    const next = options[j];
    if (next) {
      onChange(next.value);
      refs.current[j]?.focus();
    }
  };

  const hasActive = options.some((o) => o.value === value);
  return (
    <div className="segmented" role="radiogroup" aria-label={label}>
      {options.map((o, i) => (
        <button
          key={o.value}
          ref={(el) => {
            refs.current[i] = el;
          }}
          type="button"
          role="radio"
          aria-checked={o.value === value}
          tabIndex={o.value === value || (!hasActive && i === 0) ? 0 : -1}
          disabled={disabled}
          className={o.value === value ? "seg active" : "seg"}
          onClick={() => onChange(o.value)}
          onKeyDown={(e) => onKey(e, i)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
