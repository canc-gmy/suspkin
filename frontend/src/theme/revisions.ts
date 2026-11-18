export type Theme = "light" | "dark";

/**
 * Palette entries that fail contrast on a theme's sheet, replaced for that theme only.
 * Must mirror the `--rev-N` overrides in `tokens.css` (a test keeps them in sync).
 */
export const REVISION_OVERRIDES: Record<Theme, Record<number, string>> = {
  light: { 6: "#B8A800" },
  dark: { 0: "#4DA3E0", 7: "#FFFFFF" },
};

/** Colour of a dataset colour index on a theme: spec palette, with the theme overrides. */
export function revisionColor(colorIndex: number, palette: readonly string[], theme: Theme): string {
  const n = palette.length;
  if (n === 0) return "#808080";
  const i = ((colorIndex % n) + n) % n;
  return REVISION_OVERRIDES[theme][i] ?? (palette[i] as string);
}

/** Revision letter of the n-th slot: A..Z, then A1..Z1, A2... */
export function letterAt(n: number): string {
  const base = String.fromCharCode(65 + (n % 26));
  const round = Math.floor(n / 26);
  return round === 0 ? base : `${base}${round}`;
}

/** Lowest letter not in `used` (letters are assigned in open order and reused after a close). */
export function nextLetter(used: Iterable<string>): string {
  const taken = new Set(used);
  for (let n = 0; ; n++) {
    const l = letterAt(n);
    if (!taken.has(l)) return l;
  }
}
