import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { letterAt, nextLetter, REVISION_OVERRIDES, revisionColor, type Theme } from "./revisions";

// Okabe-Ito order of spec.palette (plotting/style.py)
const PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"];

describe("revisionColor", () => {
  it("keeps the index -> hue mapping and overrides only the failing entries", () => {
    expect(revisionColor(1, PALETTE, "light")).toBe("#D55E00");
    expect(revisionColor(1, PALETTE, "dark")).toBe("#D55E00");
    expect(revisionColor(6, PALETTE, "light")).toBe("#B8A800");
    expect(revisionColor(6, PALETTE, "dark")).toBe("#F0E442");
    expect(revisionColor(0, PALETTE, "light")).toBe("#0072B2");
    expect(revisionColor(0, PALETTE, "dark")).toBe("#4DA3E0");
    expect(revisionColor(7, PALETTE, "light")).toBe("#000000");
    expect(revisionColor(7, PALETTE, "dark")).toBe("#FFFFFF");
  });

  it("wraps around the palette", () => {
    expect(revisionColor(8, PALETTE, "light")).toBe(revisionColor(0, PALETTE, "light"));
    expect(revisionColor(-1, PALETTE, "light")).toBe(revisionColor(7, PALETTE, "light"));
  });

  it("matches the overrides declared in tokens.css", () => {
    // vitest runs from frontend/
    const css = readFileSync(resolve(process.cwd(), "src/theme/tokens.css"), "utf-8");
    const block = (theme: Theme) => {
      const m = theme === "light" ? /\[data-theme="light"\]\s*\{([^}]*)\}/ : /\n\[data-theme="dark"\]\s*\{([^}]*)\}/;
      return m.exec(css)?.[1] ?? "";
    };
    for (const theme of ["light", "dark"] as const) {
      const found: Record<number, string> = {};
      for (const m of block(theme).matchAll(/--rev-(\d+):\s*(#[0-9a-fA-F]{6})/g)) {
        found[Number(m[1])] = (m[2] as string).toUpperCase();
      }
      expect(found).toEqual(REVISION_OVERRIDES[theme]);
    }
  });
});

describe("revision letters", () => {
  it("runs A..Z then A1..", () => {
    expect(letterAt(0)).toBe("A");
    expect(letterAt(25)).toBe("Z");
    expect(letterAt(26)).toBe("A1");
  });

  it("reuses the lowest free letter", () => {
    expect(nextLetter([])).toBe("A");
    expect(nextLetter(["A", "B"])).toBe("C");
    expect(nextLetter(["A", "C"])).toBe("B");
  });
});
