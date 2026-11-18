export type Num = number | null;

/**
 * Linear interpolation of (xs, ys) at each query point.
 *
 * `xs` must be strictly monotonic (increasing or decreasing) and finite; otherwise every
 * result is null. A query outside the x range, a non-finite query, or a bracketing y that
 * is null gives null for that point (no extrapolation, no bridging of gaps).
 */
export function interpolate(xs: readonly Num[], ys: readonly Num[], query: readonly Num[]): Num[] {
  const n = Math.min(xs.length, ys.length);
  const out: Num[] = query.map(() => null);
  if (n < 2) {
    if (n === 1 && xs[0] != null) {
      query.forEach((q, i) => {
        if (q != null && q === xs[0]) out[i] = ys[0] ?? null;
      });
    }
    return out;
  }
  let dir = 0;
  for (let i = 0; i < n; i++) {
    const x = xs[i];
    if (x == null || !Number.isFinite(x)) return out;
    if (i > 0) {
      const d = x - (xs[i - 1] as number);
      if (d === 0 || (dir !== 0 && Math.sign(d) !== dir)) return out;
      dir = Math.sign(d);
    }
  }
  const asc = dir > 0;
  const first = xs[0] as number;
  const last = xs[n - 1] as number;
  const lo = asc ? first : last;
  const hi = asc ? last : first;
  query.forEach((q, qi) => {
    if (q == null || !Number.isFinite(q) || q < lo || q > hi) return;
    // binary search for the bracketing segment [i, i + 1]
    let a = 0;
    let b = n - 1;
    while (b - a > 1) {
      const m = (a + b) >> 1;
      const xm = xs[m] as number;
      if (asc ? xm <= q : xm >= q) a = m;
      else b = m;
    }
    const x0 = xs[a] as number;
    const x1 = xs[b] as number;
    const y0 = ys[a] ?? null;
    const y1 = ys[b] ?? null;
    if (q === x0) {
      out[qi] = y0;
      return;
    }
    if (q === x1) {
      out[qi] = y1;
      return;
    }
    if (y0 == null || y1 == null) return;
    out[qi] = y0 + ((y1 - y0) * (q - x0)) / (x1 - x0);
  });
  return out;
}

/**
 * Difference "revision minus reference" on the reference x grid.
 *
 * The revision curve is interpolated linearly onto `refX`; the result is null wherever
 * either curve is undefined there.
 */
export function deltaToReference(
  refX: readonly Num[],
  refY: readonly Num[],
  x: readonly Num[],
  y: readonly Num[],
): Num[] {
  const onRef = interpolate(x, y, refX);
  return onRef.map((v, i) => {
    const r = refY[i];
    return v == null || r == null ? null : v - r;
  });
}
