/**
 * Natural cubic splines for curve drawing and the readout.
 *
 * One spline is built per series. The readout evaluates it at the cursor and the chart draws
 * its dense resample, so the line and the numbers always agree.
 */

export type Num = number | null;

/** A dense sample of a spline: `y` is null where the series has a gap (x is still set). */
export interface Resampled {
  x: number[];
  y: Num[];
}

export interface Spline {
  /** Value at `x`, or null outside the data range or inside a gap (never extrapolates). */
  evaluate(x: number): number | null;
  /** `n` evenly spaced points per segment (>= 2); a null y separates segments. */
  resample(n: number): Resampled;
  /** Number of segments (runs of consecutive finite points). */
  readonly segments: number;
  /** Smallest and largest x of the data, or null for an empty series. */
  readonly domain: readonly [number, number] | null;
}

interface Segment {
  x: number[];
  y: number[];
  /** Second derivatives at the knots (natural spline: zero at both ends). */
  m: number[];
}

/** Second derivatives of the natural cubic spline through (x, y); x strictly increasing. */
function naturalSecondDerivatives(x: readonly number[], y: readonly number[]): number[] {
  const n = x.length;
  const m = new Array<number>(n).fill(0);
  if (n < 3) return m; // two points: a straight line
  const h = (i: number) => (x[i + 1] as number) - (x[i] as number);
  // tridiagonal system for the interior knots, solved with the Thomas algorithm
  const diag = new Array<number>(n).fill(0);
  const upper = new Array<number>(n).fill(0);
  const rhs = new Array<number>(n).fill(0);
  for (let i = 1; i < n - 1; i++) {
    const h0 = h(i - 1);
    const h1 = h(i);
    diag[i] = 2 * (h0 + h1);
    upper[i] = h1;
    rhs[i] = 6 * (((y[i + 1] as number) - (y[i] as number)) / h1 - ((y[i] as number) - (y[i - 1] as number)) / h0);
  }
  const lower = (i: number) => h(i - 1);
  for (let i = 2; i < n - 1; i++) {
    const w = lower(i) / (diag[i - 1] as number);
    diag[i] = (diag[i] as number) - w * (upper[i - 1] as number);
    rhs[i] = (rhs[i] as number) - w * (rhs[i - 1] as number);
  }
  m[n - 2] = (rhs[n - 2] as number) / (diag[n - 2] as number);
  for (let i = n - 3; i >= 1; i--) {
    m[i] = ((rhs[i] as number) - (upper[i] as number) * (m[i + 1] as number)) / (diag[i] as number);
  }
  return m;
}

function evalSegment(s: Segment, x: number): number {
  const n = s.x.length;
  if (n === 1) return s.y[0] as number;
  // binary search for the interval [i, i + 1] holding x
  let a = 0;
  let b = n - 1;
  while (b - a > 1) {
    const mid = (a + b) >> 1;
    if ((s.x[mid] as number) <= x) a = mid;
    else b = mid;
  }
  const x0 = s.x[a] as number;
  const x1 = s.x[b] as number;
  if (x === x0) return s.y[a] as number;
  if (x === x1) return s.y[b] as number;
  const y0 = s.y[a] as number;
  const y1 = s.y[b] as number;
  const h = x1 - x0;
  const m0 = s.m[a] as number;
  const m1 = s.m[b] as number;
  const p = x1 - x;
  const q = x - x0;
  return (m0 * p ** 3 + m1 * q ** 3) / (6 * h) + (y0 / h - (m0 * h) / 6) * p + (y1 / h - (m1 * h) / 6) * q;
}

/**
 * Natural cubic spline of the series (xs, ys).
 *
 * Points with a non-finite x are dropped. The rest are sorted by x and repeated x values are
 * dropped (the first one wins). A null or non-finite y is a gap: it splits the series into
 * segments and the spline never interpolates across it. A segment of two points is a straight
 * line; a segment of one point exists only at its own x.
 */
export function buildSpline(xs: readonly Num[], ys: readonly Num[]): Spline {
  const n = Math.min(xs.length, ys.length);
  const pts: { x: number; y: number | null }[] = [];
  for (let i = 0; i < n; i++) {
    const x = xs[i];
    if (x == null || !Number.isFinite(x)) continue;
    const y = ys[i];
    pts.push({ x, y: y != null && Number.isFinite(y) ? y : null });
  }
  pts.sort((a, b) => a.x - b.x); // stable: the first of equal x stays first
  const segs: Segment[] = [];
  let cur: { x: number[]; y: number[] } | null = null;
  let lastX = Number.NaN;
  for (const p of pts) {
    if (p.x === lastX) continue;
    lastX = p.x;
    if (p.y === null) {
      cur = null;
      continue;
    }
    if (!cur) {
      cur = { x: [], y: [] };
      segs.push({ x: cur.x, y: cur.y, m: [] });
    }
    cur.x.push(p.x);
    cur.y.push(p.y);
  }
  for (const s of segs) s.m = naturalSecondDerivatives(s.x, s.y);

  const lo = (s: Segment) => s.x[0] as number;
  const hi = (s: Segment) => s.x[s.x.length - 1] as number;
  const first = segs[0];
  const last = segs[segs.length - 1];
  const domain: readonly [number, number] | null = first && last ? [lo(first), hi(last)] : null;

  return {
    segments: segs.length,
    domain,
    evaluate(x: number): number | null {
      if (!Number.isFinite(x)) return null;
      let a = 0;
      let b = segs.length - 1;
      while (a <= b) {
        const mid = (a + b) >> 1;
        const s = segs[mid] as Segment;
        if (x < lo(s)) b = mid - 1;
        else if (x > hi(s)) a = mid + 1;
        else return evalSegment(s, x);
      }
      return null;
    },
    resample(count: number): Resampled {
      const k = Math.max(2, Math.floor(count));
      const x: number[] = [];
      const y: Num[] = [];
      segs.forEach((s, si) => {
        const a = lo(s);
        const b = hi(s);
        if (si > 0) {
          x.push(a); // gap marker: breaks the drawn line
          y.push(null);
        }
        if (s.x.length === 1) {
          x.push(a);
          y.push(s.y[0] as number);
          return;
        }
        for (let i = 0; i < k; i++) {
          const xi = i === k - 1 ? b : a + ((b - a) * i) / (k - 1);
          x.push(xi);
          y.push(evalSegment(s, xi));
        }
      });
      return { x, y };
    },
  };
}

/** Points per segment for drawing: 8x the solved points, at least 200. */
export function drawPoints(solved: number): number {
  return Math.max(200, 8 * solved);
}
