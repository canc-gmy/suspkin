"""Batched corner kinematics: exact forward kinematics and metric columns over many states.

Vectorised ``corner._pose`` + ``_finish`` at rack 0 for ``B`` designs (per-design side-resolved
hardpoints) and ``K`` states each, given the joint angles. Every corner column (metrics,
gradients, joint angles, point coordinates) follows from the rebuilt points with the formulas
of :mod:`suspkin.metrics.arrays`. Numpy only; the surrogate evaluator is the main user.

Array conventions: ``B`` designs, ``K`` states per design; points are ``(B, K, 3)`` in m
(vehicle frame), angles in rad with the per-side sign of ``docs/concepts/conventions.md``
section 2, columns ``(B, K)``. Column meanings and signs: ``docs/concepts/metrics.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from suspkin.geometry.corners import Corner
from suspkin.geometry.model import Axle
from suspkin.geometry.vectors import (
    cross_rows,
    dot_rows,
    norm_rows,
    rodrigues_rows,
    unit_rows,
    wheel_spin_axis,
    wrap_angle,
)
from suspkin.kinematics.ic_arrays import ic_arrays, roll_line_arrays, swing_angle

Arr = NDArray[np.float64]
_NAN = float("nan")

# metric column -> gradient column, per sweep variable (``d(metric)/d(travel or rack)``).
GRADIENTS: dict[str, dict[str, str]] = {
    "travel": {
        "camber": "camber_gain",
        "toe": "bump_steer",
        "caster": "caster_change",
        "half_track_change": "track_change_rate",
        "rc_height": "rc_migration_rate",
    },
    "rack": {"steer_angle": "steer_gain"},
}

__all__ = ["GRADIENTS", "CStatic", "Rebuilt", "corner_columns", "make_static", "rebuild"]


def _grad(y: Arr, x: Arr) -> Arr:
    """``np.gradient`` along axis 1 (typed as an array)."""
    out: Arr = np.gradient(y, x, axis=1)
    return out


# --------------------------------------------------------------------------------------
# exact forward kinematics, vectorised over designs and states
# --------------------------------------------------------------------------------------


@dataclass
class CStatic:
    """Static corner data of ``B`` designs (side-resolved hardpoints, ``(B, 3)``).

    Attributes:
        cid: Corner id (``"FL"`` ...), same for all designs.
        side: ``"left"`` or ``"right"``.
        axle: ``"front"`` or ``"rear"``.
        sign: Per-side angle sign (+1 left, -1 right).
        p: Hardpoint name -> ``(B, 3)`` array in m.
        axis: Rocker axes, ``(B, 3)`` (as stored on the corner).
        cfg: Axle configuration of the first design (shared wheel, coilover, mount type).
        spin0: Static wheel spin axis, ``(3,)``.
        scp: Static contact patch per design, ``(B, 3)``, m.
        len0: Static coilover length per design, ``(B,)``, m.
        corner0: First design's :class:`~suspkin.geometry.corners.Corner`.
    """

    cid: str
    side: str
    axle: str
    sign: float
    p: dict[str, Arr]
    axis: Arr
    cfg: Axle
    spin0: Arr
    scp: Arr
    len0: Arr
    corner0: Corner


def make_static(corners: Sequence[Corner]) -> CStatic:
    """Stack the same corner of ``B`` designs into a :class:`CStatic`.

    Args:
        corners: ``B`` corners with the same id, side and hardpoint names (one per design).
            Wheel, coilover and pushrod-mount data are taken from the first.

    Returns:
        Static data with ``(B, 3)`` hardpoint arrays.
    """
    c0 = corners[0]
    p = {
        n: np.array([c.points[n] for c in corners], dtype=np.float64).reshape(-1, 3)
        for n in c0.points
    }
    cfg = c0.axle_cfg
    spin0 = wheel_spin_axis(c0.side, cfg.wheel.static_camber, cfg.wheel.static_toe)
    fz = spin0[2]
    up = unit_rows(np.array([-fz * spin0[0], -fz * spin0[1], 1.0 - fz * spin0[2]]))
    scp = p["wheel_center"] - cfg.wheel.tyre_radius * up
    len0 = np.linalg.norm(p["rocker_spring"] - p["spring_chassis"], axis=-1)
    return CStatic(
        c0.id,
        c0.side,
        c0.axle,
        1.0 if c0.side == "left" else -1.0,
        p,
        np.array([c.rocker_axis for c in corners], dtype=np.float64).reshape(-1, 3),
        cfg,
        np.asarray(spin0, dtype=np.float64),
        scp,
        len0,
        c0,
    )


@dataclass
class Rebuilt:
    """Rebuilt corner states: ``(B, K, ...)`` arrays of the moved points and solver outputs.

    Attributes:
        st: The :class:`CStatic` these states belong to.
        moved: Moved point name -> ``(B, K, 3)`` array in m (includes ``contact_patch``).
        spin: Unit wheel spin axes, ``(B, K, 3)``.
        coil: Coilover lengths, ``(B, K)``, m.
        toggle: Toggle margins, ``(B, K)``, rad.
        rocker: Rocker angles, ``(B, K)``, rad, per-side signed.
        theta_uca: Upper wishbone angles, ``(B, K)``, rad.
        theta_lca: Lower wishbone angles, ``(B, K)``, rad.
        psi: Upright spin angles, ``(B, K)``, rad.
    """

    st: CStatic
    moved: dict[str, Arr]
    spin: Arr
    coil: Arr
    toggle: Arr
    rocker: Arr
    theta_uca: Arr
    theta_lca: Arr
    psi: Arr

    def pt(self, name: str) -> Arr:
        """Point ``name`` as ``(B, K, 3)`` in m (static points are broadcast).

        Raises:
            KeyError: If ``name`` is neither a moved nor a static point.
        """
        if name in self.moved:
            return self.moved[name]
        if name not in self.st.p:
            raise KeyError(name)
        a = self.st.p[name]
        return np.broadcast_to(a[:, None, :], (a.shape[0], self.rocker.shape[1], 3))


def rebuild(st: CStatic, th_u: Arr, th_l: Arr, psi: Arr, guess: Arr) -> Rebuilt:
    """Vectorised ``corner._pose`` + ``_finish`` at rack 0 for given joint angles.

    Args:
        st: Static corner data of ``B`` designs.
        th_u: Upper wishbone angles, ``(B, K)``, rad, per-side signed.
        th_l: Lower wishbone angles, ``(B, K)``, rad, per-side signed.
        psi: Upright spin angles, ``(B, K)``, rad, per-side signed.
        guess: Rocker angles ``(B, K)`` in rad (per-side signed) selecting the root branch.

    Returns:
        :class:`Rebuilt` with the moved points (m), spin axes, coilover lengths, rocker
        angles and toggle margins. Invalid states (flipped upright, pushrod out of reach,
        non-finite angles) give NaN; nothing raises.
    """

    def b3(a: Arr) -> Arr:
        return a[:, None, :]

    s = st.sign
    p = st.p
    with np.errstate(all="ignore"):
        uf, lf = b3(p["uca_front"]), b3(p["lca_front"])
        uca_hat = unit_rows(b3(p["uca_rear"] - p["uca_front"]))
        lca_hat = unit_rows(b3(p["lca_rear"] - p["lca_front"]))
        cu, su = np.cos(s * th_u)[..., None], np.sin(s * th_u)[..., None]
        cl, sl = np.cos(s * th_l)[..., None], np.sin(s * th_l)[..., None]
        ubj = uf + rodrigues_rows(uca_hat, cu, su, b3(p["uca_outer"] - p["uca_front"]))
        lbj = lf + rodrigues_rows(lca_hat, cl, sl, b3(p["lca_outer"] - p["lca_front"]))
        d = ubj - lbj
        n = unit_rows(d)
        u0 = unit_rows(b3(p["uca_outer"] - p["lca_outer"]))
        cc = dot_rows(u0, n, keepdims=True)
        ik = np.where(cc < -1.0 + 1e-12, _NAN, 1.0 / (1.0 + cc))
        w = cross_rows(u0, n)
        cph, sph = np.cos(s * psi)[..., None], np.sin(s * psi)[..., None]

        def rot(v: Arr) -> Arr:
            wr = cross_rows(w, v)
            return rodrigues_rows(n, cph, sph, v + wr + cross_rows(w, wr) * ik)

        tie_outer = lbj + rot(b3(p["tie_outer"] - p["lca_outer"]))
        wc = lbj + rot(b3(p["wheel_center"] - p["lca_outer"]))
        mount = st.cfg.pushrod_mount
        if mount == "upright":
            po = lbj + rot(b3(p["pushrod_outer"] - p["lca_outer"]))
        elif mount == "lca":
            po = lf + rodrigues_rows(lca_hat, cl, sl, b3(p["pushrod_outer"] - p["lca_front"]))
        else:
            po = uf + rodrigues_rows(uca_hat, cu, su, b3(p["pushrod_outer"] - p["uca_front"]))

        # rocker: closed form (RockerCtx.solve), branch nearest to the predicted angle
        pivot = b3(p["rocker_pivot"])
        axis = b3(st.axis)
        k = unit_rows(axis)
        r_in = b3(p["pushrod_inner"]) - pivot
        w_r = r_in - dot_rows(r_in, axis, keepdims=True) * axis
        ww = dot_rows(w_r, w_r, keepdims=True)
        axw = cross_rows(axis, w_r)
        length2 = np.sum((p["pushrod_inner"] - p["pushrod_outer"]) ** 2, axis=-1)[:, None, None]
        r = (po - pivot) - dot_rows(r_in, axis, keepdims=True) * axis
        ca, cb = dot_rows(r, w_r, keepdims=True)[..., 0], dot_rows(r, axw, keepdims=True)[..., 0]
        c2 = 0.5 * (dot_rows(r, r, keepdims=True) + ww - length2)[..., 0]
        amp = np.hypot(ca, cb)
        bad = (amp < 1e-12) | (np.abs(c2) > amp * (1.0 + 1e-12))
        half = np.arccos(np.clip(c2 / amp, -1.0, 1.0))
        base = np.arctan2(cb, ca)
        g = s * guess
        c_a, c_b = base + half, base - half
        best = np.where(np.abs(wrap_angle(c_a - g)) <= np.abs(wrap_angle(c_b - g)), c_a, c_b)
        phys = np.where(bad, _NAN, g + wrap_angle(best - g))
        cph_r, sph_r = np.cos(phys)[..., None], np.sin(phys)[..., None]

        def rocker_rot(v: Arr) -> Arr:
            return pivot + rodrigues_rows(k, cph_r, sph_r, v)

        p_in = rocker_rot(r_in)
        p_spring = rocker_rot(b3(p["rocker_spring"]) - pivot)
        lever = p_in - pivot
        lever = lever - dot_rows(lever, axis, keepdims=True) * axis
        push = po - p_in
        push = push - dot_rows(push, axis, keepdims=True) * axis
        nl, npu = np.linalg.norm(lever, axis=-1), np.linalg.norm(push, axis=-1)
        lp = dot_rows(lever, push, keepdims=True)[..., 0]
        cosang = np.where(nl * npu > 1e-12, lp / (nl * npu), 1.0)
        a = np.arccos(np.clip(cosang, -1.0, 1.0))
        toggle = np.minimum(a, np.pi - a)

        spin = unit_rows(rot(st.spin0[None, None, :]))
        fz = spin[..., 2:3]
        up = unit_rows(
            np.concatenate(
                [-fz * spin[..., 0:1], -fz * spin[..., 1:2], 1.0 - fz * spin[..., 2:3]], -1
            )
        )
        moved: dict[str, Arr] = {
            "uca_outer": ubj,
            "lca_outer": lbj,
            "tie_outer": tie_outer,
            "wheel_center": wc,
            "pushrod_outer": po,
            "pushrod_inner": p_in,
            "rocker_spring": p_spring,
            "contact_patch": wc - st.cfg.wheel.tyre_radius * up,
        }
        if "rocker_arb" in p:
            moved["rocker_arb"] = rocker_rot(b3(p["rocker_arb"]) - pivot)
        coil = np.linalg.norm(p_spring - b3(p["spring_chassis"]), axis=-1)
    return Rebuilt(st, moved, spin, coil, toggle, s * phys, th_u, th_l, psi)


# --------------------------------------------------------------------------------------
# corner columns (vectorised ``metrics.arrays.corner_arrays`` with per-design statics)
# --------------------------------------------------------------------------------------


def corner_columns(rb: Rebuilt, travel: Arr) -> dict[str, Arr]:
    """All ``states_to_frame(metrics=True)`` + ``add_gradients`` columns, each ``(B, K)``.

    Same formulas and signs as :func:`suspkin.metrics.arrays.corner_arrays`; the gradient
    columns are central differences along the ``K`` axis (``numpy.gradient``).

    Args:
        rb: Rebuilt states of ``B`` designs and ``K`` travels.
        travel: Wheel travels in m, ``(K,)`` strictly monotonic (also used as the gradient
            abscissa).

    Returns:
        Column name -> ``(B, K)`` array in SI units (lengths m, angles rad, rates N/m,
        gradients per metre). Rack is 0 and ``max_residual`` is 0 (exact rebuild).
    """
    st = rb.st
    b, k = rb.rocker.shape
    m = b * k

    def flat(a: Arr) -> Arr:
        return a.reshape(m, *a.shape[2:])

    def rep(a: Arr) -> Arr:
        return np.repeat(a, k, axis=0)

    cache: dict[str, Arr] = {}

    def pts(name: str) -> Arr:
        if name not in cache:
            cache[name] = flat(rb.pt(name))
        return cache[name]

    o = st.sign
    cfg = st.cfg
    coil = cfg.coilover
    spin = flat(rb.spin)
    length, rocker = flat(rb.coil), flat(rb.rocker)
    scp, swc = rep(st.scp), rep(st.p["wheel_center"])
    len0 = np.repeat(st.len0, k)
    out: dict[str, Arr] = {}
    with np.errstate(all="ignore"):
        cp, wc, lbj = pts("contact_patch"), pts("wheel_center"), pts("lca_outer")
        u = pts("uca_outer") - lbj

        def ground(z: Arr) -> Arr:
            t = (z - lbj[:, 2]) / u[:, 2]
            gr: Arr = lbj + t[:, None] * u
            return gr

        g_cp, g_wc = ground(cp[:, 2]), ground(wc[:, 2])
        un = u / norm_rows(u)[:, None]
        r = wc - lbj
        perp = r - dot_rows(r, un)[:, None] * un
        dx = cp[:, 0] - scp[:, 0]
        ht = np.abs(cp[:, 1])
        out["camber"] = -np.arcsin(np.clip(spin[:, 2], -1.0, 1.0))
        out["toe"] = np.arctan2(spin[:, 0], o * spin[:, 1])
        out["steer_angle"] = np.arctan2(-o * spin[:, 0], o * spin[:, 1])
        out["caster"] = np.arctan2(-u[:, 0], u[:, 2])
        out["kpi"] = np.arctan2(-o * u[:, 1], u[:, 2])
        out["scrub_radius"] = o * (cp[:, 1] - g_cp[:, 1])
        out["mechanical_trail"] = g_cp[:, 0] - cp[:, 0]
        out["kingpin_offset_wc"] = o * (wc[:, 1] - g_wc[:, 1])
        out["spindle_length"] = norm_rows(perp)
        out["half_track"] = ht
        out["half_track_change"] = ht - np.abs(scp[:, 1])
        out["contact_patch_dx"] = dx
        out["contact_patch_dy"] = o * (cp[:, 1] - scp[:, 1])
        out["wheel_center_recession"] = wc[:, 0] - swc[:, 0]
        out["wheelbase_change_contribution"] = dx if st.axle == "front" else -dx

        fv = ic_arrays(pts, 0, cp[:, 0])
        sv = ic_arrays(pts, 1, cp[:, 1])
        inboard = -1.0 if st.side == "left" else 1.0
        anti = -1.0 if st.axle == "front" else 1.0
        out["fvic_y"] = fv.h
        out["fvic_z"] = fv.z
        out["fvsa"] = np.where(fv.finite, np.abs(fv.h - cp[:, 1]), np.inf)
        out["fv_swing_angle"] = swing_angle(fv, cp[:, 1], cp[:, 2], inboard)
        out["svic_x"] = sv.h
        out["svic_z"] = sv.z
        out["svsa"] = np.where(sv.finite, np.abs(sv.h - cp[:, 0]), np.inf)
        out["sv_swing_angle"] = swing_angle(sv, cp[:, 0], cp[:, 2], anti)
        out["sv_wc_angle"] = swing_angle(sv, wc[:, 0], wc[:, 2], anti)
        rd0, rd1 = roll_line_arrays(cp[:, 1], cp[:, 2], fv)
        out["rc_height"] = np.where(
            np.abs(rd0) < 1e-12, _NAN, cp[:, 2] + rd1 / rd0 * (0.0 - cp[:, 1])
        )
        rod = pts("pushrod_inner") - pts("pushrod_outer")
        out["pushrod_angle"] = np.arctan2(rod[:, 2], np.hypot(rod[:, 0], rod[:, 1]))
        out["pushrod_angle_fv"] = np.arctan2(rod[:, 2], np.abs(rod[:, 1]))
        out["rocker_rotation"] = rocker
        out["coilover_compression"] = len0 - length
        out["bump_stroke_remaining"] = length - coil.length_min
        out["droop_stroke_remaining"] = coil.length_max - length
        out["spring_force"] = coil.preload + coil.spring_rate * (coil.free_length - length)
    out["spring_rate"] = np.full(m, float(coil.spring_rate))
    out["tyre_stiffness"] = np.full(m, float(cfg.wheel.tyre_vertical_stiffness))
    cols = {name: a.reshape(b, k) for name, a in out.items()}

    cols["travel"] = np.broadcast_to(travel, (b, k))
    cols["rack"] = np.zeros((b, k))
    cols["theta_uca"], cols["theta_lca"], cols["psi"] = rb.theta_uca, rb.theta_lca, rb.psi
    cols["rocker_angle"] = rb.rocker
    cols["coilover_length"] = rb.coil
    cols["toggle_margin"] = rb.toggle
    cols["max_residual"] = np.zeros((b, k))
    for i, axname in enumerate("xyz"):
        cols[f"spin_{axname}"] = rb.spin[..., i]
    for metric, name in GRADIENTS["travel"].items():
        cols[name] = _grad(cols[metric], travel)
    mr = _grad(cols["coilover_compression"], travel)
    prog = _grad(mr, travel)
    force, kk, kt = cols["spring_force"], cols["spring_rate"], cols["tyre_stiffness"]
    wr = kk * mr**2 + force * prog
    cols["mr_spring"] = mr
    cols["mr_progression"] = prog
    cols["wheel_load_spring"] = force * mr
    cols["wheel_rate_linear"] = kk * mr**2
    cols["wheel_rate"] = wr
    cols["ride_rate"] = wr * kt / (wr + kt)
    return cols
