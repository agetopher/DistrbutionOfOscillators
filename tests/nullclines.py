import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import settings

"""
Phase-plane nullclines for the single Yuval neuron with recovery variable w.

The (V, w) dynamics are
    C  dV/dt = g f(V) - w + Iapp
    tau_w dw/dt = w_inf(V) - w,        w_inf(V) = beta * max(V - T, 0)

so the nullclines in the (V, w) plane are
    V-nullcline (dV/dt = 0):  w = g f(V) + Iapp
    w-nullcline (dw/dt = 0):  w = w_inf(V)

Fixed points are the intersections of the two curves; they are classified
from the Jacobian (filled = stable, open = unstable node/spiral, x = saddle).

Two views:
  'nullclines' — interactive phase plane (w vs V); sliders move the V-nullcline
                 up/down via Iapp and change the w-nullcline slope beta.
  'fw'         — f(V) and w_inf(V) plotted against V on one pane.

Usage:
  run()                          # default: 'nullclines' phase plane
  run('nullclines', Iapp=2.0)    # start with Iapp = 2
  run('fw', beta=0.5)            # f(V) and w_inf(V) together
"""

# ── Baseline membrane parameters ──────────────────────────────────────────────
BASELINE_MEMBRANE = dict(
    L  = -70.0,     # mV  resting potential
    T  = -45.0,     # mV  depolarisation threshold
    H  = -35.0,     # mV  plateau potential
    m1 = 0.7,
    m2 = 1 / 81.0,
    m3 = -1 / 30.0,
    m4 = 0.17,
)

# ── Membrane constants ────────────────────────────────────────────────────────
settings.C = 7.0    # pF
settings.g = 1.0    # nS
tau_w      = 200.0  # ms  recovery time constant

MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')

V_range = np.linspace(-85, -20, 700)

IAPP_MIN, IAPP_MAX = -5.0, 8.0
BETA_MIN, BETA_MAX = 0.0, 1.5


# ── Helper functions ──────────────────────────────────────────────────────────

def f_scalar(v, mem):
    """Piecewise-linear f(v) for a given membrane parameter dict."""
    L, T, H = mem['L'], mem['T'], mem['H']
    if v > H:
        return mem['m4'] * (-v + H)
    elif v >= T:
        return mem['m3'] * (v - H) * (v - T)
    elif v >= L:
        return mem['m2'] * (v - L) * (v - T)
    else:
        return mem['m1'] * (-v + L)


def compute_fV(mem):
    return np.array([f_scalar(v, mem) for v in V_range])


def w_inf(mem, beta):
    """w-nullcline: w_inf(V) = beta * max(V - T, 0)."""
    return beta * np.maximum(V_range - mem['T'], 0.0)


def fixed_points(mem, fV, Iapp, beta):
    """
    Intersections of the V-nullcline (w = g f(V) + Iapp) and the w-nullcline
    (w = w_inf(V)). Returns (V*, w*, kind) with kind in {'stable','unstable','saddle'}.
    """
    v_null = settings.g * fV + Iapp
    w_null = w_inf(mem, beta)
    diff   = v_null - w_null

    fprime = np.gradient(fV, V_range)   # f'(V) over the grid

    pts = []
    for i in range(len(diff) - 1):
        d0, d1 = diff[i], diff[i + 1]
        if d0 == 0.0:
            v_star = V_range[i]
        elif d0 * d1 < 0.0:
            v0, v1 = V_range[i], V_range[i + 1]
            v_star = v0 - d0 * (v1 - v0) / (d1 - d0)
        else:
            continue

        w_star  = np.interp(v_star, V_range, w_null)
        fp      = np.interp(v_star, V_range, fprime)
        wp      = beta if v_star > mem['T'] else 0.0

        # Jacobian of (dV/dt, dw/dt)
        trace = settings.g * fp / settings.C - 1.0 / tau_w
        det   = (wp - settings.g * fp) / (settings.C * tau_w)

        if det < 0:
            kind = 'saddle'
        elif trace < 0:
            kind = 'stable'
        else:
            kind = 'unstable'
        pts.append((v_star, w_star, kind))
    return pts


def _vlines(ax, mem):
    for vline, label, color in [
        (mem['L'], "L (rest)",      "steelblue"),
        (mem['T'], "T (threshold)", "forestgreen"),
        (mem['H'], "H (plateau)",   "darkorange"),
    ]:
        ax.axvline(vline, color=color, linestyle=":", linewidth=1.1, alpha=0.6, label=label)


def plot_fw(beta=0.2, save=False):
    """f(V) and the recovery nullcline w_inf(V) on the same pane (vs V)."""
    mem = dict(BASELINE_MEMBRANE)
    fV  = compute_fV(mem)
    wV  = w_inf(mem, beta)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(V_range, fV, color="black", linewidth=2.0, label="f(V)", zorder=3)
    ax.plot(V_range, wV, color="crimson", linewidth=2.0,
            label=f"w_inf(V)  (beta={beta:.4g})", zorder=3)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    _vlines(ax, mem)

    ax.set_xlabel("V (mV)")
    ax.set_ylabel("f(V),  w_inf(V)")
    ax.set_title("Membrane nonlinearity f(V) and recovery nullcline w_inf(V)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(MEDIA_DIR, "nullclines_f_w.png"), dpi=150)
    plt.show()


def plot_nullclines(Iapp=0.0, beta=0.2, save=False):
    """Interactive (V, w) phase plane: V- and w-nullclines with fixed points."""
    mem = dict(BASELINE_MEMBRANE)
    fV  = compute_fV(mem)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    fig.subplots_adjust(bottom=0.24)

    _vlines(ax, mem)

    (v_line,) = ax.plot([], [], color="black", linewidth=2.0,
                        label="V-nullcline:  w = g·f(V) + Iapp", zorder=3)
    (w_line,) = ax.plot([], [], color="crimson", linewidth=2.0,
                        label="w-nullcline:  w = w_inf(V)", zorder=3)
    stable_sc   = ax.scatter([], [], s=80, facecolors="black", edgecolors="black",
                             zorder=6, label="stable")
    unstable_sc = ax.scatter([], [], s=80, facecolors="white", edgecolors="black",
                             zorder=6, label="unstable")
    saddle_sc   = ax.scatter([], [], s=90, marker="x", c="dimgray",
                             linewidths=2.0, zorder=6, label="saddle")

    ax.set_xlim(V_range[0], V_range[-1])
    ax.set_ylim(-2.0, max(8.0, BETA_MAX * (V_range[-1] - mem['T']) * 0.4))
    ax.set_xlabel("V (mV)")
    ax.set_ylabel("w  (recovery)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    ax_iapp = fig.add_axes([0.18, 0.10, 0.65, 0.035])
    ax_beta = fig.add_axes([0.18, 0.04, 0.65, 0.035])
    s_iapp = Slider(ax_iapp, "Iapp", IAPP_MIN, IAPP_MAX, valinit=Iapp, valstep=0.1)
    s_beta = Slider(ax_beta, "beta", BETA_MIN, BETA_MAX, valinit=beta, valstep=0.05)

    def update(_=None):
        I = s_iapp.val
        b = s_beta.val
        v_line.set_data(V_range, settings.g * fV + I)
        w_line.set_data(V_range, w_inf(mem, b))

        pts = fixed_points(mem, fV, I, b)
        by = {'stable': [], 'unstable': [], 'saddle': []}
        for v, w, kind in pts:
            by[kind].append((v, w))
        for sc, key in ((stable_sc, 'stable'), (unstable_sc, 'unstable'), (saddle_sc, 'saddle')):
            sc.set_offsets(np.array(by[key]) if by[key] else np.empty((0, 2)))

        summary = ", ".join(f"V*={v:.1f} ({k})" for v, _, k in pts) if pts else "none"
        ax.set_title(f"Phase plane  Iapp={I:+.1f}, beta={b:.2g}    fixed points: {summary}")
        fig.canvas.draw_idle()

    s_iapp.on_changed(update)
    s_beta.on_changed(update)
    update()

    if save:
        fig.savefig(os.path.join(MEDIA_DIR, "nullclines_phaseplane.png"), dpi=150)
    plt.show()


def run(plot='nullclines', **kwargs):
    """
    Dispatch to the chosen plot.

    Parameters
    ----------
    plot : {'nullclines', 'fw'}
        'nullclines' — interactive (V, w) phase plane (default).
        'fw'         — f(V) and w_inf(V) vs V on one pane.
    **kwargs
        Forwarded: plot_nullclines(Iapp=..., beta=..., save=...)
        or plot_fw(beta=..., save=...).
    """
    if plot == 'fw':
        plot_fw(**kwargs)
    elif plot == 'nullclines':
        plot_nullclines(**kwargs)
    else:
        raise ValueError(f"unknown plot {plot!r}; choose 'nullclines' or 'fw'")


if __name__ == "__main__":
    run('nullclines', Iapp=0.0, beta=0.3, save=False)
