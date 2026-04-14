import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import settings

"""
Nullcline analysis for the Yuval piecewise-linear neuron model.

Compares a baseline set of membrane parameters (L, T, H) against an
alternative set so you can see how the nullclines shift.

Figure 1 — Single-neuron geometry (both parameter sets overlaid):
  Left:  f(V) curves for baseline vs alternative.
  Right: V-nullcline diagram — fixed points where g·f(V) + Iapp = 0.

Figure 2 — Half-center oscillator phase plane (V_suppressed, h):
  Left panel: baseline membrane parameters.
  Right panel: alternative membrane parameters.
  Both panels use the same synapse config ('yuval' or 'boyle').

Usage:
  run()                                          # default: shift T and H up by 5 mV
  run(synapse_config='boyle')                    # same comparison with Boyle synapses
  run(params_alt={'T': -40.0, 'H': -28.0})      # custom alternative
  run(params_alt={'T': -50.0, 'H': -38.0, 'L': -72.0})
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

# ── Membrane constants that do not change between comparisons ─────────────────
settings.C = 7.0    # pF
settings.g = 1.0    # nS
settings.E_syni = -100.0

# ── Synapse defaults per config ───────────────────────────────────────────────
SYNAPSE_DEFAULTS = {
    'yuval': dict(G_syni=0.5,  k_syn=0.5,   V_th=-52.0),
    'boyle': dict(G_syni=0.3,  k_syn=100.0, V_th=-70.0),
}

MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')

V_range = np.linspace(-85, -20, 700)

IAPP_VALS   = [2, 3, 4]
IAPP_COLORS = ["steelblue", "forestgreen", "crimson"]


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


def sigma(v, k, Vth):
    return 1.0 / (1.0 + np.exp(-k * (v - Vth)))


def _vlines(mem):
    return [
        (mem['L'], "L (rest)",      "steelblue"),
        (mem['T'], "T (threshold)", "forestgreen"),
        (mem['H'], "H (plateau)",   "darkorange"),
    ]


def _plot_fV_panel(ax, mem_base, fV_base, mem_alt, fV_alt):
    """Overlay f(V) curves for baseline and alternative on one axis."""
    ax.plot(V_range, fV_base, color="black",  linewidth=2.0, label="f(V) baseline",    zorder=3)
    ax.plot(V_range, fV_alt,  color="dimgray", linewidth=2.0, label="f(V) alternative",
            linestyle="--", zorder=3)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    for vline, label, color in _vlines(mem_base):
        ax.axvline(vline, color=color, linestyle=":",  linewidth=1.1, alpha=0.7, label=f"{label} (base)")
    for vline, label, color in _vlines(mem_alt):
        ax.axvline(vline, color=color, linestyle="-.", linewidth=1.1, alpha=0.55, label=f"{label} (alt)")
    ax.set_xlabel("V (mV)")
    ax.set_ylabel("f(V)")
    ax.set_title("Membrane nonlinearity f(V)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _plot_nullcline_panel(ax, mem_base, fV_base, mem_alt, fV_alt):
    """Fixed-point diagram: f(V) vs -Iapp/g for baseline and alternative."""
    ax.plot(V_range, fV_base, color="black",  linewidth=2.0, label="f(V) baseline",    zorder=3)
    ax.plot(V_range, fV_alt,  color="dimgray", linewidth=2.0, label="f(V) alternative",
            linestyle="--", zorder=3)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    for Iapp, color in [(-1, "purple"), (0, "royalblue"), (2, "forestgreen"), (4, "crimson")]:
        ax.axhline(-Iapp / settings.g, color=color, linestyle="--", linewidth=1.2,
                   label=f"−Iapp/g  (Iapp={Iapp})")
    for vline, _, color in _vlines(mem_base):
        ax.axvline(vline, color=color, linestyle=":",  linewidth=0.9, alpha=0.5)
    for vline, _, color in _vlines(mem_alt):
        ax.axvline(vline, color=color, linestyle="-.", linewidth=0.9, alpha=0.4)
    ax.set_xlabel("V (mV)")
    ax.set_ylabel("f(V)")
    ax.set_title("Fixed points: intersections with −Iapp/g")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _plot_halfcenter_panel(ax, mem, fV, syn, title):
    """V-nullclines and h-nullcline for the half-center phase plane."""
    sig_active = sigma(mem['H'], k=syn['k_syn'], Vth=syn['V_th'])
    denom      = syn['G_syni'] * sig_active * (V_range - settings.E_syni)

    for Iapp, color in zip(IAPP_VALS, IAPP_COLORS):
        numer  = settings.g * fV + Iapp
        h_null = np.where(np.abs(denom) > 1e-10, numer / denom, np.nan)
        h_null = np.where((h_null >= 0) & (h_null <= 1.0), h_null, np.nan)
        ax.plot(V_range, h_null, color=color, linewidth=2.0,
                label=f"V-nullcline  Iapp={Iapp}")

    ax.axhline(0, color="black", linestyle="--", linewidth=1.6,
               label="h-nullcline  (h = 0)")

    for vline, label, color in _vlines(mem):
        ax.axvline(vline, color=color, linestyle=":", linewidth=1.0, alpha=0.55, label=label)

    ax.set_xlim(V_range[0], V_range[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("V_suppressed (mV)")
    ax.set_ylabel("h  (synaptic fatigue)")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)


def run(synapse_config='yuval', params_alt=None, save=True):
    """
    Parameters
    ----------
    synapse_config : 'yuval' or 'boyle'
        Synapse parameter set to use in the half-center phase plane.
    params_alt : dict, optional
        Override any membrane parameters for the right panel.
        Keys: 'L', 'T', 'H', 'm1', 'm2', 'm3', 'm4'.
        Defaults to shifting T and H up by 5 mV each.
    save : bool
        Save figures to media/.
    """
    mem_base = dict(BASELINE_MEMBRANE)

    if params_alt is None:
        params_alt = {'T': mem_base['T'] + 5.0, 'H': mem_base['H'] + 5.0}

    mem_alt = dict(mem_base)
    mem_alt.update(params_alt)

    fV_base = compute_fV(mem_base)
    fV_alt  = compute_fV(mem_alt)

    syn = dict(SYNAPSE_DEFAULTS[synapse_config])

    # Build a readable label for what changed
    changed = [k for k in params_alt if mem_base.get(k) != mem_alt.get(k)]
    change_str = ",  ".join(f"{k}: {mem_base[k]:.4g} → {mem_alt[k]:.4g}" for k in changed)

    base_label = f"L={mem_base['L']:.4g}  T={mem_base['T']:.4g}  H={mem_base['H']:.4g}"
    alt_label  = f"L={mem_alt['L']:.4g}  T={mem_alt['T']:.4g}  H={mem_alt['H']:.4g}"

    # ── Figure 1: f(V) comparison ─────────────────────────────────────────────
    fig1, (ax_fv, ax_null) = plt.subplots(1, 2, figsize=(13, 4))
    fig1.suptitle(f"Single-Neuron Nullcline Analysis  [{change_str}]", fontsize=12)

    _plot_fV_panel(ax_fv, mem_base, fV_base, mem_alt, fV_alt)
    _plot_nullcline_panel(ax_null, mem_base, fV_base, mem_alt, fV_alt)

    fig1.tight_layout()
    if save:
        fig1.savefig(
            os.path.join(MEDIA_DIR, f"nullclines_single_neuron_{synapse_config}.png"), dpi=150)

    # ── Figure 2: Half-center phase plane ─────────────────────────────────────
    fig2, (ax_base, ax_alt) = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle(
        f"Half-Center Oscillator Phase Plane  [{synapse_config} synapses]\n"
        f"h = synaptic fatigue of the active cell's synapse",
        fontsize=12,
    )

    _plot_halfcenter_panel(ax_base, mem_base, fV_base, syn,
                           f"Baseline\n{base_label}")
    _plot_halfcenter_panel(ax_alt,  mem_alt,  fV_alt,  syn,
                           f"Alternative  ({change_str})\n{alt_label}")

    fig2.tight_layout()
    if save:
        fig2.savefig(
            os.path.join(MEDIA_DIR, f"nullclines_halfcenter_{synapse_config}.png"), dpi=150)

    plt.show()


if __name__ == "__main__":
    # Default: Yuval synapses, T and H shifted up 5 mV
    run(synapse_config='yuval', params_alt={'T': -40.0})

    # Uncomment to use Boyle synapse config:
    # run(synapse_config='boyle')

    # Uncomment to specify your own alternative membrane parameters:
    # run(params_alt={'T': -40.0, 'H': -28.0})
    # run(params_alt={'T': -50.0, 'H': -38.0, 'L': -72.0})
    # run(synapse_config='boyle', params_alt={'T': -42.0, 'H': -30.0})
