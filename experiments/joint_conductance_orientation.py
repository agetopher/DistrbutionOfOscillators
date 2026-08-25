import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.colors import TwoSlopeNorm
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

import network
import settings
from analysis import phase_difference


"""Joint global-conductance screen for biologically correct wave orientation.

Connectivity and all within-type relative weights remain fixed. The screen
tests the Cartesian product of five values each for G_syne, G_syni, and G_gap
under separate steady AVA and AVB commands. This explores the 3-D parameter
volume rather than the three one-at-a-time lines tested previously.
"""

G_SYNE_VALUES = 0.07 * np.array([0.50, 0.75, 1.00, 1.25, 1.50])
G_SYNI_VALUES = 0.05 * np.array([0.50, 0.75, 1.00, 1.25, 1.50])
G_GAP_VALUES = 0.03 * np.array([0.50, 0.75, 1.00, 1.25, 1.50])
K_SYN = 0.25
DT = 20.0
TF = 10000.0
MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')


def _metrics(t, voltage):
    keep = t >= 0.5 * t[-1]
    time = t[keep]
    classes = network.classes
    segments = np.arange(network.N_CELLS) // 17
    dorsal_indices = np.where(classes == 8)[0]
    ventral_indices = np.where(classes == 9)[0]

    dorsal = np.array([
        voltage[dorsal_indices[segments[dorsal_indices] == segment]][:, keep].mean(axis=0)
        for segment in range(6)
    ])
    centered = dorsal - dorsal.mean(axis=1, keepdims=True)
    peaks, _ = find_peaks(centered[0], prominence=8.0)
    if len(peaks) < 3:
        slope = np.nan
        coherence = np.nan
        period = np.nan
    else:
        dt = float(np.median(np.diff(time)))
        period = float(np.median(np.diff(peaks)) * dt)
        max_lag = max(1, int(period / (2.0 * dt)))
        center = centered.shape[1] - 1
        lags = []
        for trace in centered:
            correlation = np.correlate(trace, centered[0], mode='full')
            window = correlation[center - max_lag:center + max_lag + 1]
            lags.append((np.argmax(window) - max_lag) * dt)
        position = np.arange(6)
        fit = np.polyfit(position, lags, 1)
        prediction = np.polyval(fit, position)
        residual = np.sum((np.asarray(lags) - prediction) ** 2)
        total = np.sum((np.asarray(lags) - np.mean(lags)) ** 2)
        slope = float(fit[0])
        coherence = float(1.0 - residual / total) if total > 0 else 1.0

    muscle_indices = np.where(np.isin(classes, (8, 9)))[0]
    amplitude = float(
        np.ptp(voltage[muscle_indices][:, keep], axis=1).mean()
    )
    dv_phases = []
    for segment in range(6):
        dorsal_mean = voltage[
            dorsal_indices[segments[dorsal_indices] == segment]
        ][:, keep].mean(axis=0)
        ventral_mean = voltage[
            ventral_indices[segments[ventral_indices] == segment]
        ][:, keep].mean(axis=0)
        dv_phases.append(
            phase_difference(
                time,
                dorsal_mean,
                ventral_mean,
                transient_frac=0.0,
                prominence=5.0,
            )
        )
    dv_phase = float(np.nanmean(dv_phases)) if np.any(np.isfinite(dv_phases)) else np.nan
    return {
        'gradient': slope,
        'coherence': coherence,
        'amplitude': amplitude,
        'dv_phase': dv_phase,
        'period': period,
    }


def _simulate(command, G_syne, G_syni, G_gap, seed=0):
    settings.reset_defaults()
    network.configure()
    settings.k_syn = K_SYN
    settings.G_syne = float(G_syne)
    settings.G_syni = float(G_syni)
    settings.G_gap = float(G_gap)

    rng = np.random.default_rng(seed)
    state = network.reduced_rest_state()
    state[:network.N_CELLS] += rng.normal(0.0, 2.0, network.N_CELLS)
    current = network.drive_vector(
        IAVA=2.5 if command == 'AVA' else 0.0,
        IAVB=2.5 if command == 'AVB' else 0.0,
    )
    time = np.arange(0.0, TF, DT)
    solution = solve_ivp(
        lambda _, x: network.rhs_vw(x, current),
        (0.0, TF),
        state,
        method='BDF',
        t_eval=time,
        max_step=50.0,
        rtol=1e-4,
        atol=1e-6,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return _metrics(solution.t, solution.y[:network.N_CELLS])


def _run_point(G_syne, G_syni, G_gap):
    return {
        'G_syne': float(G_syne),
        'G_syni': float(G_syni),
        'G_gap': float(G_gap),
        'AVA': _simulate('AVA', G_syne, G_syni, G_gap),
        'AVB': _simulate('AVB', G_syne, G_syni, G_gap),
    }


def is_candidate(record):
    """Strict biological orientation and wave-quality screen."""
    ava = record['AVA']
    avb = record['AVB']
    return (
        avb['gradient'] > 3.0
        and ava['gradient'] < -3.0
        and avb['coherence'] > 0.8
        and ava['coherence'] > 0.8
        and avb['amplitude'] > 10.0
        and ava['amplitude'] > 10.0
    )


def screen(n_jobs=-1):
    combinations = [
        (G_syne, G_syni, G_gap)
        for G_gap in G_GAP_VALUES
        for G_syni in G_SYNI_VALUES
        for G_syne in G_SYNE_VALUES
    ]
    return Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_point)(*combination)
        for combination in combinations
    )


def save_table(records):
    path = os.path.join(MEDIA_DIR, 'joint_conductance_orientation_screen.csv')
    fields = ['G_syne', 'G_syni', 'G_gap', 'command', 'gradient_ms_per_segment',
              'phase_profile_r2', 'muscle_excursion_mv', 'dv_phase', 'period_ms',
              'candidate']
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for command in ('AVA', 'AVB'):
                metric = record[command]
                writer.writerow({
                    'G_syne': record['G_syne'],
                    'G_syni': record['G_syni'],
                    'G_gap': record['G_gap'],
                    'command': command,
                    'gradient_ms_per_segment': metric['gradient'],
                    'phase_profile_r2': metric['coherence'],
                    'muscle_excursion_mv': metric['amplitude'],
                    'dv_phase': metric['dv_phase'],
                    'period_ms': metric['period'],
                    'candidate': is_candidate(record),
                })
    print(f'Saved → {path}')


def plot(records, save=True):
    finite_gradients = [
        abs(record[command]['gradient'])
        for record in records
        for command in ('AVA', 'AVB')
        if np.isfinite(record[command]['gradient'])
    ]
    limit = max(20.0, float(np.percentile(finite_gradients, 95)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(2, 5, figsize=(17, 7), sharex=True, sharey=True)
    image = None

    for column, G_gap in enumerate(G_GAP_VALUES):
        subset = [
            record for record in records
            if np.isclose(record['G_gap'], G_gap)
        ]
        for row, command in enumerate(('AVB', 'AVA')):
            matrix = np.full((len(G_SYNI_VALUES), len(G_SYNE_VALUES)), np.nan)
            invalid = np.zeros_like(matrix, dtype=bool)
            candidates = np.zeros_like(matrix, dtype=bool)
            for record in subset:
                i = int(np.argmin(abs(G_SYNI_VALUES - record['G_syni'])))
                j = int(np.argmin(abs(G_SYNE_VALUES - record['G_syne'])))
                metric = record[command]
                matrix[i, j] = metric['gradient']
                invalid[i, j] = (
                    not np.isfinite(metric['coherence'])
                    or metric['coherence'] <= 0.8
                    or metric['amplitude'] <= 10.0
                )
                candidates[i, j] = is_candidate(record)
            axis = axes[row, column]
            image = axis.imshow(
                matrix,
                origin='lower',
                aspect='auto',
                cmap='coolwarm',
                norm=norm,
                extent=[
                    G_SYNE_VALUES[0], G_SYNE_VALUES[-1],
                    G_SYNI_VALUES[0], G_SYNI_VALUES[-1],
                ],
            )
            for i, G_syni in enumerate(G_SYNI_VALUES):
                for j, G_syne in enumerate(G_SYNE_VALUES):
                    if invalid[i, j]:
                        axis.plot(G_syne, G_syni, 'x', color='k', ms=6, mew=1.2)
                    if candidates[i, j]:
                        axis.plot(G_syne, G_syni, '*', color='gold', mec='k', ms=13)
            if row == 0:
                axis.set_title(f'$G_{{gap}}={G_gap:.4f}$ nS')
            if column == 0:
                target = 'target: positive' if command == 'AVB' else 'target: negative'
                axis.set_ylabel(f'{command} ({target})\n$G_{{syni}}$ (nS)')
            if row == 1:
                axis.set_xlabel('$G_{syne}$ (nS)')

    colorbar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    colorbar.set_label('wave gradient (ms/segment)')
    count = sum(is_candidate(record) for record in records)
    fig.suptitle(
        'Joint global-conductance orientation screen '
        f'($k_{{syn}}={K_SYN}$; direct steady commands)\n'
        f'gold star = both directions pass; black × = weak/incoherent; '
        f'candidates: {count}/{len(records)}',
        y=0.995,
    )
    fig.subplots_adjust(left=0.07, right=0.9, bottom=0.09, top=0.86,
                        hspace=0.2, wspace=0.12)
    if save:
        path = os.path.join(MEDIA_DIR, 'joint_conductance_orientation_screen.png')
        fig.savefig(path, dpi=150)
        print(f'Saved → {path}')
    return fig


def summarize(records):
    candidates = [record for record in records if is_candidate(record)]
    print(
        f'Joint conductance orientation screen: '
        f'{len(candidates)}/{len(records)} strict candidates'
    )
    for record in candidates:
        print(
            f"  G_syne={record['G_syne']:.4f}, "
            f"G_syni={record['G_syni']:.4f}, G_gap={record['G_gap']:.4f}: "
            f"AVB={record['AVB']['gradient']:+.1f} "
            f"(R2={record['AVB']['coherence']:.2f}), "
            f"AVA={record['AVA']['gradient']:+.1f} "
            f"(R2={record['AVA']['coherence']:.2f})"
        )
    return candidates


def run(save=True, n_jobs=-1):
    records = screen(n_jobs=n_jobs)
    candidates = summarize(records)
    save_table(records)
    plot(records, save=save)
    return records, candidates


if __name__ == '__main__':
    run()
