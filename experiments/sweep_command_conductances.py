import os
import sys
import csv
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

import ava_master_command
import network
import settings
from analysis import phase_difference


"""One-at-a-time conductance sensitivity under the Meng command protocol.

Each point runs tonic AVB → phasic AVA → AVB recovery with unchanged
connectivity. A useful parameter region must satisfy all three tests:

1. AVB/forward gives an anterior→posterior wave (positive gradient);
2. AVA/backward gives a posterior→anterior wave (negative gradient);
3. muscle oscillations and dorsoventral antiphase remain intact.
"""

BASELINES = {'G_syne': 0.07, 'G_syni': 0.05, 'G_gap': 0.03}
MULTIPLIERS = np.array([0.50, 0.75, 1.00, 1.25, 1.50])
PHASES = {
    'AVB before': (5000.0, 10000.0),
    'AVA event': (12000.0, 20000.0),
    'AVB recovery': (22000.0, 30000.0),
}
MEDIA_DIR = os.path.join(os.path.dirname(__file__), '..', 'media')

def _phase_metrics(t, voltage, start, end):
    mask = (t >= start) & (t < end)
    time = t[mask]
    dorsal_indices = np.where(network.classes == 8)[0]
    ventral_indices = np.where(network.classes == 9)[0]
    segments = np.arange(network.N_CELLS) // 17

    dorsal = np.array([
        voltage[dorsal_indices[segments[dorsal_indices] == segment]][:, mask].mean(axis=0)
        for segment in range(6)
    ])
    centered = dorsal - dorsal.mean(axis=1, keepdims=True)
    peaks, _ = find_peaks(centered[0], prominence=8.0)
    if len(peaks) < 3:
        slope = np.nan
        coherence = np.nan
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
        predicted = np.polyval(fit, position)
        residual = np.sum((np.asarray(lags) - predicted) ** 2)
        total = np.sum((np.asarray(lags) - np.mean(lags)) ** 2)
        slope = float(fit[0])
        coherence = float(1.0 - residual / total) if total > 0 else 1.0

    dv_phases = []
    for segment in range(6):
        dorsal_mean = voltage[
            dorsal_indices[segments[dorsal_indices] == segment]
        ][:, mask].mean(axis=0)
        ventral_mean = voltage[
            ventral_indices[segments[ventral_indices] == segment]
        ][:, mask].mean(axis=0)
        dv_phases.append(
            phase_difference(
                time,
                dorsal_mean,
                ventral_mean,
                transient_frac=0.0,
                prominence=5.0,
            )
        )
    amplitude = 0.5 * (
        np.ptp(voltage[dorsal_indices][:, mask], axis=1).mean()
        + np.ptp(voltage[ventral_indices][:, mask], axis=1).mean()
    )
    dv_phase = float(np.nanmean(dv_phases)) if np.any(np.isfinite(dv_phases)) else np.nan
    return slope, coherence, float(amplitude), dv_phase


def _simulate_direct(conductances, tf=30000.0, dt=20.0, k_syn=0.25):
    """Abrupt independent AVB→AVA→AVB blocks, with no Meng filter."""
    settings.reset_defaults()
    network.configure()
    settings.k_syn = float(k_syn)
    for name, value in conductances.items():
        setattr(settings, name, float(value))

    protocol = (
        ('AVB', 0.0, 10000.0),
        ('AVA', 10000.0, 20000.0),
        ('AVB', 20000.0, tf),
    )
    time = np.arange(0.0, tf + 0.5 * dt, dt)
    voltage = np.zeros((network.N_CELLS, time.size))
    state = network.reduced_rest_state()
    for command, start, end in protocol:
        block_time = np.arange(start, end + 0.5 * dt, dt)
        current = network.drive_vector(
            IAVA=2.5 if command == 'AVA' else 0.0,
            IAVB=2.5 if command == 'AVB' else 0.0,
        )
        solution = solve_ivp(
            lambda _, x: network.rhs_vw(x, current),
            (start, end),
            state,
            method='BDF',
            t_eval=block_time,
            max_step=50.0,
            rtol=1e-5,
            atol=1e-7,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        indices = np.searchsorted(time, block_time)
        voltage[:, indices] = solution.y[:network.N_CELLS]
        state = solution.y[:, -1]
    return time, voltage


def _run_point(parameter, multiplier, command_model='meng', k_syn=0.25):
    value = BASELINES[parameter] * multiplier
    conductances = {parameter: value}
    if command_model == 'meng':
        t, voltage, *_ = ava_master_command.simulate(
            pulse=(10000.0, 20000.0),
            tf=30000.0,
            dt=20.0,
            conductances=conductances,
            k_syn=k_syn,
        )
    elif command_model == 'direct':
        t, voltage = _simulate_direct(conductances, k_syn=k_syn)
    else:
        raise ValueError(f'Unknown command model: {command_model!r}')
    metrics = {
        phase: _phase_metrics(t, voltage, start, end)
        for phase, (start, end) in PHASES.items()
    }
    return {
        'parameter': parameter,
        'multiplier': float(multiplier),
        'value': float(value),
        'command_model': command_model,
        'k_syn': float(k_syn),
        'metrics': metrics,
    }


def sweep(n_jobs=-1, command_models=('meng',), k_syn=0.25):
    """Run each one-at-a-time scan in independent processes."""
    jobs = [
        (parameter, multiplier, command_model)
        for command_model in command_models
        for parameter in BASELINES
        for multiplier in MULTIPLIERS
    ]
    return Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(_run_point)(parameter, multiplier, command_model, k_syn)
        for parameter, multiplier, command_model in jobs
    )


def _records_for(results, parameter, command_model='meng'):
    return sorted(
        (
            record for record in results
            if record['parameter'] == parameter
            and record['command_model'] == command_model
        ),
        key=lambda record: record['value'],
    )


def plot(results, save=True, command_model='meng'):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    colors = {'AVB before': 'steelblue', 'AVA event': 'crimson',
              'AVB recovery': 'navy'}
    labels = {
        'G_syne': r'$G_{\rm syne}$ (nS)',
        'G_syni': r'$G_{\rm syni}$ (nS)',
        'G_gap': r'$G_{\rm gap}$ (nS)',
    }

    for column, parameter in enumerate(BASELINES):
        records = _records_for(results, parameter, command_model)
        x = np.array([record['value'] for record in records])

        for phase in PHASES:
            values = np.array([record['metrics'][phase][0] for record in records])
            axes[0, column].plot(
                x, values, 'o-', color=colors[phase],
                label=phase if column == 0 else None,
            )
        axes[0, column].axhline(0.0, color='0.4', lw=0.8)
        axes[0, column].axvline(BASELINES[parameter], color='0.5', ls=':', lw=1.0)
        axes[0, column].set_title(labels[parameter])
        axes[0, column].set_ylabel('wave gradient\n(ms/segment)'
                                   if column == 0 else '')

        for phase in PHASES:
            values = np.array([record['metrics'][phase][2] for record in records])
            axes[1, column].plot(x, values, 'o-', color=colors[phase])
        axes[1, column].axhline(10.0, color='0.5', ls='--', lw=0.8)
        axes[1, column].axvline(BASELINES[parameter], color='0.5', ls=':', lw=1.0)
        axes[1, column].set_ylabel('mean muscle\nexcursion (mV)' if column == 0 else '')

        for phase in PHASES:
            values = np.array([record['metrics'][phase][3] for record in records])
            axes[2, column].plot(x, values, 'o-', color=colors[phase])
        axes[2, column].axhline(0.5, color='0.5', ls='--', lw=0.8)
        axes[2, column].axvline(BASELINES[parameter], color='0.5', ls=':', lw=1.0)
        axes[2, column].set_ylim(-0.02, 0.52)
        axes[2, column].set_ylabel('dorsoventral phase\n(0.5 = antiphase)'
                                   if column == 0 else '')
        axes[2, column].set_xlabel(labels[parameter])

    fig.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(
        'Conductance sensitivity under the Meng AVA-master protocol\n'
        'one parameter varied at a time; vertical dotted line = current baseline',
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    if save:
        path = os.path.join(MEDIA_DIR, 'meng_command_conductance_sweep.png')
        fig.savefig(path, dpi=150)
        print(f'Saved → {path}')
    return fig


def summarize(results):
    for command_model in sorted({record['command_model'] for record in results}):
        print(f'\nCommand model: {command_model}')
        for parameter in BASELINES:
            print(f'\n{parameter} (baseline {BASELINES[parameter]:.3f} nS)')
            print(' value   AVB-before   AVA-event   AVB-recovery   amp-min  DV-min  R2-min')
            for record in _records_for(results, parameter, command_model):
                metrics = record['metrics']
                slopes = [metrics[phase][0] for phase in PHASES]
                amplitudes = [metrics[phase][2] for phase in PHASES]
                dv_phases = [metrics[phase][3] for phase in PHASES]
                coherences = [metrics[phase][1] for phase in PHASES]
                corrected = (
                    slopes[0] > 0 and slopes[1] < 0 and slopes[2] > 0
                    and np.nanmin(amplitudes) > 10.0
                    and np.nanmin(dv_phases) > 0.35
                    and np.nanmin(coherences) > 0.8
                )
                marker = '  CORRECTED' if corrected else ''
                dv_min = (
                    np.nanmin(dv_phases) if np.any(np.isfinite(dv_phases)) else np.nan
                )
                coherence_min = (
                    np.nanmin(coherences) if np.any(np.isfinite(coherences)) else np.nan
                )
                print(
                    f" {record['value']:.4f}  "
                    f"{slopes[0]:+8.2f}    {slopes[1]:+8.2f}    {slopes[2]:+8.2f}"
                    f"      {np.nanmin(amplitudes):5.1f}    {dv_min:.2f}"
                    f"    {coherence_min:.2f}"
                    f'{marker}'
                )


def save_table(results, filename='meng_command_conductance_sweep.csv'):
    path = os.path.join(MEDIA_DIR, filename)
    fields = ['command_model', 'k_syn', 'parameter', 'multiplier', 'value', 'phase',
              'gradient_ms_per_segment', 'phase_profile_r2',
              'muscle_excursion_mv', 'dv_phase']
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in results:
            for phase, metrics in record['metrics'].items():
                writer.writerow({
                    'command_model': record['command_model'],
                    'k_syn': record['k_syn'],
                    'parameter': record['parameter'],
                    'multiplier': record['multiplier'],
                    'value': record['value'],
                    'phase': phase,
                    'gradient_ms_per_segment': metrics[0],
                    'phase_profile_r2': metrics[1],
                    'muscle_excursion_mv': metrics[2],
                    'dv_phase': metrics[3],
                })
    print(f'Saved → {path}')


def plot_comparison(results, save=True, k_syn=0.25):
    """Overlay Meng (solid) and direct commands (dashed) for wave gradients."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    colors = {'AVB before': 'steelblue', 'AVA event': 'crimson',
              'AVB recovery': 'navy'}
    labels = {
        'G_syne': r'$G_{\rm syne}$ (nS)',
        'G_syni': r'$G_{\rm syni}$ (nS)',
        'G_gap': r'$G_{\rm gap}$ (nS)',
    }
    for axis, parameter in zip(axes, BASELINES):
        for command_model, linestyle in (('meng', '-'), ('direct', '--')):
            records = _records_for(results, parameter, command_model)
            x = np.array([record['value'] for record in records])
            for phase in PHASES:
                y = np.array([record['metrics'][phase][0] for record in records])
                axis.plot(
                    x, y, marker='o', ls=linestyle, color=colors[phase],
                    alpha=0.9,
                )
        axis.axhline(0.0, color='0.4', lw=0.8)
        axis.axvline(BASELINES[parameter], color='0.5', ls=':', lw=1.0)
        axis.set_title(labels[parameter])
        axis.set_xlabel(labels[parameter])
    axes[0].set_ylabel('wave gradient (ms/segment)')

    phase_handles = [
        Line2D([0], [0], color=colors[phase], marker='o', label=phase)
        for phase in PHASES
    ]
    model_handles = [
        Line2D([0], [0], color='0.25', ls='-', label='Meng filter'),
        Line2D([0], [0], color='0.25', ls='--', label='direct blocks'),
    ]
    fig.legend(
        handles=phase_handles + model_handles,
        loc='upper center',
        ncol=5,
        bbox_to_anchor=(0.5, 0.94),
    )
    fig.suptitle(
        'Does the conductance sensitivity depend on the Meng command dynamics?\n'
        f'$k_{{syn}}={k_syn}$',
        y=0.995,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    if save:
        suffix = str(k_syn).replace('.', 'p')
        path = os.path.join(
            MEDIA_DIR,
            f'command_model_conductance_comparison_ksyn_{suffix}.png',
        )
        fig.savefig(path, dpi=150)
        print(f'Saved → {path}')
    return fig


def run(save=True, n_jobs=-1):
    results = sweep(n_jobs=n_jobs)
    summarize(results)
    save_table(results)
    plot(results, save=save)
    return results


def compare_command_models(save=True, n_jobs=-1, k_syn=0.25):
    results = sweep(
        n_jobs=n_jobs,
        command_models=('meng', 'direct'),
        k_syn=k_syn,
    )
    summarize(results)
    suffix = str(k_syn).replace('.', 'p')
    save_table(results, f'command_model_conductance_sweep_ksyn_{suffix}.csv')
    plot_comparison(results, save=save, k_syn=k_syn)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k-syn', type=float, default=0.25)
    arguments = parser.parse_args()
    compare_command_models(k_syn=arguments.k_syn)
