# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Computational model of *C. elegans* forward locomotion. The goal is to find synaptic conductance parameters (G_syne, G_syni, G_gap) that produce oscillatory motor activity — specifically dorsoventral antiphase oscillation with an anterior-to-posterior travelling wave across six body segments.

The network has 102 cells organized into 9 classes: classes 1–7 are interneurons and motor neurons (driven oscillators), classes 8–9 are dorsal and ventral muscles (pure receivers).

## Environment Setup

```bash
conda env create -f environment.yml
conda activate simple-worm-scripts
```

Python 3.9, key dependencies: numpy, scipy, matplotlib, pytest, joblib.

## Running Code

```bash
# Full 102-cell network simulation
python src/network.py

# Small circuit tests
python tests/test_ramp.py          # Single neuron under triangular ramp
python tests/test_two_II.py        # Two mutually inhibitory neurons
python tests/test_three_EIG.py     # Three-neuron E/I/gap circuit

# Parameter sweeps
python tests/sweep_two_II.py       # G_syni sweep for 2-cell mutual inhibition
python tests/sweep_two_EI.py       # G_syne × G_syni × Iapp sweep
python tests/sweep_three_EIG.py    # G_syne × G_syni × G_gap sweep

# Connectivity visualization
python tests/plot_connectivity.py

# Run tests
pytest
```

Output figures are saved to `media/`. Filenames encode parameter values and synapse config (e.g., `test_two_II_Iapp3_sf030.0_sf140.0_yuval.png`).

## Architecture

### `src/settings.py` — Global parameter store
All scripts share parameters through this module via `settings.init()`. Holds membrane constants (C, g, L, T, H, m1–m4), synaptic parameters (k_syn, V_th), fatigue parameters (a, b), conductances (G_syne, G_syni, G_gap), and the loaded connectivity matrices.

### `src/functions.py` — Neuron model equations
Implements the piecewise-linear Yuval neuron model:
- `f(v)` / `f_vec(v)` — voltage-dependent membrane conductance (scalar and vectorized)
- `sf_vec(v, s)` — synaptic fatigue dynamics

### `src/network.py` — Full 102-cell ODE network
`run(Iapp=0, tf=5000, save=True)` integrates the full network using `scipy.integrate.solve_ivp` with the BDF method (stiff solver). Combines membrane dynamics, synaptic transmission, synaptic fatigue, and gap junction coupling. Loads connectivity from `data/`.

### `src/analysis.py` — Post-processing
- `oscillation_metric(t, V, ...)` — detects oscillations, returns mean ISI and CV
- `phase_difference(t, V0, V1, ...)` — estimates normalized phase offset between neurons (0=in-phase, 0.5=antiphase)

### `data/` — Network connectivity and configuration
- `ConnectivityMatrix_SixSegments_*.txt` — three 102×102 matrices (excitatory synapses, inhibitory synapses, gap junctions); row=post, col=pre
- `InitialVoltages.dat` — starting membrane potentials for all 102 cells
- `CellsClassification.dat` — cell type labels (1–9) for all 102 cells
- `OscillatorComb/` — 128 binary classification files specifying which non-muscle cells act as oscillators in each configuration

## Key Concepts

**Synapse configurations:** All test/sweep scripts accept a `synapse_config` argument (`'yuval'` or `'boyle'`). The Yuval config uses a shallow sigmoid; the Boyle config uses a steep sigmoid derived from electrophysiology data.

**Synaptic fatigue:** Each synapse has a fatigue variable `s` that adapts over time, controlled by parameters `a` and `b`. Initial values of `s` are varied in sweeps to explore different network states.

**OscillatorComb files:** Specify which cells are intrinsic oscillators vs. passive followers in a given configuration. Used to test whether different subsets of the network can sustain rhythmic activity.
