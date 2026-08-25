# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

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

# Small circuit experiments
python experiments/single_neuron_ramp.py
python experiments/two_neuron_ii.py
python experiments/three_neuron_eig.py

# Parameter sweeps
python experiments/sweep_two_ii.py
python experiments/sweep_two_ei.py
python experiments/sweep_three_eig.py

# Connectivity visualization
python experiments/connectivity.py

# Run tests
pytest
```

Output figures are saved to `media/`. Filenames encode parameter values and synapse config (e.g., `test_two_II_Iapp3_sf030.0_sf140.0_yuval.png`).

## Architecture

### `src/settings.py` — Global parameter store
All scripts share parameters through this module. Importing it supplies the
current learned baseline (`G_syne=0.07`, `G_syni=0.05`, `G_gap=0.03`,
`beta=1.03`, `tau_w=400 ms`). Call `settings.reset_defaults()` before
configuring an experiment that shares a Python process with other experiments.
Connectivity matrices and applied-current vectors remain circuit-specific.

### `src/functions.py` — Neuron model equations
Implements the piecewise-linear Yuval neuron model:
- `f(v)` / `f_vec(v)` — voltage-dependent membrane conductance (scalar and vectorized)
- `sf_vec(v, s)` — synaptic fatigue dynamics

### `src/network.py` — Full 102-cell ODE network
`run(Iapp=0, tf=5000, save=True)` integrates the full network using `scipy.integrate.solve_ivp` with the BDF method (stiff solver). Combines membrane dynamics, synaptic transmission, synaptic fatigue, and gap junction coupling. Loads connectivity from `data/`.

### `src/segment.py` — Reusable 17-cell segment model
Owns the first-segment connectivity, AVA/AVB drive construction, initial states,
and the full `[V, SF, w]` and reduced `[V, w]` right-hand sides. Experiments
must import this model layer instead of importing functions from other
experiment scripts.

### `src/analysis.py` — Post-processing
- `oscillation_metric(t, V, ...)` — detects oscillations, returns mean ISI and CV
- `phase_difference(t, V0, V1, ...)` — estimates normalized phase offset between neurons (0=in-phase, 0.5=antiphase)

### `experiments/` — Runnable scientific analyses
Short-circuit studies, parameter sweeps, bifurcation analysis, nullclines, and
network-wave experiments. These scripts generate figures and are not tests.
Each standalone experiment must call `settings.reset_defaults()` before
declaring only its intentional parameter overrides.

### `tests/` — Automated regression tests
Fast checks for the shared neuron equations and analysis utilities. Long
simulations do not belong here.

### `data/` — Network connectivity and configuration
- `ConnectivityMatrix_SixSegments_*.txt` — three 102×102 matrices (excitatory synapses, inhibitory synapses, gap junctions); row=post, col=pre
- `InitialVoltages.dat` — starting membrane potentials for all 102 cells
- `CellsClassification.dat` — cell type labels (1–9) for all 102 cells
- `OscillatorComb/` — 128 binary classification files specifying which non-muscle cells act as oscillators in each configuration

## Key Concepts

**Synapse configurations:** Supported experiment scripts accept a `synapse_config` argument (`'yuval'` or `'boyle'`). The Yuval config uses a shallow sigmoid; the Boyle config uses a steep sigmoid derived from electrophysiology data.

**Synaptic fatigue:** Each synapse has a fatigue variable `s` that adapts over time, controlled by parameters `a` and `b`. Initial values of `s` are varied in sweeps to explore different network states.

**OscillatorComb files:** Specify which cells are intrinsic oscillators vs. passive followers in a given configuration. Used to test whether different subsets of the network can sustain rhythmic activity.

## Vault Sync

This project is tracked in an Obsidian vault at:
`/Users/christopheragesen/Library/Mobile Documents/iCloud~md~obsidian/Documents/Batin/`

The project note lives at:
`003 Collaboration/Distribution of Oscillators/DoO - Overview.md`

At the end of any substantive work session, append a dated entry to that note summarizing:
- What was worked on
- Key decisions or results
- Open questions or next steps

Keep entries short (3-5 lines). Do not overwrite existing content — append below the last entry.
