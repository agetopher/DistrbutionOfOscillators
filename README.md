# Distribution of Oscillators

**AVA/AVB bifurcations and XPP code:** [start here](docs/ava_avb_handoff/README.md) for the current diagrams, model download, parameter baseline, and reproduction instructions.

Computational model of *C. elegans* forward locomotion using a minimal neural circuit.
The goal is to find synaptic conductance parameters (G\_syne, G\_syni, G\_gap) that
produce locomotion-like activity: dorsoventral antiphase oscillation with an
anterior-to-posterior travelling wave across six body segments.

## Scientific Background

The network consists of 102 cells organised into 9 classes. Classes 1–7 are
interneurons and motor neurons; classes 8 and 9 are the dorsal and ventral muscles
respectively (pure output layer — they receive synaptic input but send none).
Oscillatory behaviour is driven by subsets of the non-muscle cells designated as
oscillators via the OscillatorComb files.

The current neuron model is a piecewise-linear approximation (Yuval model) with
synaptic fatigue. Known limitation: the model is bistable, with stable fixed points
at the resting potential and the plateau potential. This can cause network-level
plateau locking under strong excitatory drive. Alternatives are documented in
`docs/neuron_model_comparison/neuron_model_comparison.tex`.

Two synapse configurations are supported across the test and sweep scripts:

- **Yuval** (`synapse_config='yuval'`): shallow sigmoid (k = 0.125, V\_th = −52 mV),
  activates near threshold. Parameters hand-tuned for rhythmic activity (Yuval thesis, Table 6).
- **Boyle** (`synapse_config='boyle'`): steep sigmoid activating at rest (k\_inh = 100,
  k\_exc = 500, V\_th = −70 mV), derived from RMD electrophysiology (Mellem 2008, Liu 2009).
  Synaptic fatigue is retained in both configurations.

## Repository Structure

```
DistrbutionOfOscillators/
├── src/
│   ├── settings.py       # Learned baseline parameters + reset_defaults()
│   ├── functions.py      # Yuval model equations: f(v), f_vec(v), sf_vec(v, s)
│   ├── segment.py        # Reusable 17-cell segment, drives, and ODEs
│   ├── network.py        # Full 102-cell ODE + run() — loads connectivity from data/
│   └── analysis.py       # Post-processing: oscillation_metric(), phase_difference()
│
├── experiments/              # Runnable scientific analyses and figure generation
│   ├── bifurcations.py       # Equilibrium continuation and stability analysis
│   ├── nullclines.py         # Single-cell phase-plane analysis
│   ├── single_neuron.py      # Single-neuron pulse experiment
│   ├── single_segment.py     # One 17-cell segment
│   ├── network_alternation.py
│   ├── wave_direction.py
│   └── sweep_*.py            # Parameter sweeps
│
├── tests/                    # Fast automated regression tests
│   ├── test_analysis.py
│   ├── test_model_equations.py
│   └── test_segment.py
│
├── xppaut/
│   ├── single_segment.ode            # 17-cell reduced model for XPPAUT/AUTO
│   └── generate_single_segment.py    # Regenerates it from src/ and data/
│
├── data/
│   ├── ConnectivityMatrix_SixSegments_ExcitatorySynapses.txt
│   ├── ConnectivityMatrix_SixSegments_InhibitorySynapses.txt
│   ├── ConnectivityMatrix_SixSegments_GapJunctions.txt
│   ├── InitialVoltages.dat       # Starting voltages for all 102 cells
│   ├── CellsClassification.dat   # Cell type label (1–9) for each cell
│   └── OscillatorComb/           # 128 files — each specifies one oscillator configuration
│
├── docs/
│   └── neuron_model_comparison/
│       └── neuron_model_comparison.tex   # LaTeX comparison of alternative neuron models
│
├── media/                # Generated figures (git-ignored)
├── environment.yml       # Conda environment (simple-worm-scripts, Python 3.9)
└── README.md
```

## Neuron Model

Each cell follows the piecewise-linear Yuval model:

```
C dV/dt = g·f(V) - I_syn + I_app

f(V) = m1·(-V + L)                    V < L
       m2·(V - L)·(V - T)             L ≤ V ≤ T
       m3·(V - H)·(V - T)             T < V ≤ H
       m4·(-V + H)                    V > H
```

Key voltage levels: L = −70 mV (rest), T = −45 mV (threshold), H = −35 mV (plateau).

Synaptic transmission uses a sigmoid gating variable and a synaptic fatigue variable SF:

```
s  = 1 / (1 + exp(-k_syn·(V - V_th)))     # presynaptic gating
I_exc = G_syne · s · SF · (V_post - E_syne)
I_inh = G_syni · s · SF · (V_post - E_syni)
I_gap = G_gap  · (V_i - V_j)

dSF/dt = -a·(V - L)    if V > L and SF > 0   # fatigue during activity
          b             if V ≤ L and SF < 1    # recovery at rest
```

Default parameters: C = 7 pF, g = 1 nS, E\_syne = 0 mV, E\_syni = −100 mV.

### Synapse configurations

| Parameter     | Yuval                        | Boyle                             |
|---------------|------------------------------|-----------------------------------|
| k\_syn (exc)  | 0.125                        | 500                               |
| k\_syn (inh)  | 0.125                        | 100                               |
| V\_th         | −52 mV                       | −70 mV (resting potential)        |
| G\_syne       | 0.5 nS (default)             | 0.02 nS                           |
| G\_syni       | 0.5 nS (default)             | 0.01 nS                           |
| Fatigue SF    | yes                          | yes (retained)                    |
| Source        | Yuval thesis Table 6, hand-tuned | Mellem 2008 / Liu 2009 electrophysiology |

The Boyle sigmoid is effectively a step function that activates at rest; the Yuval sigmoid
is a gentle ramp that activates near threshold. Pass `synapse_config='boyle'` or
`synapse_config='yuval'` to supported experiment functions to select between them.

## Network Connectivity

Connectivity matrices. 

Cell class connectivity summary:

| Class | n  | Sends              | Receives           |
|-------|----|--------------------|--------------------|
| 1–3   | 24 | heavy exc          | weak exc           |
| 4     | 6  | inh only           | heavy exc          |
| 5     | 12 | heavy inh          | heavy exc + light inh |
| 6–7   | 24 | exc                | inh only           |
| 8     | 18 | —                  | exc + inh (dorsal muscle) |
| 9     | 18 | —                  | exc + inh (ventral muscle) |

## OscillatorComb Files

128 files in `data/OscillatorComb/`, each designating a different subset of
non-muscle cells as oscillators. The optimisation target is to find the combination
of oscillator configuration and conductance parameters that best reproduces:

- Antiphase activity between class 8 (dorsal) and class 9 (ventral) muscles
- Anterior-to-posterior phase progression across the six segments
- Regular inter-spike intervals (low CV)

## Running Experiments

Scientific scripts live in `experiments/` and are run directly from the
repository root:

```bash
conda activate simple-worm-scripts
python experiments/two_neuron_ii.py       # mutual inhibition
python experiments/three_neuron_eig.py    # three-neuron circuit
python experiments/sweep_two_ii.py        # inhibitory-conductance sweep
python experiments/sweep_two_ei.py        # excitation/inhibition/current sweep
python experiments/bifurcations.py        # continuation and stability analysis
python experiments/drive_snic.py           # focused AVA/AVB SNIC evidence figures
python experiments/compare_k_syn.py       # meeting-ready k_syn comparison figures
python experiments/wave_direction.py      # AVA/AVB wave-direction comparison
python experiments/ava_master_command.py  # Meng AVA-master command dynamics
python experiments/sweep_command_conductances.py  # Meng vs direct conductance sensitivity
python experiments/sweep_command_conductances.py --k-syn 0.125
python experiments/joint_conductance_orientation.py  # 5x5x5 global-conductance screen
python src/network.py                      # full 102-cell simulation
```

Figures are saved to `media/`.

For interactive equilibrium and periodic-orbit continuation of the 17-cell
segment, open the generated XPPAUT model:

```bash
xppaut xppaut/single_segment.ode
```

See `xppaut/README.md` for AVA/AVB, `k_syn`, recovery, and conductance
continuation workflows.

## Starting a New Experiment

Import `settings` to begin from the current robust baseline. Reset first when
several experiments may run in the same Python process:

```python
import settings

settings.reset_defaults()
settings.numCells = 17

# Override only the parameter being investigated.
settings.G_gap = 0.02
```

The baseline uses `G_syne=0.07`, `G_syni=0.05`, `G_gap=0.03`,
`beta=1.03`, and `tau_w=400 ms`. Connectivity matrices and applied-current
vectors are circuit-specific and must be supplied by the experiment.

Existing experiments follow the same rule: reset first, then declare only
intentional departures from the baseline. The shared 17-cell segment model can
be used without importing an analysis script:

```python
import segment
import settings

settings.reset_defaults()
segment.configure()

current = segment.drive_vector(IAVA=2.5)
initial_state = segment.reduced_rest_state()
derivative = segment.rhs_vw(initial_state, current)
```

## Running Tests

The test suite is separate from the research experiments. It checks shared
model equations and analysis utilities without launching long simulations:

```bash
conda activate simple-worm-scripts
pytest
```

## Environment

```bash
conda env create -f environment.yml
conda activate simple-worm-scripts
```

Python 3.9. Key dependencies: numpy, scipy, matplotlib.
