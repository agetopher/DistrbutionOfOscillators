# Distribution of Oscillators

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
`docs/neuron_model_comparison.tex`.

## Repository Structure

```
DistrbutionOfOscillators/
├── src/
│   ├── settings.py       # Global parameter store (shared across all scripts)
│   ├── functions.py      # Yuval model equations: f(v), f_vec(v), sf_vec(v, s)
│   ├── network.py        # Full 102-cell ODE + run() — loads connectivity from data/
│   └── analysis.py       # Post-processing: oscillation_metric(), phase_difference()
│
├── tests/
│   ├── test_ramp.py      # Single neuron under a triangular applied-current ramp
│   ├── test_two_inh.py   # Two mutually inhibitory neurons — confirmed oscillations
│   ├── test_three.py     # Three-neuron exc/inh/gap circuit (work in progress)
│   ├── sweep_two.py      # Parameter sweep: G_syne x G_syni x Iapp (2-cell exc-inh)
│   ├── sweep_three.py    # Parameter sweep: G_syne x G_syni x G_gap (3-cell circuit)
│   └── plot_connectivity.py  # Visualise connectivity matrices by cell class
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
│   └── neuron_model_comparison.tex   # LaTeX comparison of alternative neuron models
│
├── media/                # Output figures (git-ignored)
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

## Network Connectivity

Connectivity matrices. 

Cell class connectivity summary (correct orientation):

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

## Running Scripts

All scripts in `tests/` are run directly:

```bash
conda activate simple-worm-scripts
python tests/test_two_II.py        # confirmed oscillations
python tests/sweep_two_EI.py           # exc-inh parameter sweep
python src/network.py             # full 102-cell simulation
```

Figures are saved to `media/`.

## Environment

```bash
conda env create -f environment.yml
conda activate simple-worm-scripts
```

Python 3.9. Key dependencies: numpy, scipy, matplotlib.
