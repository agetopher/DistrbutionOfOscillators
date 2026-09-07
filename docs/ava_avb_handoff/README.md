# AVA/AVB bifurcations and XPPAUT model

Start here for the current single-segment results. The canonical model uses
`tau_w = 400 ms`, `beta = 1.03`, `k_syn = 0.25`, and conductances
`G_syne = 0.07`, `G_syni = 0.05`, `G_gap = 0.03` nS.
Vary one command at a time: IAVA with IAVB = 0, or IAVB with IAVA = 0.

## Bifurcation diagrams

The two command onsets are consistent with SNIC bifurcations: a zero eigenvalue
at the equilibrium fold, finite-amplitude oscillations at onset, and a period
that diverges approximately as the inverse square root of distance from onset.
The critical drives are approximately **IAVA = 1.68143 pA** and
**IAVB = 1.59643 pA**.

| Artifact | What it shows |
| --- | --- |
| [Equilibrium onset](../../media/bifurcation_drive_snic_equilibrium.png) | AVA/AVB folds and eigenvalue diagnostics |
| [Periodic onset](../../media/bifurcation_drive_snic_cycles.png) | Dorsoventral amplitude, period, and SNIC scaling |
| [Numerical results](../../media/bifurcation_drive_snic_results.json) | Parameters, fitted thresholds, and cycle measurements |
| [Drive-stability diagram](../../media/bifurcation_stability_eigen.png) | Fresh 400 ms equilibrium/stability and simulated oscillation envelopes; incomplete branches |
| [β-stability diagram](../../media/bifurcation_beta_stability.png) | β swept from 0.8 to 2.0 at 400 ms; active command fixed at 2.5 pA |
| [Drive-stability data](../../media/bifurcation_drive_stability_results.json) | Saved equilibrium branches, eigenvalues, envelopes, and frequencies |
| [β-stability data](../../media/bifurcation_beta_stability_results.json) | Saved β sweep, eigenvalues, envelopes, and frequencies |
| [Run manifest](../../media/casey_bifurcations_manifest.json) | Execution times, parameters, and SHA-256 hashes of the regenerated files |
| [Analysis source](../../experiments/drive_snic.py) | Reproduces the focused onset diagrams |

The muscle observable is `mean(dorsal V) - mean(ventral V)` in mV.
These figures come from Python equilibrium searches, eigenvalue calculations,
and time integration; they are not completed AUTO branch continuations.

**Equilibrium-branch limitation:** the broad diagram uses natural-parameter
continuation seeded from simulation. Its apparent AVB-only unstable upper branch
is a seed artifact: previous multistart searches also found unstable AVA equilibria
and additional AVB equilibria. Missing curves do not establish that equilibria are
absent. The refreshed figure removes the old false “no equilibrium” annotation;
complete multibranch geometry remains an AUTO/pseudo-arclength task.

**Older figures:** files elsewhere in the project may retain historical 200 ms
results or use other observables and β values. Use the artifacts linked above
for this 400 ms handoff. The dense 0.01 pA drive sweep and up/down hysteresis
sweeps were not rerun in this focused update. The broad drive-stability diagram
uses a 0.1 pA simulation grid with separate near-onset analysis.

## XPPAUT code

- [Download/open single_segment.ode](../../xppaut/single_segment.ode)
- [XPPAUT and AUTO instructions](../../xppaut/README.md)
- [Generator](../../xppaut/generate_single_segment.py)
- [Python model](../../src/segment.py)

The `.ode` file is self-contained: 17 cells, 34 voltage/recovery states.
Its defaults match the canonical Python model. Synaptic fatigue is omitted from
this reduced system because it does not feed back into the current voltage and
recovery equations.

From the repository root:

```bash
xppaut xppaut/single_segment.ode
```

Use `iava` or `iavb` for the continuation parameter, keeping the other at zero.
`tauw` is the XPP name for the recovery timescale. The auxiliary variables
`dorsm` and `ventm` report the side-averaged muscle voltages.

## Reproduce the focused 400 ms handoff

```bash
conda env create -f environment.yml
conda activate simple-worm-scripts
python experiments/casey_bifurcations.py
pytest -q
```

For the onset figures alone, run `python experiments/drive_snic.py --tau-w 400 --beta 1.03`.

The [410 ms equilibrium figure](../../media/bifurcation_drive_snic_equilibrium_tau_w_410.png) and [410 ms periodic-onset figure](../../media/bifurcation_drive_snic_cycles_tau_w_410.png) use the same circuit with `tauw = 410`; they are
separate from the canonical baseline. To reproduce that check, use `python experiments/drive_snic.py --tau-w 410 --beta 1.03`.
