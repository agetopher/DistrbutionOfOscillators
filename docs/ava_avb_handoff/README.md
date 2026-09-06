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
| [Historical broad drive diagram](../../media/bifurcation_stability_eigen.png) | Equilibria and simulated oscillation envelopes; see limitation below |
| [Analysis source](../../experiments/drive_snic.py) | Reproduces the focused onset diagrams |

The muscle observable is `mean(dorsal V) - mean(ventral V)` in mV.
These figures come from Python equilibrium searches, eigenvalue calculations,
and time integration; they are not completed AUTO branch continuations.

**Equilibrium-branch limitation:** the historical broad figure still contains the obsolete annotation “no equilibrium beyond the fold”; that annotation is incorrect and is superseded by the diagnosis below.

 the broad diagram uses natural-parameter
continuation seeded from simulation. Its apparent AVB-only unstable upper branch
is a seed artifact: multistart searches also find unstable AVA equilibria and
additional AVB equilibria. Missing curves do not establish that equilibria are
absent. Complete multibranch geometry remains an AUTO/pseudo-arclength task.
Historical upper-Hopf results at other recovery timescales should not be carried
over to the canonical 400 ms regime.

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

## Reproduce the canonical onset figures

```bash
conda env create -f environment.yml
conda activate simple-worm-scripts
python experiments/drive_snic.py --tau-w 400 --beta 1.03
pytest -q
```

The [410 ms equilibrium figure](../../media/bifurcation_drive_snic_equilibrium_tau_w_410.png) and [410 ms periodic-onset figure](../../media/bifurcation_drive_snic_cycles_tau_w_410.png) use the same circuit with `tauw = 410`; they are
separate from the canonical baseline. To reproduce that check, replace
`--tau-w 400` with `--tau-w 410`.
