# XPPAUT single-segment bifurcation model

`single_segment.ode` is the 17-cell segment from `src/segment.py`, expressed as
a 34-dimensional XPPAUT system with states `[V, w]`. Synaptic fatigue is not
included because the current Python model evolves it without multiplying it
into the synaptic signal; it therefore has no effect on the voltage-recovery
subsystem or its bifurcations.

The model includes all motor neurons and muscles. Connectivity is the first
17-by-17 block of the three six-segment matrices, with the same `row=post,
column=pre` convention used by Python. The default scalar parameters are
generated from `src/settings.py`.

## Open and integrate

From the repository root:

```bash
xppaut xppaut/single_segment.ode
```

The initial condition is the Python reduced rest state (`V=-70 mV`, `w=0`).
Set `iava` or `iavb` in **Parameters**, then use **Initialconds → Go**. The
auxiliary variables `avam`, `avbm`, `dorsm`, and `ventm` provide mean command
target and muscle voltages for time-series inspection.

## One-parameter AUTO continuation

1. Leave `iava=iavb=0`, integrate until the trajectory has settled, and choose
   **Initialconds → Last**.
2. Open **File → AUTO**.
3. Under **Parameter**, select the parameter to continue. Useful choices are
   `iava`, `iavb`, `ksyn`, `beta`, `gse`, `gsi`, and `ggap`.
4. For an AVA drive diagram, choose `iava` as the main parameter and `v2` (DA)
   as the plotted variable. For AVB, use `iavb` and `v3` (DB). `v4` (DD) is a
   useful default for the zero-drive `ksyn` transition.
5. Run the steady-state branch in the positive direction. AUTO labels folds as
   `LP` and Hopf points as `HB`.
6. Grab an `HB` point and select **Periodic** to continue its limit-cycle
   branch. Use **Axes → Hi-lo** to plot the voltage envelope.

For a branch that is difficult to reach from rest, first integrate at a
parameter value where that attractor is stable, use **Initialconds → Last**,
and start AUTO from that state. Near a SNIC, reduce `ds`/`dsmax` and increase
`nmax`; the orbit period becomes large near the fold.

Only one command branch should normally be active in a one-parameter drive
continuation: hold `iavb=0` while varying `iava`, or vice versa. To continue a
codimension-two curve, grab an `LP` or `HB`, choose **Two Param**, and select a
second parameter such as `beta`, `ksyn`, or a conductance.

## Regenerate after Python-model changes

```bash
python xppaut/generate_single_segment.py
```

Commit both the generator and regenerated `.ode` file. The automated tests
check that the committed model still matches the Python source of truth.
