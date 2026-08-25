# Does the conductance result depend on the Meng command model?

## Control

The full one-at-a-time conductance scan was repeated without the Meng
AVA→AVB filter. The control used abrupt, independent 2.5 pA command blocks:

```text
0–10 s:  AVB
10–20 s: AVA
20–30 s: AVB
```

The network state was carried continuously across switches. Conductance values,
initial conditions, connectivity, simulation tolerances, phase windows, and
diagnostics were identical to the Meng scan.

## Answer

The main conclusion is the same, but the detailed transition and recovery
behavior is not.

### Results that do not depend on Meng

- At the baseline conductances, both command models produce the same
  biologically inverted mapping. The AVA gradients are +24.6 ms/segment (Meng)
  and +27.4 ms/segment (direct); AVB recovery is -18.3 and -14.9 ms/segment.
- The pre-event AVB curves are identical across the entire scan, as expected:
  tonic Meng activity was calibrated to the same direct 2.5 pA AVB command.
- Increasing \(G_{\rm syne}\) to 0.0875 nS or above eliminates sustained muscle
  oscillation in both models.
- \(G_{\rm gap}\) remains the strongest direction-control parameter.
- Neither model contains a one-at-a-time conductance setting that corrects both
  biological directions while retaining a coherent wave through recovery.

These findings belong to the downstream motor network, not to the Meng filter.

### Results that do depend on Meng

The differences concentrate near state-transition boundaries:

- At \(G_{\rm gap}=0.015\) nS, direct AVA has the biological negative sign
  (-76.0 ms/segment), whereas Meng AVA remains positive (+24.6).
- At \(G_{\rm gap}=0.015\) and 0.0225 nS, direct AVB recovery remains negative
  (-48.0 and -56.6), while Meng recovery becomes positive (+19.4 and +6.3).
  These low-gap profiles have poor linear coherence, so none is a valid
  corrected travelling wave.
- At \(G_{\rm syni}=0.0375\) nS, direct commands recover a large-amplitude
  rhythm, while the Meng protocol nearly loses recovery oscillation.
- At \(G_{\rm syne}=0.035\) nS, the direct AVA sign becomes negative but with
  very poor phase coherence; Meng retains a coherent positive gradient.

Thus Meng's slow AVA history is not creating the baseline inversion, but it does
alter basin selection and post-switch recovery near marginal conductances.

## Interpretation

Two conclusions should be kept separate:

1. **Structural:** the wrong baseline direction is a downstream network
   property and survives removal of the Meng dynamics.
2. **Dynamical:** the Meng filter matters near bifurcation/basin boundaries,
   especially for whether AVB recovers, which direction the transient selects,
   and whether a marginal wave remains coherent.

This strengthens the case for retaining the Meng model as a biologically
motivated command hypothesis while diagnosing the wave inversion downstream.
It should not be treated as an innocuous input smoother during bifurcation
analysis.

## Outputs

- `media/command_model_conductance_comparison.png`
- `media/meng_command_conductance_sweep.csv` (contains a `command_model` column)
