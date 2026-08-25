# Conductance sensitivity under the AVA-master command

## Design

The full Meng AVB→AVA→AVB protocol was repeated while varying one global
conductance at a time. Connectivity, command calibration, \(k_{\rm syn}=0.25\),
\(\beta=1.03\), and all other parameters were held fixed.

| Parameter | Baseline | Values tested |
|---|---:|---|
| \(G_{\rm syne}\) | 0.070 nS | 0.035, 0.0525, 0.070, 0.0875, 0.105 |
| \(G_{\rm syni}\) | 0.050 nS | 0.025, 0.0375, 0.050, 0.0625, 0.075 |
| \(G_{\rm gap}\) | 0.030 nS | 0.015, 0.0225, 0.030, 0.0375, 0.045 |

For the biological mapping, a valid result requires:

- AVB before the event: positive gradient (anterior→posterior);
- AVA during the event: negative gradient (posterior→anterior);
- AVB recovery: positive gradient;
- a coherent phase profile, continuing muscle oscillation, and dorsoventral
  coordination.

The gradient here is measured in milliseconds per **segment**. This differs
from `ava_master_command.py`, which reports milliseconds per muscle row.

## Result

No one-at-a-time conductance change corrected the biological mapping while
preserving a coherent wave through all three command phases.

### Excitatory chemical conductance

Reducing \(G_{\rm syne}\) strengthened the magnitude and linear coherence of the
existing, biologically inverted phase gradients; it did not reverse their
mapping. Increasing it to 0.0875 nS (1.25× baseline) or 0.105 nS (1.5×)
eliminated the sustained muscle rhythm. Thus excitation has a sharp upper
operating boundary in this protocol.

### Inhibitory chemical conductance

Changing \(G_{\rm syni}\) had relatively little effect on the pre-event AVB
direction, which remained posterior→anterior throughout the range. It strongly
affected recovery: 0.0375 nS nearly eliminated recovery oscillation, while
0.075 nS moved the recovery gradient through zero but with essentially no
linear wave coherence. This looks like basin/state selection, not a corrected
travelling wave.

### Gap-junction conductance

\(G_{\rm gap}\) was the strongest direction-control parameter. At 0.015 nS,
both AVB phases changed sign but the AVA phase did not; at 0.0225 nS, the AVA
phase changed to the biological sign and recovery changed sign, but the
pre-event AVB phase remained wrong. These low-gap sign changes had very low
phase-profile \(R^2\), showing that the body-wide wave had become non-monotone
or incoherent. Stronger gap coupling restored the original inverted mapping.

## Interpretation

The global conductance magnitudes cannot repair a phase offset imposed by
specific circuit edges. The scan strengthens the structural diagnosis:
heterotypic intersegment gap-junction organization, rather than the scalar
\(G_{\rm gap}\) alone, is the appropriate next control variable.

The measured dorsoventral peak-phase offset was also below the ideal 0.5 across
the scan. Because the muscle waveforms are asymmetric, this metric should be
treated as a diagnostic flag and checked against a waveform-based antiphase
measure before retuning to it.

## Reproduce

```bash
conda run -n simple-worm-scripts python experiments/sweep_command_conductances.py
```

Outputs:

- `media/meng_command_conductance_sweep.png`
- `media/meng_command_conductance_sweep.csv`

## Next experiment

Keep the successful 2.5 pA Meng command calibration and baseline chemical
conductances. Split the intersegment gap matrix into named heterotypic edge
families (especially AS↔VA and DB↔VB), then scale those families independently.
Validate any candidate correction across multiple initial conditions.
