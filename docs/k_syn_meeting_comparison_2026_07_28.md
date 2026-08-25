# Comparing \(k_{\mathrm{syn}}=0.25\) and \(0.125\)

## Controlled comparison

These figures use the same 17-cell segment, membrane parameters, recovery
variable, connectivity, conductances, initial voltage, and command currents.
Only \(k_{\mathrm{syn}}\) changes.  Synaptic fatigue is excluded because it is
inert in the current network model.

## 1. The synaptic activation curves

![Synaptic activation comparison](../media/k_syn_activation_comparison.png)

At the resting voltage \(L=-70\) mV, the gate is approximately 0.011 for
\(k_{\mathrm{syn}}=0.25\) and 0.095 for \(k_{\mathrm{syn}}=0.125\).  With the
present conductances, making the sigmoid shallower therefore increases the
basal current contributed by every chemical connection by about 8.7-fold.

## 2. The undriven segment

![Zero-drive segment comparison](../media/k_syn_zero_drive_comparison.png)

The \(k_{\mathrm{syn}}=0.25\) segment converges to a quiescent equilibrium.  At
\(k_{\mathrm{syn}}=0.125\), DD instead develops a spontaneous oscillation with
approximately 21.7 mV peak-to-peak amplitude.  Smaller oscillations appear in
VD and dorsal muscle.  This transition has been located at
\(k_{\mathrm{syn}}\approx0.13767\) and classified as a saddle-node on invariant
circle (SNIC).

## 3. Commanded muscle output

![Command-response comparison](../media/k_syn_command_comparison.png)

With either AVA or AVB held at 3 pA, both parameter choices produce strong
dorsoventral alternation in the isolated segment.  The shallower sigmoid gives
a faster rhythm:

| command | \(f\), \(k=0.25\) | \(f\), \(k=0.125\) |
|---|---:|---:|
| AVA, 3 pA | 1.40 Hz | 1.59 Hz |
| AVB, 3 pA | 1.33 Hz | 1.64 Hz |

The mean dorsal-muscle excursion is also modestly smaller at
\(k_{\mathrm{syn}}=0.125\): 36.4 versus 39.8 mV for AVA and 38.2 versus
40.7 mV for AVB.

## Interpretation

The \(0.125\) value remains viable as a synaptic slope, but not as an isolated
parameter replacement under the current hand-chosen conductances.  It preserves
commanded alternation while changing the circuit from command-gated to
spontaneously active.  The next comparison should retune chemical conductance
or half-activation voltage and ask whether \(0.125\) can recover a stable
zero-command state without losing the commanded rhythms.

Reproduce all three figures with:

```bash
conda run -n simple-worm-scripts python experiments/compare_k_syn.py
```
