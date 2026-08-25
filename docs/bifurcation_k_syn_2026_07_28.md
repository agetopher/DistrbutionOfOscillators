# Shallower synaptic gating creates a zero-command SNIC

## Question

Does the full 17-cell segment retain the previously identified
command-current bifurcation structure when the synaptic sigmoid steepness is
changed from the working value \(k_{\mathrm{syn}}=0.25\) to the literature value
\(k_{\mathrm{syn}}=0.125\)?

All other parameters were held at the current shared baseline:

\[
G_{\mathrm{syne}}=0.07,\quad
G_{\mathrm{syni}}=0.05,\quad
G_{\mathrm{gap}}=0.03,\quad
\beta=1.03,\quad
\tau_w=200\ \mathrm{ms},\quad
V_{\mathrm{th}}=-52\ \mathrm{mV}.
\]

The reduced \((V,w)\) system was used because synaptic fatigue currently does
not feed back into the voltage equations.

## Result

At zero AVA/AVB command drive:

- \(k_{\mathrm{syn}}=0.25\): the segment converges to a quiescent equilibrium.
- \(k_{\mathrm{syn}}=0.125\): DD develops a robust spontaneous rhythm across
  five randomized initial conditions, with approximately \(22\) mV amplitude
  and \(1.6\) Hz frequency.

Continuation of the zero-drive equilibrium in \(k_{\mathrm{syn}}\) locates a
saddle-node at

\[
\boxed{k_c \approx 0.13767}.
\]

The critical eigenvalue is real and approaches zero at the fold. Independent
continuation of the DD periodic orbit from \(k_{\mathrm{syn}}=0.125\) upward
places its termination at approximately \(0.13765\), within the
\(10^{-4}\)-resolution of the equilibrium fold. Its amplitude remains finite
(\(\approx23.6\) mV) while its period increases to approximately \(4.16\) s at
the last resolved point.

These observations identify the transition as a **saddle-node on invariant
circle (SNIC) in \(k_{\mathrm{syn}}\)**. No bistable interval was resolved at
the current continuation resolution.

## Why the qualitative behavior changes

The synaptic gate is

\[
s(V)=\frac{1}{1+\exp[-k_{\mathrm{syn}}(V-V_{\mathrm{th}})]}.
\]

At the resting voltage \(V=-70\) mV:

\[
s(-70)\approx0.011\quad(k_{\mathrm{syn}}=0.25),
\qquad
s(-70)\approx0.095\quad(k_{\mathrm{syn}}=0.125).
\]

Thus the shallower sigmoid increases resting synaptic activation almost
ninefold. With the current conductances and threshold, \(0.125\) is not simply
a smoother version of the same model; it moves the undriven segment into a
different dynamical regime.

## Interpretation

The literature value may be biologically preferable as a sigmoid shape, but it
is not portable independently of \(V_{\mathrm{th}}\), the conductances, and the
recovery parameters. At the present baseline it removes command gating:
oscillation exists before AVA or AVB is applied.

If command-gated oscillation remains a modeling requirement,
\(k_{\mathrm{syn}}=0.125\) requires retuning. The next useful continuation is
the quiescence boundary in

\[
(k_{\mathrm{syn}},G_{\mathrm{syne}},G_{\mathrm{syni}})
\]

at zero command drive, followed by the AVA/AVB bifurcation curves inside the
quiescent region. A smaller first map—\((k_{\mathrm{syn}},G_{\mathrm{syne}})\)
with the inhibitory-to-excitatory conductance ratio fixed—would determine
whether the literature steepness can be retained without spontaneous DD
oscillation.

## Reproduction

Run:

```bash
conda activate simple-worm-scripts
python experiments/bifurcations.py
```

with `MODE = 'k_syn'`. The generated figure is
`media/bifurcation_k_syn_zero_drive.png`.
