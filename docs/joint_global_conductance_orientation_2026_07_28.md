# Joint global-conductance orientation screen

## Question

Can coordinated changes in \(G_{\rm syne}\), \(G_{\rm syni}\), and
\(G_{\rm gap}\) correct the AVA/AVB wave mapping even though changing each
conductance alone could not?

## Design

The screen tested the complete Cartesian product

\[
(G_{\rm syne},G_{\rm syni},G_{\rm gap})
\in
\{0.5,0.75,1,1.25,1.5\}
\times
\{0.5,0.75,1,1.25,1.5\}
\times
\{0.5,0.75,1,1.25,1.5\}
\]

relative to the baseline \((0.07,0.05,0.03)\) nS: 125 parameter triples and
250 full-network simulations. Connectivity and within-type relative weights
were unchanged. The screen used \(k_{\rm syn}=0.25\) and separate steady 2.5 pA
AVA and AVB commands to isolate downstream network orientation.

A strict candidate required:

- AVB gradient \(>+3\) ms/segment (anterior→posterior);
- AVA gradient \(<-3\) ms/segment (posterior→anterior);
- phase-profile \(R^2>0.8\) for both commands;
- mean muscle excursion \(>10\) mV for both commands.

## Result

There were **0 strict candidates out of 125**.

More strongly, there were **0 parameter triples with even the two raw
orientation signs simultaneously correct**, before applying coherence or
amplitude thresholds. Regions that made AVB positive also left AVA positive.
Regions that made AVA negative left AVB negative. Global conductance balance
can move the preferred travelling-wave direction, but it moves both command
branches together rather than separating their biological orientations.

The closest simultaneous-orientation margin was still -20 ms/segment. At
\((G_{\rm syne},G_{\rm syni},G_{\rm gap})=(0.07,0.025,0.0225)\) nS, AVA remained
positive (+20) and AVB negative (-12). At \((0.035,0.025,0.0375)\) nS, AVA was
negative (-9.1) but AVB was also negative (-20).

## Interpretation

The negative one-at-a-time result was not an axis-sampling artifact. Across the
tested 3-D volume, the three global conductances do not provide independent
control of the two command-dependent wave directions. This strongly suggests
that orientation depends on relative phases within particular cell-class
pathways, not on the global excitation/inhibition/electrical balance.

With the connectivity graph fixed, the remaining minimally invasive choices
are:

1. class- or pathway-specific conductance multipliers, if relative weights are
   not part of the frozen configuration;
2. class-specific recovery dynamics, especially for the heterotypically
   coupled AS/VA and DB/VB classes;
3. spatial command or excitability gradients;
4. explicit synaptic kinetics or delays.

## Outputs

- `media/joint_conductance_orientation_screen.png`
- `media/joint_conductance_orientation_screen.csv`
