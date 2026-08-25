# Conductance orientation scan at k_syn = 0.125

## Question

Does repeating the full global-conductance scan at the shallower literature
sigmoid correct the AVA/AVB wave orientation?

## Design

The same 30 matched simulations used for \(k_{\rm syn}=0.25\) were repeated at
\(k_{\rm syn}=0.125\):

- \(G_{\rm syne}\), \(G_{\rm syni}\), and \(G_{\rm gap}\) varied one at a time
  over 0.5–1.5× baseline;
- both Meng-filtered and direct 2.5 pA AVB→AVA→AVB commands;
- unchanged connectivity, recovery parameters, timing, and diagnostics.

A valid correction required coherent positive AVB-before and AVB-recovery
gradients, a coherent negative AVA gradient, sustained muscle oscillation, and
dorsoventral coordination.

## Result

No parameter point corrected all three command phases with a coherent wave.

Partial sign changes did occur:

- At \(G_{\rm gap}=0.015\) nS, both AVB phases became positive, but AVA remained
  positive and phase-profile coherence was poor.
- At \(G_{\rm syni}=0.025\) nS under Meng, AVA became negative and recovery
  positive, but the initial AVB wave remained negative; minimum phase-profile
  \(R^2\) was only 0.42.
- At \(G_{\rm syne}=0.0525\) nS under direct commands, AVA became negative and
  recovery positive, but initial AVB remained negative and minimum \(R^2\) was
  approximately 0.02.
- At \(G_{\rm syne}=0.105\) nS, sustained oscillation was lost in both command
  models. The Meng run was already nearly silent at 0.0875 nS.

At the baseline conductances, the mapping remained inverted for both command
models: AVB was negative and AVA positive.

## Interpretation

The shallower sigmoid makes direction more labile but does not produce a valid
orientation correction. Because \(k_{\rm syn}=0.125\) already creates a
zero-command SNIC and spontaneous segment rhythm, low-coherence sign changes
cannot be assumed to be command-driven locomotor waves.

The global conductance magnitudes and either tested sigmoid slope are therefore
insufficient. With connectivity fixed, the next levers must change relative
cell phases, spatial initiation, or synaptic timing rather than globally
rescaling every edge.

## Outputs

- `media/command_model_conductance_comparison_ksyn_0p125.png`
- `media/command_model_conductance_sweep_ksyn_0p125.csv`
