# AVA-master command experiment

## Question

Can the AVA→AVB dynamical model proposed by Meng et al. (2024) replace the
independent `IAVA`/`IAVB` protocol while leaving the 102-cell motor network,
connectivity, and conductances unchanged?

## Command model

The wild-type fit reported by Meng et al. is

```text
x_AVB = a x_AVA + b y
dy/dt = -y/tau + x_AVA

a = -0.51, b = 0.02, tau = 3.5 s
```

Thus AVA rapidly inhibits AVB through the negative direct term, while its
low-pass-filtered history slowly excites AVB. The published variables are
calcium activities, not currents. The experiment therefore keeps these
coefficients in activity units and adds a separate, explicit calibration:

- tonic AVA activity maps to `IAVA=0`, `IAVB=2.5 pA`;
- phasic AVA activity maps to `IAVA=2.5 pA`;
- the AVB scale is chosen so the instantaneous phasic AVA jump suppresses
  `IAVB` to zero;
- after the event, the residual slow state transiently raises `IAVB` above its
  2.5 pA baseline.

The 2.5 pA calibration is a model-operating point, not a biological current
estimate. A 3 pA trial trapped the network in a non-oscillatory state after the
switch; 2.5 pA gave reproducible transitions.

## Result

The hierarchical command layer works as intended and produces reversible
AVB→AVA→AVB state switching. The measured dorsal-muscle phase gradients were:

| State | Motor command | Phase gradient | Model wave |
|---|---|---:|---|
| tonic / forward | `IAVB=2.5 pA` | -7.0 ms/row | posterior→anterior |
| phasic AVA / backward | `IAVA=2.5 pA`, AVB strongly suppressed | +6.8 ms/row | anterior→posterior |
| recovery / forward | AVB restored and slowly relaxes | -6.7 ms/row | posterior→anterior |

This is a successful **command-state transition**, but not a correction of the
locomotor-wave mapping. The unchanged downstream network still assigns the
biologically wrong direction to each state: AVB/forward should be
anterior→posterior, and AVA/backward should be posterior→anterior.

## Reproduce

```bash
conda run -n simple-worm-scripts python experiments/ava_master_command.py
```

Output: `media/network_ava_master_command.png`.

## Next test

Keep this command layer fixed and perturb only the downstream intersegment
coupling, beginning with the heterotypic gap-junction edges already identified
as the dominant source of the inverted phase gradient.
