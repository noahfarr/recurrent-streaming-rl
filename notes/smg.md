# Streaming Memory Gradient (SMG)

Status: parked 2026-08-20. Implementation lives in `streamlet/algorithms/smg_lambda.py`
and is correct and tested. Parked because the contribution is not novel enough to
carry a paper, not because it is broken.

## What it is

In a recurrent agent whose own past actions feed back into its inputs,
`h_t = g_psi(h_{t-1}, o_t, a_{t-1})` and `a_t = f_theta(h_t, eps_t)`, the hidden
state genuinely depends on the actor parameters. The pathwise gradient therefore
contains a term everyone drops:

```
(dQ/dh + dQ/da * df/dh) * dh_t/dtheta
```

SMG computes it online with a trace `M_t = D_t M_{t-1} + (dh_t/da_{t-1})(da_{t-1}/dtheta)`,
which is the RTRL analogue for actor parameters rather than cell parameters.

## Why we parked it

**It is not novel as mathematics.** It is exactly what you get by not truncating
the pathwise gradient. A reviewer can correctly say "this is the chain rule".

**It is genuinely absent from the literature, but for a structural reason rather
than an oversight.** Two independent confirmations:

- RDPG (Heess et al. 2015) defines the history as `h_t = (o_1, a_1, ..., a_{t-1}, o_t)`,
  explicitly containing past actions, yet its published gradient is only
  `dQ/da * dmu/dtheta`. No `dh/dtheta` term.
- Ni et al. 2022 (recurrent model-free RL) has `Actor_RNN.forward(prev_actions, rewards, observs)`
  taking previous actions as a replay-buffer tensor. Buffer data carries no
  gradient, so the path does not exist even in principle.

The reason is that off-policy replay makes `da_{t-1}/dtheta` undefined for the
current parameters: those actions came from an old policy. Streaming is the
setting where the term becomes both well-defined and computable. That is a real
observation, but it is an observation, not an algorithm.

**Only pathwise estimators are missing anything.** For score-function methods the
likelihood-ratio derivation is already complete, because `h_t` is a deterministic
function of realised data. So RL-squared style agents that feed previous actions
and train with BPTT are not buggy. This also makes the LLM case vacuous: token-level
RLHF and GRPO use score-function estimators and miss nothing.

**It is architecturally conditional and empirically narrow.** The term is
identically zero if the feature extractor does not consume the previous action.
Even when it is nonzero, it only carries signal if the agent's own action is part
of the hidden state that matters. On position-masked MuJoCo, velocity is
recoverable by differencing positions, so the action channel is largely redundant
and SMG is predicted not to help.

## Implementation learnings worth keeping

**The complex-diagonal cells are block-diagonal, not diagonal.** This is the
subtlest thing we found. RTU and GTU look elementwise but each unit is a 2x2
scaled rotation on `(real, imaginary)`, so the state Jacobian in the flattened
carry has off-diagonal mass of 0.75 and 0.81 respectively. MinGRU is genuinely
diagonal (0.00).

Consequences:

- A scalar `memory_decay = 0.9` captured the block magnitude `r` but dropped the
  phase entirely. The true diagonal entries `r*cos(theta)` span `[-0.758, 0.747]`,
  so the scalar had the wrong sign for many units, not merely the wrong size. For
  a complex-diagonal cell the phase is the whole mechanism of memory.
- Extracting the Jacobian with a single ones-tangent JVP is wrong for exactly the
  cell SMG uses: it returns block row-sums, not the diagonal.
- One JVP per carry leaf recovers the blocks exactly (2 for RTU/GTU, 1 for MinGRU),
  matching the dense Jacobian to 5.5e-8 at 2 JVPs instead of one per carry dimension.

**RTRL was stop-gradienting the carry.** `local_jacobian` stop-gradiented the
carry as well as the parameters, which zeroed `dQ/dh` and made SMG silently
impossible. Dropping it is safe because nothing in the codebase unrolls; verified
by 29 tests plus bit-identical parameters for `stream_ac` and `qrc` after 500 steps.

**Flax has `nn.custom_vjp` but no `nn.custom_jvp`.** The influence injection was
reverse-mode only, which made `jacfwd` illegal anywhere downstream. We added a
lifted `custom_jvp` in `src/cells/injection.py` mirroring the flax API. JAX
transposes the JVP to recover reverse mode, so one rule serves both directions,
and gradients stayed bit-identical across all four cells and all three injection
variants.

**Forward mode did not pay off in wall-clock.** For the action sensitivity,
`jacfwd` costs one JVP per action dimension against `jacrev`'s one VJP per carry
dimension, which is 7x fewer FLOPs (4.4e7 vs 3.1e8). Measured wall-clock on CPU
is a wash, within 10% once you interleave the two and take minimums. Naive timing
on a loaded box showed 2.5x and 3x gaps in *both* directions, which was pure
contention noise.

**Failure modes that produce a silent zero.** `actor/memory_gradient_norm` went
to exactly zero twice: once because the recipe's feature extractor ignored the
action argument, once because of the RTRL carry stop-gradient. Always check that
metric first. With the exact trace it sits around 0.26 to 0.34 against an actor
gradient norm of 0.49 to 0.57, so the term is roughly 60% of the actor term and
not a rounding error.

## What would revive it

A task where the agent's own action is provably part of the hidden state, so the
term has signal rather than noise. In order of fit:

- **Action delay or actuator latency.** If the applied torque is the command from
  k steps ago, action history *is* unobserved state and cannot be recovered from
  observations. Best fit, because delayed action is a real property of real-time
  control, which is the premise of the streaming setting.
- **Observation dropout**, where dead reckoning from own commands is the only way
  to bridge the gap.
- **Egocentric navigation** requiring path integration.

The cheap falsifiable prediction: ablate the action input to the feature extractor
per task. If removing it costs nothing, SMG cannot help there. SMG's benefit should
track the action input's benefit across tasks, and that correlation would be a
better result than a single win.

The one setting where the online trace, rather than the observation, would be
load-bearing is continuous latent reasoning at context lengths too long to
backpropagate through. There the action is continuous, pathwise is natural, and
the memory path is 100% of the state's parameter dependence. At shorter lengths
plain BPTT gives the term for free provided nobody detaches, which is worth
checking in existing continuous-thought RL code.
