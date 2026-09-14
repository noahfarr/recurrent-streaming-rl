# Relearning versus memory under slow non-stationarity (Foragax)

Observation (2026-09-05, Foragax NeverEndingRelearning, 9x9 one-hot, eps floor 0.1, 5M frames, 10 seeds):
stream Q with a memoryless FFN relearns after every reward flip and plateaus at a reward rate of 1.6;
RTU ends at 1.7 and its post-flip dip shrinks monotonically over the nine flips (0.70 to 1.64) while the FFN's stays near 1.4;
minGRU is unstable; intentional Q is weak with every cell. TD error spikes and Q-values drop at each flip for the FFN, then recover within one epoch.

## Closest prior result

- Tang et al. 2026 (Forager, arXiv 2605.01131), sections 7 to 9. Same task. DQN relearns at every switch with minor plasticity loss, PPO loses plasticity progressively.
  A reward trace as input helps both. RTU-PPO is the best learner, eliminates PPO's plasticity loss, and when frozen still drops, so it is "continually relearning its policy at every switch, in addition to achieving better performance due to the recurrent network mitigating partial observability" and its frozen policy beats the others, i.e. memory "makes the policy less dependent on tracking".
  They also state the mechanism we see: "memory would help the agent anticipate the switches rather than relying on sampling negative rewarding mushrooms before adapting."

## Tracking in the weights

- Sutton, Koop, Silver 2007, On the role of tracking in stationary environments (ICML). Tracking (never converging, keep adapting the weights to the current part of the stream) can beat converging even in stationary problems; the value of temporal locality. Our FFN is a tracker.
- Abel et al. 2023, A definition of continual RL (NeurIPS): continual RL is the setting where the best agents never stop learning. Khetarpal et al. 2022 (JAIR) taxonomy of non-stationarity by scope and driver; a periodic reward flip is a slow, exogenous driver.
- Hidden-mode MDPs (Choi, Yeung, Zhang 2000) and context detection (da Silva et al. 2006, RL-CD) formalise the alternative: detect the regime and switch models. Relearning is the degenerate version with one model and a fast step size.

## Adaptation in activations instead of weights

- Wang et al. 2016, Learning to reinforcement learn; Duan et al. 2016 RL^2: recurrent agents trained across tasks adapt to a new task or a switch within the hidden state, with weights fixed. Inputs include last action and reward, as in Forager and here.
- Botvinick et al. 2019, Reinforcement learning, fast and slow (TICS): the slow path is incremental weight adjustment, the fast path is adaptation in activations learned by the slow path. The shrinking RTU dips are the fast path being acquired by the slow path.
- Ortega et al. 2019, Meta-learning of sequential strategies: memory-based meta-learning amortises Bayesian filtering in the recurrent state, which is what a phase tracker for the square wave is.

## Why the batched learner should lose

- Dohare et al. 2024 (Nature) and Abbas et al. 2023 (CoLLAs): loss of plasticity under repeated task change; PPO policy collapse over long training. Forager confirms for PPO on this exact task, and L2-init only partly fixes it.
- Elsayed et al. 2024 (stream-x) motivates streaming as the setting for continual and on-device learning; a replay buffer averages over both phases of a flip. Prediction for our PPO-FFN run: deeper and longer dips than stream Q FFN, degrading over flips.

## Framing for the paper

Two routes absorb a slowly changing hidden variable: relearn the weights or carry the phase in the recurrent state. With a streaming update the first route is fast enough that a memoryless agent nearly saturates the task; the RTRL-trained recurrent state then progressively takes over, which shows as the post-flip dip shrinking across flips. Memory becomes necessary only when the hidden variable changes faster than the weights can follow (period sweep would show the crossover).
