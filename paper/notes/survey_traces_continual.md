# Survey: eligibility traces with recurrence, continual RL with recurrence (2026-09-05)

## Traces
- Kozuno et al. ICML 2021 (2103.00107) Peng's Q(lambda): preserved traces during exploration inject bias (Thm 2). Our Watkins result differs: RTRL keeps credit in the state, not the trace, so no off-policy bias; say so.
- Daley et al. ICML 2023 (2301.11321) trajectory-aware traces: cut traces cannot be restored.
- Daley 2607.28916 (Jul 2026) Gated Q-learning: lambda_t = lambda*chi on non-greedy steps, tabular. Precedent for a soft cut; streaming deep version with RTRL is open.
- Kapturowski 2209.07550 soft Watkins; Mahajan & Seymour 2604.13780 Soft Q(lambda) theory (relevant to soft_q recipe).
- van Hasselt 2007.01839 expected traces.
- Gogianu, Lutu, Pascanu 2605.06764 (May 2026) Adaptive Q(lambda): Adam-style second moment decaying at the trace rate, beats StreamQ on Atari. New streaming baseline reviewers may ask about.
- Kobayashi 2008.10040 drift-aware trace decay; Gupta 2312.12972 stale accumulated traces at large lambda under nonlinear FA (our lambda=1 result is a counterexample in the RTRL regime).
- Zhao 1904.11439 meta-learned per-state lambda (only prior per-state lambda; nothing ties lambda to model memory).
- Bellec e-prop 2020: per-synapse trace decays at the neuron's own retention, reward-based version low-passes with gamma. Closest prior composition of a retention kernel with a gamma kernel; no max(gamma*lambda, rho) statement.
- Lemmel & Grosu RTRRL 2311.04830: prior combination, no decay result.
- Merin 2603.28750: traces should decay at the measured self-propagation (~0.01 for trained vanilla RNNs), recommends lambda=0 with Adam. Same quantity as rho; contradicts lambda=1 headline naively. Reply: learned rho approaches 1 on MemoryChain; the trace also carries lambda-return target credit, distinct from memory credit.
- Ni et al. 2307.03864: memory length vs credit-assignment length are distinct; MemoryChain couples them. One decoupled task would tighten the claim.
- Harutyunyan 1602.04951: lambda bound from off-policyness (only prior lambda chosen from a system property).
- Eberhard et al. 2503.15200 memory traces as agent state (Forager's reward-trace baseline).
- Livi 2512.05790 learnability window.

## Continual
- Dohare Nature 2024, Abbas CoLLAs 2023, Lyle 2023/2024 (plasticity loss); Elsayed & Mahmood UPGD ICLR 2024; Tang C-CHAIN ICML 2025.
- Forager 2605.01131: freeze test done; RTU-PPO best; value agents' exploration-floor collapse not in their paper (expect hyperparameter scrutiny; Mesbahi et al. 2404.02113 lifetime-tuning critique).
- Anand & Precup 2312.11669 permanent/transient values; Sun et al. ICLR 2026 2603.00903 and Chua et al. 2605.26357 fast/slow learners.
- Liu & Mou 2602.09234: plasticity loss tied to abrupt transitions (Foragax flips are abrupt).
- Wang et al. 2605.09044 optimisation readiness metric.
- Run & Ding 2607.11906 ICRL under non-stationarity survey; Shchendrigin et al. 2601.15086 memory overwriting benchmark.
- Lee et al. bioRxiv 2025.11.30.691382: mice shift from synaptic plasticity to recurrent-dynamics value updates under frequent reversals; RNN with online weight updates reproduces it. Strongest external support for the shrinking post-flip dip.
- Sutton, Koop, Silver 2007 tracking; Wang 2016, Botvinick 2019 meta-RL in activations.

## Idea candidates (agent's ranking)
1. Retention-matched trace decay: per-parameter-group lambda, heatmap over (lambda, rho) with the gamma*lambda = rho boundary drawn; credit-weight-vs-lag collapse onto max(gamma*lambda, rho)^k. 3 to 4 days. Risk: ObGD step-size bound depends on trace norm.
2. Watkins-cut robustness via the sensitivity path with a chi-gate sweep (return vs epsilon for TBPTT(1) vs RTRL). 3 days.
3. Foragax in-state vs in-weight: freeze weights vs reset state at each flip, flip-period sweep, crossover where adaptation moves into state. 4 to 5 days.
4. Retention audit: learned rho spectrum over training vs best lambda per task (preempts the stale-trace objection). 1 to 2 days.
5. (stretch) Expected traces conditioned on recurrent state.
