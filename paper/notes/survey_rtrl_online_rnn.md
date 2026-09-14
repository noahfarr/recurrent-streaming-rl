# Survey: RTRL and online recurrent training (2026-09-05)

## Must cite / position against
- Lemmel & Grosu, RTRRL, AAAI 2025, arXiv 2311.04830: e_R <- gamma*lambda e_R + J_hat (g_A + g_C), J_hat from RTRL or RFLO; single-layer CT-RNN; MemoryChain to 16. Prior combination of RTRL sensitivities and TD(lambda) traces, no decay analysis.
- Lemmel et al., arXiv 2602.02236 (Feb 2026): RTRRL with LrcSSM (nonlinear input-dependent diagonal SSM), exact RTRL, BC pretraining + online fine-tuning, physical car. Partial scoop of "RTRL for other diagonal cells"; not of composition theory, coherence, or streaming from scratch.
- Zucchet et al., NeurIPS 2023, arXiv 2305.15947: online LRU, exact within-layer RTRL, cross-layer temporal paths dropped. Cite for "multi-layer is open".
- Bellec et al. e-prop 2020: on a diagonal cell e-prop is exact RTRL (worth a remark). Hoyer et al. 2209.15502 e-prop + DRQN.
- Menick et al. SnAp 2021 (2006.07232): SnAp-1 is exact on a diagonal cell; truncated RTRL keeps stale influence.
- Irie et al. ICLR 2024 (2305.19044): TBPTT with large span matches RTRL except 1000-step tasks; declares multi-layer intractable. Both support and complication.
- Elelimy et al. RTU NeurIPS 2024 (2409.01449): Appendix G, stale traces help PPO (trust region); we find staleness grows with K under streaming.
- Own workshop paper arXiv 2605.24709 (RLC 2026 workshop): contains the plumbing, MemoryChain sweep, 5 POPGym, masked MuJoCo, staleness bound, and the Taylor correction evaluated on KMemoryChain. Position ICLR as the analysis paper on top; cite it in third person.

## Contradictions
- Merin "Temporal Credit Is Free" / "Immediate derivatives suffice" arXiv 2603.28750 (Mar 2026): TBPTT(1)-like immediate derivatives match RTRL on supervised online adaptation; Adam, float64. Our reply: streaming has no span; collapse is cell-dependent; MemoryChain is the long-horizon regime.
- Shakerinava et al. ICLR 2026 (2603.01959): single-layer diagonal SSM cannot track non-Abelian state; Heo et al. 2603.05573 depth reduces error exponentially; Chung et al. 2605.07755 affine recurrences cannot correct drift. Do not claim generality of the single layer.
- Francois, Orvieto, Bach 2502.09287 uncertainty principle: recall at lag K with S coefficients is smeared; explains why lambda=1 rescued L=128.
- Lam et al. 2501.08040: RTRL converges with stale sensitivities asymptotically; our bound is finite-time.

## Diagonal recurrence theory
- Orvieto LRU 2303.06349 (ring init); Orvieto 2024 universality 2307.11888; Ran-Milo 2410.14067 complex beats real provably; Yu 2410.02035 frequency bias set by Im(A) init; Mamba-3 2603.15569 back to complex rotation; Dubinin 2602.12021 block-diagonal LRU; Livi 2512.05790 learnability window = decay envelope of effective learning rate over lag (supervised cousin of Prop 1).

## Idea candidates (agent's ranking)
A. Coherence-preserving RTU init (phases near zero, radius from target horizon). High feasibility. Risk: collapses to minGRU-like real decay; include an oscillation-needing POPGym task.
B. Retention-aware trace decay: per-parameter-group lambda with gamma*lambda_rec matched to rho. High feasibility. Frame relative to RTRRL's separate lambda_A, lambda_C, lambda_R.
C. Two-layer exact RTRL with cross-layer sensitivity S <- diag(a2) S + J21 S1, O(n2 * |theta1|); compare to Zucchet truncation; Prop 1 extends to max(gamma*lambda, rho1, rho2). Medium feasibility (4 to 6 days). The reviewer-requested one.
D. Ship the Taylor staleness correction with a rho sweep (1/(1-rho) scaling). High feasibility if the workshop implementation is ported (not in current src/cells).
E. Selective (input-dependent radius) complex cell under exact RTRL; positioned against Lemmel 2602.02236. Medium.
