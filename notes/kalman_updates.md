# Kalman Updates

Status: parked 2026-08-20. Implementation lives in `streamlet/optimizers/kalman.py`
and is wired as `kalman_q`. The MinAtar sigma-floor sweep was cancelled before
producing results.

## Idea

Treat the value parameters as a Gaussian posterior and set the streaming step
size from the Kalman gain rather than from a worst-case overshoot bound. This
came out of the second-moment (EWRL) note: if we already track the curvature
`X = z(grad q - gamma grad q')` for implicit and calibrated updates, we have
most of what a scalar Kalman filter needs.

The observation model is the TD error. Per step:

- `m` is the posterior variance of the parameter error, scalar or per-parameter.
- `sigma_sq` is the observation noise, estimated as the part of `E[delta^2]`
  not explained by the curvature: `sigma_sq = max(delta_sq - m * dg_sq, floor * delta_sq)`.
- gain `alpha = m * max(interaction, 0) / ((m * dg_sq + sigma_sq) * |z|^2)`
- `m` shrinks by the information gained and grows by `process_noise`.

The diagonal variant keeps one `m` per parameter, which makes it a preconditioner
as well as a step size.

## What we learned

**Process noise compounds, and 5M steps is a long time.** v1 used
`process_noise = 1e-6` and `m` drifted to about 0.30, roughly 100x too large.
Nothing else in the filter can compensate because `m` enters both the gain and
the noise estimate. Dropping it to `1e-9` fixed the `m` dynamics. Scores barely
moved, which was the first sign that the gain was not the binding constraint.

**The metric has to be used in the step, not only in the gain.** This was the
real bug and the largest single effect we found. The diagonal variant computed
the gain using `m` but then applied the update along the raw trace `z` instead
of `m * z`. That is not a preconditioner: it uses the geometry to decide how far
to move and then moves in the unpreconditioned direction. Fixing it to apply
`m * z`, and correspondingly using `m^2` in the interaction and `|m*z|^2` in the
denominator, gave:

| game | no precondition | preconditioned |
|---|---|---|
| Asterix | 30.9 | 41.8 |
| SpaceInvaders | 66.8 | 98.0 |

Seaquest moved from roughly 2.4 to 13.2 in the earlier single-config comparison.

**The autocovariance story for `sigma_sq` was wrong.** The hypothesis was that
TD errors are autocorrelated, so `delta_sq` overstates observation noise and the
filter should subtract the lag-1 cross term instead. Measured: TD errors are only
about 12.7% autocorrelated at lag 1, and the `sigma_floor` clamp fires on just
0.1% to 1.1% of steps. So neither the autocovariance correction nor the floor was
doing meaningful work. Both were dead ends, but chasing them is what surfaced the
preconditioning bug, so the `sigma_mode = "autocovariance"` path is still in the
code and can be deleted.

## Where it stands

Pooled MinAtar means, preconditioned diagonal Kalman against the other streaming
optimizers:

| game | kalman_q | implicit_q | stream_q | calibrated_q | intentional_q |
|---|---|---|---|---|---|
| Asterix | **41.8** | 22.6 | | 3.3 | |
| Breakout | **12.7** | 10.1 | 11.0 | 6.5 | 7.8 |
| Seaquest | 18.8 | **55.5** | | 0.4 | |
| SpaceInvaders | 98.0 | **139.2** | | 30.3 | |

Two clear wins, two clear losses, and the losses are large. Read these as
indicative only: they are means pooled over every config that ever ran under
each name, with very different run counts (implicit_q on Breakout pools 196 runs
including tuning sweeps), not matched head-to-head comparisons at equal budget.
A real comparison needs per-variant tuning of the same kind job 133348 is doing
for implicit.

## If we come back to it

The open question is why it splits so cleanly: strong on Asterix and Breakout,
weak on Seaquest and SpaceInvaders. Seaquest and SpaceInvaders are the two games
with longer horizons and sparser scoring, which would be the first thing to test.

The cancelled sweep was `sigma_floor` in `{0.01, 0.001}` against the default
`0.1`, on the theory that the floor is too aggressive. Given the clamp only fires
on about 1% of steps, that sweep was unlikely to move anything and is not worth
rerunning as specified.

The transferable lesson is the second one above, and it is not Kalman-specific:
if a method derives a metric, the metric must appear in the direction as well as
the magnitude. That mistake is easy to make and cost us most of a day.
