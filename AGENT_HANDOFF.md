# Expert design history and diagnostics (reference)

`CLAUDE.md` is authoritative for status, constraints and next steps. This file holds the detailed history of
the scripted throw expert (Phase 1, done at 87%). Read it only when working on the expert or the env mechanics.

## Expert and env decisions (and why)

1. **Weld ruling: no weld. The grasp is purely physical (friction).**
   `physics_step` only writes ctrl and steps; the `grasp_right_*` welds stay disabled. "Grasping" is
   `ThrowEnv.holding(ball)`: both finger bodies (`openarm_right_ee_inner_finger` / `_outer_finger`) touch the ball.
   Measured on 40 seeds: the ball was still held at the release tick in 36/37 grasped episodes with no tuning
   (40 g ball, fingertip condim 4 / friction 1.0, finger kp=10 squeezes well above throw loads). Success was 26/40
   vs 25/40 with the old weld assist, so the weld bought nothing. Demos and VLA eval use identical mechanics, and
   `dropped_during_grasp` is reachable again.
2. **Gripper polarity:** finger joint `0` = closed (tips ~9 mm apart), `-0.7854` = fully open (~15 cm).
   The original scaffold had this inverted.
3. **Grasp point** = `openarm_right_ee_base_link` origin + (0, 0, -0.145) in the EE frame. Measured from the
   collision meshes: the palm collider reaches z = -0.111, the fingertips are inward hooks ending at ~-0.155, and the
   open finger gap is only wider than the ball below ~-0.115. The approach must be **fully open** (a partial
   opening leaves the hooks narrower than the 6 cm ball).
4. **Only the grasp welds are disabled at reset.** Disabling all equalities also killed the finger mimic joint,
   so the outer finger never opened.
5. **Ball placement is rejection-sampled** until no ball touches anything but the table (the jitter could overlap
   balls, and the `home` wrist sits next to the ball set).
6. **Expert plan** (`throw_expert.py`), all segments quintic and sampled exactly at 50 Hz:
   1. home, with the elbow (joint 4) raised to 2.4 first (a direct path sweeps the balls)
   2. `throw_ready`, gripper closed during transit
   3. hover (open)
   4. descend
   5. close for 10 ticks
   6. lift
   7. wind-up
   8. throw, ending at `q_rel` with joint velocity `qd = J^+ v`; gripper commanded open `release_lead` (1) tick
      before `q_rel` (the fingers are slow; 2 ticks was worse)
   9. coast at `qd` for `coast_ticks` (6) so the opening fingers keep pace with the ball, then decelerate
   10. hold

   Sweep (40 seeds): lead 0 -> 29, lead 1 -> 32, lead 2 -> 26 (coast 0). On 80 seeds: lead 1/coast 0 -> 61,
   lead 1/coast 6 -> 64. `aim_iters` 14 instead of 8: +1, not worth the planning time.
7. **Wrist yaw per episode.** Grasp yaw maximizes finger clearance from neighbor balls; throw yaw points the
   gripper's open side (local ±x) along the throw heading so the ball does not leave through a finger. Each IK chain
   restarts from `throw_ready` and tries candidate yaws until the position residual is < 5 mm. If no yaw reaches,
   the chain retries with `fallback_rot_weight` (0.01, a tilted wrist). This fixed the far-corner ball (XML body
   `ball_green`, x=0.33, y=-0.35), unreachable straight-down from the `throw_ready` branch (joints 1 and 5 hit limits;
   a straight-down solution exists only on a different arm branch), and also the throw-IK misses: 64 -> 73/80.
   Fallback 0.0 -> 71/80, 0.003 -> 70/80.
8. **Aiming by shooting.** MuJoCo is deterministic, so the expert simulates the pre-throw part once, snapshots
   (`get_sim_state` / `set_sim_state`, `mjSTATE_INTEGRATION`), then replays the throw up to
   `aim_iters` times. It measures where the ball descends through `AIM_Z` = 0.16 (rim + radius) and updates
   the aim point with a 2-D Broyden (secant) step. Plain fixed-point iteration oscillated (landing/aim gain ~1.7).
   Planning costs ~0.4 s/episode.
9. **Bin jitter is rejection-sampled** so no two bins are closer (Chebyshev) than `BIN_MIN_SEP` = 0.10, the MJCF's
   nominal spacing. Bins are 0.14 m across their walls, so the old independent ±4 cm jitter pushed a neighbour's wall
   across the target opening; balls landed dead centre (planned error < 1 cm) and rolled away on top of it.
   Fix: 26 -> 29/40.

## Remaining expert failures (13/100), stopped on diminishing returns

- `missed` / `wrong_bucket`: Broyden aim does not converge on some seeds; the landing is still a jagged
  function of aim (fingertip contact right after release). A few extra aim iterations gained only 1/80.
- `never_grasped`: the far-corner ball, a few seeds still leave 3-6 cm IK residual even with the tilt fallback.
  A real fix is planning on a second arm branch (joint 1 around +1.24), with its own collision-free transit.
- Non-target balls occasionally get knocked off the table (seen in a success video); harmless to the metric.

## "Wrong-colored basket" report: explained, not a bug

`eval_expert.py` filenames carry `{ball}-into-{bin}` (e.g. `ok_000_success_green-into-purple.mp4`), and ball/bin
colors are independent. From `frontcam` a landed ball is hidden behind the 13 cm walls and the front bins occlude
the back row, so the video alone cannot show which bin holds the ball; `ball_in_bin` is authoritative. The old
videos in `scripts/artifacts/expert/` and the unlabeled `artifacts/expert/*_missed.mp4` / `ok_010_success.mp4`
predate this code. `eval_policy.py` does not label its videos yet.

## Diagnostic scripts (throwaway, in `artifacts/`, run from repo root)

- `probe.py N`: headless expert over seeds 0..N-1: outcome, planned error, aim log, IK residuals, planning time.
- `grasp_diag.py 1,2,3`: grasp point vs ball (EE frame), finger angles, and contacts at the close tick.
- `release_diag.py 1,2,3`: ball velocity vs nominal at release, and contacts in the ticks after release.
- `trace.py 1,2,3`: first robot-ball contact during the approach.
- `snap.py SEED STEP,STEP`: close-up renders around the target ball to `artifacts/snap.png`.
- `color_check.py`: checks rendered colors and landing bin vs the instruction.
- `probe2.py` / `probe3.py`: finger mesh gap profile; joint-path collision check.
- `sweep.py N key=val ...`: outcome counts over seeds 0..N-1 with `ExpertConfig` overrides (run several in parallel).
- `grip_diag.py N`: when the physical hold starts / is lost relative to the release tick.
- `land_diag.py 1,2,3`: ball position relative to the target bin and its contacts after release.
