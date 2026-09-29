# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

# Project: OpenArm VLA throw pipeline

Read this whole section before touching code. It supersedes `AGENT_HANDOFF.md` where they disagree.

## Goal

Given an instruction like "throw the red ball into the blue bucket", a LoRA-fine-tuned VLA drives the
**right** OpenArm (MuJoCo) to grasp the named ball and throw it into the named floor bin.

Pipeline: Gymnasium env -> scripted expert -> LeRobot-style demo dataset -> VLA fine-tune (SmolVLA default,
pi0/openpi optional) -> closed-loop eval.

The approved plan lives at `c:\Users\black\.cursor\plans\vla_throw_pipeline_bce6bacc.plan.md`.
**Do not edit that plan file.** Its todos (`phase0-env`, `phase1-expert`, `phase2-data`, `phase3-train`,
`phase4-eval`, `docs`) already exist: mark them in progress or completed, never recreate them.

## Hard constraints (from the user and the plan)

- **Do not change** `main.py` (passive viewer; it does not apply the `home` keyframe, intentionally),
  actuator gains, or scene physics in `v2/`.
- Scene colors (bin walls, ball colors) are changed **in Python at reset**, not by editing MJCF.
- Right arm only; the left arm is held at `home` ctrl every step.
- Do not claim a trained policy exists unless a checkpoint exists.

## Status (last updated 2026-09-29)

| Phase | State | Evidence |
|---|---|---|
| 0 Env | **Done** | `uv run python -m pytest -v`: 8/8 pass (7 env/dataset + SmolVLA smoke); `artifacts/phase0_rgb.png` (front and wrist cameras side by side) |
| 1 Expert | **Done, 87% success** (target 85%), physical grasp (no weld) | `eval_expert.py --n-episodes 100`: 87 success, 9 missed, 3 wrong_bucket, 1 never_grasped; `artifacts/expert/summary.json` + labeled mp4s. Headless `sweep.py 80`: 73/80 |
| 2 Dataset | Not started | Writer and scripts exist but are unverified against the current env |
| 3 Train | Not started | `train_smolvla.py` / `train_openpi.py` are dry-run stubs |
| 4 Eval | **Pretrained inference works; no fine-tuned checkpoint exists** | `tests/test_smolvla_smoke.py` and `eval_policy.py --policy smolvla --checkpoint lerobot/smolvla_base --n-episodes 1` run closed loop (~0.45 s/chunk on the GTX 1660); the base model is untrained on OpenArm, so the arm flails (`never_grasped`). `artifacts/smolvla_smoke/` |
| Docs | Not started | `PLAN.md` and `README.md` are required deliverables and are still missing (`README.md` is empty) |

Nothing is committed yet (only the initial "Created Repository" commit exists).

## How to run

```powershell
cd "Z:\1 Github Projects\Robotics\Open Arm Folding"
$env:MUJOCO_GL = "glfw"          # Windows offscreen GL. Linux: egl or osmesa
uv sync --extra train            # Python 3.12; openarm_vla editable + CUDA torch (cu128) + lerobot[smolvla]==0.6.1
# HF_HOME=Z:\hf_cache is set persistently (setx): C: has no room for the ~3 GB of weights

uv run python -m pytest -v                                                   # env + dataset tests
uv run python scripts/eval_expert.py --n-episodes 20 --video-dir artifacts/expert
uv run python artifacts/probe.py 40                                          # fast headless expert check
uv run python scripts/eval_policy.py --policy smolvla --checkpoint lerobot/smolvla_base --n-episodes 1 --video-dir artifacts/smolvla_smoke
uv run python scripts/vla_viewer.py            # live MuJoCo viewer; type instructions in the terminal (r = reset, q = quit)
```

Always run from the **repo root**. `video_dir` / `out` paths are relative, so running from `scripts/`
writes to `scripts/artifacts/...`. Videos from such a run exist there now and may predate the current code.

## Codebase map

```
main.py                         passive viewer (do not change)
v2/pedestal/throw_multi_scene.xml   the scene: 5 balls on a side table, 5 floor bins, inactive welds grasp_right_{color}
openarm_vla/
  constants.py                  joint/actuator names, colors, gripper constants, GRASP_OFFSET, dims
  config.py                     EnvConfig dataclass + load_yaml
  env/throw_env.py              ThrowEnv (Gymnasium): reset randomization, physics_step, contact-based holding(), obs, sim-state save/restore
  env/success.py                ball_in_bin, EpisodeTracker, FailureMode taxonomy
  expert/ik.py                  6-D damped-least-squares IK on a body-fixed point; jacobian_velocity
  expert/ballistic.py           release_velocity (closed-form projectile), quintic_interp
  expert/throw_expert.py        ThrowExpert: open-loop FSM plan + shooting-method aim correction
  data/lerobot_writer.py        npz + mp4 + meta/info.json (NOT real LeRobot parquet yet)
  policies/base.py              Policy protocol: predict_chunk(obs) -> (T, 8)
  policies/dummy.py             DummyPolicy (hold pose, gripper open), ExpertPolicy wrapper
  policies/smolvla.py           SmolVLAAdapter (lerobot 0.6.1). Base checkpoint: features overridden to state 15 / action 8
                                (SmolVLA pads to 32), no norm stats (SO-100 stats are 6-D), fp32 (no bf16 on Turing).
                                A checkpoint with an 8-D action loads its own processors (untested until Phase 3)
  policies/openpi_pi0.py        OpenPiAdapter (raises NotImplementedError after import)
scripts/                        eval_expert, collect_demos, dataset_stats, viz_episode, eval_policy, vla_viewer, train_smolvla, train_openpi
configs/                        env.yaml, expert.yaml, dataset.yaml, train_smolvla.yaml, eval.yaml
tests/                          test_env.py, test_dataset.py, test_smolvla_smoke.py (skipped without lerobot/CUDA)
artifacts/                      gitignored outputs + throwaway diagnostic scripts (see below)
```

## Interfaces (locked)

- Control: **50 Hz**, 20 physics substeps, dt = 0.001.
- Action (8): absolute position targets for right joints 1-7 + `right_finger1_ctrl`, clipped to ctrlrange.
- State (15): right arm qpos (7) + qvel (7) + finger joint qpos (1).
- Obs: `image_front` (`frontcam`), `image_wrist` (`camera_wrist_right`), both 256x256 RGB; `state`; `instruction` string.
- Success: target ball COM inside the target bin's inner box for 0.5 s. Failures: `never_grasped`,
  `dropped_during_grasp`, `missed`, `wrong_bucket`, `timeout`.
- VLA inference: physics paused during `predict_chunk`; chunk 16, replan every 8.

## Decisions taken (and why)

1. **Weld ruling: no weld. The grasp is purely physical (friction), deviating from AGENT_HANDOFF.md.**
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
9. **Color grounding:** ball colors and bin colors are shuffled **independently**, and the target ball color and
   target bin color are sampled independently. So "throw the green ball into the blue bucket" is normal; ball
   color == bin color only happens by chance. Bin walls are recolored to the floor color so color is visible.
10. YAML: write floats like `1.0e-4`. PyYAML parses `1e-4` as a string.
11. **Bin jitter is rejection-sampled** so no two bins are closer (Chebyshev) than `BIN_MIN_SEP` = 0.10, the MJCF's
    nominal spacing. Bins are 0.14 m across their walls, so the old independent ±4 cm jitter pushed a neighbour's wall
    across the target opening; balls landed dead centre (planned error < 1 cm) and rolled away on top of it.
    Fix: 26 -> 29/40.

## Known issues / next steps (in priority order)

1. **Remaining expert failures (13/100), stopped here on diminishing returns.**
   - `missed` / `wrong_bucket`: Broyden aim does not converge on some seeds; the landing is still a jagged
     function of aim (fingertip contact right after release). A few extra aim iterations gained only 1/80.
   - `never_grasped`: the far-corner ball, a few seeds still leave 3-6 cm IK residual even with the tilt fallback.
     A real fix is planning on a second arm branch (joint 1 around +1.24), with its own collision-free transit.
   - Non-target balls occasionally get knocked off the table (seen in a success video); harmless to the metric.
2. **Wrong-colored basket report: explained, not a bug.** `eval_expert.py` filenames now carry
   `{ball}-into-{bin}` (e.g. `ok_000_success_green-into-purple.mp4`), and ball/bin colors are independent
   (decision 9). From `frontcam` a landed ball is hidden behind the 13 cm walls and the front bins occlude the
   back row, so the video alone cannot show which bin holds the ball; `ball_in_bin` is authoritative.
   The old reported videos in `scripts/artifacts/expert/` and the unlabeled `artifacts/expert/*_missed.mp4` /
   `ok_010_success.mp4` predate this code. `eval_policy.py` does not label its videos yet.
3. Phase 2: `collect_demos.py` smoke (3 episodes), then the full N=500. Run `dataset_stats.py` to confirm many
   ball/bin color pairs, and `viz_episode.py`. Upgrade the writer to real LeRobot v2 parquet if lerobot is installed.
   Note `collect_demos.py` records `obs` *before* each action (correct), and re-samples the instruction template
   (the env's own instruction differs; pick one source).
4. Phase 3/4: see `AGENT_HANDOFF.md` sections "Phase 3" and "Phase 4". `eval_policy.py` goes through `env.step`,
   so the policy gets the same physical grasp as the demos.
5. Write `PLAN.md` (architecture, locked decisions, phase results with numbers, weld ruling, chunking tradeoff,
   what was not tested, Phase 5 skipped) and `README.md` (install, data, train both backends, eval, GL notes).
6. Delete the throwaway `artifacts/*.py` diagnostics before committing (the folder is gitignored anyway).

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