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

Read this whole section before touching code. `AGENT_HANDOFF.md` holds the detailed expert design history and
diagnostic scripts; read it only when working on the expert or env mechanics.

## Goal

Given an instruction like "throw the red ball into the blue bucket", a fine-tuned VLA drives the **right**
OpenArm (MuJoCo) to grasp the named ball and throw it into the named floor bin. The VLA sees only camera
images, the arm's own joint state and the text; it never gets ball/bin positions (the expert does).

Pipeline: Gymnasium env -> scripted expert -> lerobot demo dataset -> SmolVLA fine-tune -> closed-loop eval.

The approved plan lives at `c:\Users\black\.cursor\plans\vla_throw_pipeline_bce6bacc.plan.md`.
**Do not edit that plan file.** Its todos (`phase0-env`, `phase1-expert`, `phase2-data`, `phase3-train`,
`phase4-eval`, `docs`) already exist: mark them in progress or completed, never recreate them.

## Hard constraints (from the user and the plan)

- **Do not change** `main.py` (passive viewer; it does not apply the `home` keyframe, intentionally),
  actuator gains, or scene physics in `v2/`.
- Scene colors (bin walls, ball colors) are changed **in Python at reset**, not by editing MJCF.
- Right arm only; the left arm is held at `home` ctrl every step.
- Do not claim a trained policy exists unless a checkpoint exists.
- Never let downloads land on `C:` (nearly full). See "Environment" below.

## Status (last updated 2026-09-29)

| Phase | State | Evidence |
|---|---|---|
| 0 Env | **Done** | `pytest -v`: 8/8 pass (7 env/dataset + SmolVLA smoke); `artifacts/phase0_rgb.png` |
| 1 Expert | **Done, 87% success** (target 85%), physical grasp (no weld) | `eval_expert.py --n-episodes 100`: 87 success, 9 missed, 3 wrong_bucket, 1 never_grasped; `artifacts/expert/summary.json` |
| 2 Dataset | **Pipeline done; 500-demo set NOT collected yet** | Real lerobot v3 dataset writer, `collect_demos.py`, deterministic replay verified (`viz_episode.py --replay`: state diff 0.0, `success`). Datasets on disk: `data/datasets/smoke3` (3 eps), `data/datasets/overfit10` (10 eps) |
| 3 Train | **Launcher works; only overfit checkpoints exist** | `scripts/train_smolvla.py` runs lerobot's trainer (see "Training"). 100M of 450M params trainable. Speed 2.2-3.5 s/step at batch 8, 4.5 GB allocated (5.7 GB reserved of 6 GB) |
| 4 Eval | **Overfit gate FAILED: 0/10 on the training seeds** | `checkpoints/overfit10/checkpoints/002000/pretrained_model` (2000 steps, loss 3.0 -> 0.027). Result in `artifacts/overfit10/eval_summary.json`: 9 `never_grasped`, 1 `missed`. Videos show roughly the right motion but heavy twitching. Diagnosis not done, see "Overfit result and next steps". Fine-tuned adapter path is verified to load and run |
| Docs | Not started | `PLAN.md` and `README.md` are required deliverables (`README.md` is empty) |

Nothing is committed yet (only the initial "Created Repository" commit exists).

## Environment (verified 2026-09-29)

- Python 3.12, torch 2.11 with CUDA 12.8, lerobot 0.6.1, transformers 5.5, mujoco 3.14. GPU: GTX 1660
  (6 GB, sm_75), which has **no bf16 compute**, so run in fp32. 12 logical CPUs.
- **C: has ~3 GB free of 223 GB. Nothing may be written to C:.** `HF_HOME` / `UV_CACHE_DIR` set via setx are NOT
  seen by Cursor terminals, and `TEMP` is on C: by default (video encoding, pytest and pip write there). In **every**
  new shell run first:
  `$env:HF_HOME="Z:\hf_cache"; $env:UV_CACHE_DIR="Z:\uv_cache"; $env:TEMP="Z:\tmp"; $env:TMP="Z:\tmp"; $env:TORCH_HOME="Z:\hf_cache\torch"; $env:XDG_CACHE_HOME="Z:\hf_cache\xdg"; $env:MUJOCO_GL="glfw"`.
  Check C: free space with `[System.IO.DriveInfo]::new('C').AvailableFreeSpace` (`Get-PSDrive` returns nothing/0 and
  `Get-CimInstance`/`fsutil` fail here). Big C: folders (Cursor sandbox cache, pip cache) are not ours; don't delete.
- **Video decoding uses pyav, not torchcodec** (torchcodec installs but can't load: no shared FFmpeg; the user chose
  pyav to avoid a system install). `openarm_vla/data/lerobot_writer.py` sets `VIDEO_BACKEND = "pyav"` and passes it
  to `create_dataset` / `open_dataset`; the trainer gets `--dataset.video_backend=pyav`. Encoding is AV1 (SVT), a
  10-episode record takes ~3 min.
- Long jobs: launch with `Start-Process .venv\Scripts\python.exe ... -RedirectStandardOutput/-Error ... -WindowStyle Hidden`
  so a killed Cursor terminal doesn't matter (a killed terminal does NOT stop a detached python: kill it explicitly with
  `Stop-Process`, remember the dataloader workers). Progress bars go to the `-RedirectStandardError` file.
- rg/Glob skip `.venv`. To read lerobot source, locate it with
  `uv run python -c "import lerobot,os;print(os.path.dirname(lerobot.__file__))"` and read files by path.
  Use the installed 0.6.1 source, not memory or old docs (`lerobot.common.*` paths are gone).

## How to run

```powershell
cd "Z:\1 Github Projects\Robotics\Open Arm Folding"
$env:MUJOCO_GL = "glfw"          # Windows offscreen GL. Linux: egl or osmesa
uv sync --extra train            # openarm_vla editable + CUDA torch (cu128 index) + lerobot[smolvla,dataset]==0.6.1

uv run python -m pytest -v
uv run python scripts/collect_demos.py --n-success 10 --max-attempts 30 --out data/datasets/NAME   # --out must NOT exist
uv run python scripts/dataset_stats.py --dataset data/datasets/NAME
uv run python scripts/viz_episode.py --dataset data/datasets/NAME --episode 0 --replay
uv run python scripts/train_smolvla.py --dataset-root data/datasets/NAME --output-dir checkpoints/RUN --steps 2000 --warmup-steps 100   # output dir must NOT exist
uv run python scripts/eval_policy.py --policy smolvla --checkpoint checkpoints/RUN/checkpoints/002000/pretrained_model --seeds-file data/datasets/NAME/openarm_seeds.json --video-dir artifacts/RUN
uv run python scripts/eval_expert.py --n-episodes 20 --video-dir artifacts/expert
uv run python scripts/eval_policy.py --policy smolvla --checkpoint lerobot/smolvla_base --n-episodes 1 --video-dir artifacts/smolvla_smoke
uv run python scripts/vla_viewer.py            # live MuJoCo viewer; type instructions in the terminal (r = reset, q = quit)
```

Always run from the **repo root**: `video_dir` / `out` paths are relative.

## Codebase map

```
main.py                         passive viewer (do not change)
v2/pedestal/throw_multi_scene.xml   the scene: 5 balls on a side table, 5 floor bins, inactive welds grasp_right_{color}
openarm_vla/
  constants.py                  joint/actuator names, colors, gripper constants, GRASP_OFFSET, dims
  config.py                     EnvConfig dataclass + load_yaml
  env/throw_env.py              ThrowEnv (Gymnasium): reset randomization, physics_step, contact-based holding(), obs, sim-state save/restore
  env/success.py                ball_in_bin, EpisodeTracker, FailureMode taxonomy
  expert/                       ik.py (DLS IK), ballistic.py, throw_expert.py (FSM plan + shooting aim); see AGENT_HANDOFF.md
  data/lerobot_writer.py        FEATURES, create_dataset(root), open_dataset(root), VIDEO_BACKEND (real lerobot v3 dataset)
  policies/base.py              Policy protocol: predict_chunk(obs) -> (T, 8)
  policies/dummy.py             DummyPolicy (hold pose, gripper open), ExpertPolicy wrapper
  policies/smolvla.py           SmolVLAAdapter (lerobot 0.6.1), see "How inference is wired"
  policies/openpi_pi0.py        OpenPiAdapter (raises NotImplementedError after import)
scripts/                        eval_expert, collect_demos, dataset_stats, viz_episode, eval_policy, vla_viewer, train_smolvla, train_openpi
configs/                        env.yaml, expert.yaml, dataset.yaml, train_smolvla.yaml, eval.yaml
tests/                          test_env.py, test_dataset.py, test_smolvla_smoke.py (skipped without lerobot/CUDA)
artifacts/                      gitignored outputs + throwaway diagnostic scripts (listed in AGENT_HANDOFF.md)
```

## Interfaces (locked)

- Control: **50 Hz**, 20 physics substeps, dt = 0.001. Episodes cap at 400 steps.
- Action (8): absolute position targets for right joints 1-7 + `right_finger1_ctrl`, clipped to ctrlrange.
- State (15): right arm qpos (7) + qvel (7) + finger joint qpos (1).
- Obs: `image_front` (`frontcam`), `image_wrist` (`camera_wrist_right`), both 256x256 RGB; `state`; `instruction` string.
- Success: target ball COM inside the target bin's inner box for 0.5 s. Failures: `never_grasped`,
  `dropped_during_grasp`, `missed`, `wrong_bucket`, `timeout`.
- VLA inference: physics paused during `predict_chunk`; chunk 16, replan every 8.

## Key decisions (details and measurements in AGENT_HANDOFF.md)

1. **No weld:** the grasp is purely physical (friction); `grasp_right_*` welds stay disabled. Demos and VLA eval
   use identical mechanics.
2. **Gripper polarity:** finger joint `0` = closed, `-0.7854` = fully open.
3. **Color grounding:** ball colors, bin colors, target ball and target bin are all sampled **independently**
   (25 ball/bin pairs), so the policy must read the text. Bin walls are recolored in Python so color is visible.
   From `frontcam` a landed ball is hidden by the walls; `ball_in_bin` is authoritative, not the video.
4. YAML: write floats like `1.0e-4`. PyYAML parses `1e-4` as a string.

## How inference is wired (keep compatible)

- `SmolVLAAdapter(checkpoint)` feeds `observation.images.image_front`, `observation.images.image_wrist`
  (3x256x256, float 0..1), `observation.state` (15) and `task` (string) through lerobot's preprocessor, calls
  `predict_action_chunk`, then the postprocessor, and returns the first 16 of the model's 50 steps as (16, 8).
- **Base checkpoint** (6-D SO-100 action): the adapter overrides the config features to 15/8 (SmolVLA pads to 32)
  and uses processors with no norm stats. That is why base-model actions are meaningless on OpenArm.
- **Fine-tuned checkpoint** (8-D action): the adapter loads the checkpoint's own processors via
  `make_pre_post_processors(cfg, ckpt, preprocessor_overrides={"device_processor": {"device": ...}})`.
  Verified to load and run on the bench and overfit checkpoints. The checkpoint's preprocessor carries the
  `rename_observations_processor` (image_front -> camera1, image_wrist -> camera2), so the adapter keys are unchanged.
  **Unresolved:** the checkpoint config still declares `observation.state` shape [6] (inherited from smolvla_base) although
  data/stats are 15-D; it trains and runs without errors but has not been checked for correct normalization.
- So the dataset **must** use exactly these keys: `observation.images.image_front`, `observation.images.image_wrist`,
  `observation.state` (15, float32), `action` (8, float32), plus the task string.

## Phase 2/3 results (done; do not redo)

Done and verified: deps -> lerobot-format writer -> `collect_demos.py` -> replay check -> training launcher ->
benchmark -> overfit run.

**Dataset**
- `collect_demos.py` records through `LeRobotDataset` (`create` / `add_frame` / `save_episode` / `clear_episode_buffer` /
  `finalize`). It stores obs *before* each action, keeps only `success` episodes (failed frames are cleared; tested in
  `tests/test_dataset.py`), uses `env.task["instruction"]` as the task string (single source; the old template
  re-sampling is gone) and writes `<out>/openarm_seeds.json` (`episode_index, seed, instruction, ball_color, bin_color`).
  `--out` must not exist. For parallel runs give each process its own `--out` and `--seed-offset`.
- Episodes are ~222 steps (not 400); 10 episodes ~2.2k frames. Expert success in collection was 10/14 attempts.
- `viz_episode.py --replay` re-runs the stored actions from the stored seed: state diff 0.0, result `success`,
  video-vs-render pixel diff ~0.008. `dataset_stats.py` prints lengths, action stats and the color-pair histogram.
- Merging parallel datasets: `lerobot.datasets.aggregate.aggregate_datasets(repo_ids, aggr_repo_id, roots=, aggr_root=)`
  exists (not yet used); also concatenate the `openarm_seeds.json` files and shift `episode_index`.

**Training** (`scripts/train_smolvla.py` + `configs/train_smolvla.yaml`, runs `lerobot.scripts.lerobot_train` in-process)
- Fine-tunes `lerobot/smolvla_base`: `freeze_vision_encoder` / `train_expert_only` are true, so 100M of 450M params train.
- Windows workarounds already in the launcher: (1) `--policy.path` is a local `snapshot_download` dir, because lerobot
  wraps it in `Path` and `lerobot/smolvla_base` becomes `lerobot\smolvla_base`; (2) `lt.make_policy` is wrapped with
  `.float()` (the VLM loads as bf16, the 1660 has no bf16 compute); (3) `lt.update_last_checkpoint` is a no-op (needs
  Windows symlink privilege); (4) `--rename_map` maps image_front/image_wrist to camera1/camera2 (required, else the
  visual-feature check fails); (5) scheduler warmup/decay are overridden (defaults 1000/30000 are wrong for short runs).
- Only one checkpoint is saved (`save_freq = steps`). Output dir must not exist. Model load can take minutes cold.
- Measured: 2.2-3.5 s/step at batch 8 (overfit run averaged ~2.4 s): 2000 steps ~80 min, 20k steps ~12-19 h.

**Overfit run** (10 eps, 2000 steps, warmup 100, lr 1e-4): loss 2.98 (step 10) -> 0.55 (100) -> 0.11 (800) -> 0.076
(1000) -> 0.027 (2000), still falling. Eval on the 10 training seeds: **0 success**, 9 `never_grasped`, 1 `missed`
(orange->purple, throw error 0.21). Videos in `artifacts/overfit10/`: right general motion, too much twitching.
The gate (>= 5/10) failed. The 500-demo collection has NOT been started; don't start it or the long run yet.

### Overfit result: next steps (unresolved; do these before scaling)

Hypotheses, ranked: (1) flow matching resamples noise on every `predict_chunk`, so replanning every 8 steps hops
between trajectories, and the grasp needs cm precision (gripper opens ~15 cm around a 6 cm ball); (2) closed-loop
compounding error on states the 10 demos never covered; (3) only 10 scenes with a frozen vision encoder; (4) the 6-D
declared state, and an unverified harness.

1. **Harness control:** run `ExpertPolicy` (`policies/dummy.py`) through `eval_policy.py`'s chunk-16 / replan-8 loop. It
   should get ~85%; if not, the harness is the bug, not the model.
2. **Offline check:** feed the 10 training episodes' recorded observations to the checkpoint and compare predicted
   16-step chunks with the dataset actions (no simulator). Small error => closed-loop/noise problem; large => not learned.
3. **Inference-only fixes on the same checkpoint** (no retrain): fixed noise reused across replans, replan every 16-25
   steps, or temporal ensembling of overlapping chunks. Also check state normalization (6-D vs 15-D config).
4. Only then retrain (longer than 2000 steps, more episodes, or a state-shape fix) and compare with the 0/10 baseline.
   Do the 500-demo collection and the 10k-20k-step fine-tune only once the pipeline is sound (12-19 h on this card; a
   cloud A100 ~2-4 h). The user decides whether the overfit gate is waived.

**After the first fine-tuned checkpoint** (not before), test language grounding within throwing before adding
tasks: hold out some ball/bin color pairs from training and test on them, try phrasings outside the 5 templates,
and change only the text on an identical seed to check the robot targets a different ball or bin. Only then add
a clearly different second task (scenes exist in `v2/pedestal/`: valve, bottle, peg_socket, move_puck,
articulated, balance, catch). Each needs its own scripted expert and success check; that's the main cost.

## Other open items

- Write `PLAN.md` (architecture, locked decisions, phase results with numbers, weld ruling, chunking tradeoff,
  what was not tested, Phase 5 skipped) and `README.md` (install, data, train, eval, GL notes, HF_HOME).
- `eval_policy.py` does not label its videos with `{ball}-into-{bin}` yet (`eval_expert.py` does).
- Delete the throwaway `artifacts/*.py` diagnostics before committing (the folder is gitignored anyway).
