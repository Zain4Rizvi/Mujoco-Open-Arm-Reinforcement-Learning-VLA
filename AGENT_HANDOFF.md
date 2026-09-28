# Agent handoff: OpenArm VLA throw pipeline

This is a status dump for the **next agent**. Do **not** edit the Cursor plan file at `c:\Users\black\.cursor\plans\vla_throw_pipeline_bce6bacc.plan.md`. Implement against that plan + this file. Existing todos (`phase0-env` … `docs`) already exist — mark them in_progress/completed; do not recreate them.

**Do not change** [`main.py`](main.py), actuator gains, or scene physics. Bin **wall colors** are applied in Python at reset (not via MJCF edit).

---

## What was done (this session)

Scaffolding for Phases 0–4 exists on disk. **Nothing was verified**: no pytest run, no expert eval, no demo collection, no training dry-run, no `PLAN.md` / `README.md` (those two files are still missing and are required deliverables).

### Code added

| Area | Paths |
|------|--------|
| Env | [`openarm_vla/env/throw_env.py`](openarm_vla/env/throw_env.py), [`openarm_vla/env/success.py`](openarm_vla/env/success.py) |
| Expert | [`openarm_vla/expert/ik.py`](openarm_vla/expert/ik.py), [`ballistic.py`](openarm_vla/expert/ballistic.py), [`throw_expert.py`](openarm_vla/expert/throw_expert.py) |
| Data | [`openarm_vla/data/lerobot_writer.py`](openarm_vla/data/lerobot_writer.py) (npz + mp4 + `meta/info.json`, **not** real parquet) |
| Policies | [`openarm_vla/policies/base.py`](openarm_vla/policies/base.py), [`dummy.py`](openarm_vla/policies/dummy.py), [`smolvla.py`](openarm_vla/policies/smolvla.py), [`openpi_pi0.py`](openarm_vla/policies/openpi_pi0.py) |
| Scripts | [`scripts/eval_expert.py`](scripts/eval_expert.py), [`collect_demos.py`](scripts/collect_demos.py), [`dataset_stats.py`](scripts/dataset_stats.py), [`viz_episode.py`](scripts/viz_episode.py), [`eval_policy.py`](scripts/eval_policy.py), [`train_smolvla.py`](scripts/train_smolvla.py), [`train_openpi.py`](scripts/train_openpi.py) |
| Tests | [`tests/test_env.py`](tests/test_env.py), [`tests/test_dataset.py`](tests/test_dataset.py) |
| Config | [`configs/*.yaml`](configs/) |
| Deps | [`pyproject.toml`](pyproject.toml) lists gymnasium, pyyaml, imageio, pytest; optional `[train]` extra |

### Locked design (already in the approved plan)

- Scene: `v2/pedestal/throw_multi_scene.xml`, **right arm only** (8-D absolute position + gripper), 50 Hz, 20 substeps, dt=0.001.
- State dim **15** (7 qpos + 7 qvel + 1 gripper). Action dim **8**.
- Cameras: `frontcam` + `camera_wrist_right`, 256×256 for VLA.
- Expert: DLS IK + ballistic release velocity; **weld-on-hold by default** (`use_grasp_weld: true`) plus impart `qvel` at release.
- Inference: **pause physics** during `predict_chunk`. Chunk 16, replan every 8.
- Train default: SmolVLA LoRA; pi0/openpi behind the same protocol, not vendored.

### Known issues the next agent must fix

1. **Unverified env.** `ThrowEnv` may fail on: `spaces.Text`, unnamed `freejoint` IDs, `data.eq_active`, `model.eq(...)`, Renderer/GL on Windows (`MUJOCO_GL=glfw`).
2. **Expert likely <85%.** IK is position-only from `throw_ready`; weld relative pose is not written into `eq_data`; release uses a hardcoded Cartesian height (`0.95 m`) that may be unreachable. Tune after `eval_expert`.
3. **Dataset format shortcut.** Writer stores `episode_XXXXXX.npz` + mp4, not LeRobot parquet. Convert or use `lerobot` APIs when generating the real dataset.
4. **SmolVLA import path is a guess** (`lerobot.common.policies.smolvla...`) — pin to the installed lerobot version.
5. **`OpenPiAdapter` raises `NotImplementedError`** after a successful `import openpi`.
6. **`pyproject.toml` has no `[build-system]`** — `uv pip install -e .` may fail until hatchling/setuptools is added.
7. **`eval_expert.py` / `eval_policy.py`** assume env is already reset then call `_obs`; collect_demos does a proper loop.
8. **`main.py` left untouched** (viewer still does not apply the `home` keyframe). That is intentional.

---

## What you (next agent) must execute

Work **phase by phase**. After each phase, run the verification command and record the result in `PLAN.md` (create it). Keep `main.py` runnable.

### Phase 0 — Env (todo `phase0-env`)

1. `uv sync` / install deps from `pyproject.toml`.
2. Run `python -m pytest tests/test_env.py -v`.
3. Fix failures until green. Write `artifacts/phase0_rgb.png` (test already tries this).
4. Confirm: seed-stable reset, action clip, planted-ball success, two RGB cameras.

### Phase 1 — Expert (todo `phase1-expert`)

1. `python scripts/eval_expert.py --n-episodes 20 --video-dir artifacts/expert` (smoke), then `--n-episodes 100`.
2. Tune IK/release/weld until success **≥85%** or document the gap in `PLAN.md`.
3. Save success and failure mp4s + `artifacts/expert/summary.json`.

### Phase 2 — Dataset (todo `phase2-data`)

1. Smoke: `python scripts/collect_demos.py --n-success 3 --max-attempts 20 --out data/datasets/openarm_throw_smoke`.
2. Full: default N=500 in `configs/dataset.yaml` (slow; can start smaller and note it).
3. `python scripts/dataset_stats.py --dataset <path>` — confirm many ball/bin color pairs (no 1:1 shortcut).
4. `python scripts/viz_episode.py --dataset <path> --episode 0`.
5. If you have lerobot, upgrade writer to real v2 parquet.

### Phase 3 — Train (todo `phase3-train`)

1. `python scripts/train_smolvla.py --dry-run` and `--overfit`.
2. `python scripts/train_openpi.py`.
3. Wire real lerobot/openpi CLIs in **README.md**; pin versions; GPU memory notes (SmolVLA LoRA ~8–12 GB; pi0 much more).
4. Do not claim a trained policy unless a checkpoint exists.

### Phase 4 — Eval (todo `phase4-eval`)

1. `python scripts/eval_policy.py --policy expert --n-episodes 10 --video-dir artifacts/eval`.
2. `python scripts/eval_policy.py --policy dummy --n-episodes 3`.
3. If a checkpoint exists: `--policy smolvla --checkpoint ... --match-seeds`.
4. Write a small results table (even if VLA untrained).

### Docs (todo `docs`) — required even if later phases are partial

Create:

- **`PLAN.md`** at repo root: architecture, locked decisions (copy from the approved plan), phase status (pass/fail + numbers), expert weld ruling, chunking tradeoff, what was not tested.
- **`README.md`**: exact commands for install, data gen, train (both backends), eval; Windows `MUJOCO_GL=glfw`; Linux EGL/osmesa.

Phase 5 (DAgger / distractors / hybrid release) only if time; otherwise one paragraph in `PLAN.md` saying skipped.

---

## Suggested first commands

```powershell
cd "Z:\1 Github Projects\Robotics\Open Arm Folding"
$env:MUJOCO_GL = "glfw"
uv sync
python -m pytest tests/test_env.py tests/test_dataset.py -v
```

If `uv sync` fails on `[build-system]`, add hatchling and `packages = ["openarm_vla"]`, or run with `PYTHONPATH=.`.

---

## Todo mapping

| ID | Status at handoff | Next action |
|----|-------------------|-------------|
| `phase0-env` | in_progress / unverified | pytest + RGB artifact |
| `phase1-expert` | pending | eval_expert + tune |
| `phase2-data` | pending | collect + stats + viz |
| `phase3-train` | pending | dry-run + README train cmds |
| `phase4-eval` | pending | expert + dummy (+ VLA if ckpt) |
| `docs` | in_progress | write `PLAN.md` + `README.md` |

Mark each completed only after its verification command has been run and the output inspected.
