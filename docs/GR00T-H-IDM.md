# GR00T-H-IDM: Downstream LeRobot finetuning and IDM-style evaluation

This guide mirrors the **surgical robotic video generator** workflow (GR00T-Dreams Medbot LeRobot prep and IDM scripts, as in common internal tutorials) for **GR00T N1.6 / GR00T-H** (Eagle VLM + flow-matching action head): finetune on a **Medbot-style LeRobot** dataset, then measure **open-loop action error** (MSE / MAE) on a held-out test split — similar to `GR00T-Dreams/scripts/idm_training.py` + `idm_inference_simple.py`, but reusing this repo’s training and eval stack.

---

## 1. Environment setup

From the repository root (Python **3.10** matches `pyproject.toml`):

```bash
git clone --recurse-submodules https://github.com/NVIDIA-Medtech/GR00T-H.git
cd GR00T-H
uv sync --python 3.10
uv pip install -e .
```

Optional: FlashAttention (recommended on NVIDIA GPUs for training):

```bash
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Or use the bundled helper:

```bash
bash scripts/setup_gr00t_h_idm_env.sh
```

**Hugging Face:** set `HF_TOKEN` if you pull checkpoints or datasets from private hubs.

**Weights & Biases:** set `WANDB_API_KEY` only if you pass `--use-wandb` to finetuning.

---

## 2. What is “GR00T-H-IDM” vs stock GR00T-H?

| Aspect | Stock [GR00T-H](../README.md) | This “IDM-style” downstream workflow |
|--------|-------------------------------|--------------------------------------|
| **Goal** | General VLA post-training on Open-H | Same **GR00T-H checkpoint**, finetuned on **your** LeRobot dataset (e.g. Medbot) and scored like an inverse-dynamics benchmark |
| **Model** | GR00T N1.6 (Eagle + diffusion action head) | **Unchanged architecture** — no separate SigLIP IDM backbone (that is [GR00T-Dreams](https://github.com/NVIDIA/GR00T-Dreams)) |
| **Temporal layout** | Open-H uses two video frames at indices **0** and **16**, action chunk **16** steps (`open_h/embodiments/temporal_layout.py`) | Same layout when using registered Open-H configs |
| **Medbot embodiment** | — | `EmbodimentTag.MEDBOT` + `open_h/embodiments/medbot/` (modality aligned with the tutorial’s LeRobot layout) |
| **“IDM” eval** | `gr00t/eval/open_loop_eval.py` (per-trajectory plots) | `gr00t/eval/idm_lerobot_eval.py` aggregates **MSE/MAE** over episodes and writes `inference_summary.json` |

So **GR00T-H-IDM** here means: *GR00T-H training/eval patterns applied to downstream LeRobot data with IDM-like reporting*, not a different model class.

---

## 3. Dataset layout (Medbot LeRobot)

The tutorial’s Medbot layout is supported via **`open_h/embodiments/medbot/modality.json`**:

- **Video:** `left_endo_image` ← `observation.images.left_endo`
- **State / action:** 20D vectors split into `left_cartesian` (3), `left_rotation` (6), `left_jaw` (1), and the same for the right arm
- **Language:** `task_index` in parquet → text via `meta/tasks.jsonl` (annotation key `task`)

If you already converted data with that Medbot LeRobot pipeline, **replace** `meta/modality.json` with the one from this repo if your file used a different language key (this repo standardizes on **`annotation.task`** for Open-H).

Copy into each dataset you train or evaluate on:

```bash
REPO_ROOT=/path/to/GR00T-H
DATASET=/path/to/train_lerobot   # or test_lerobot

mkdir -p "$DATASET/meta"
cp "$REPO_ROOT/open_h/embodiments/medbot/modality.json" "$DATASET/meta/modality.json"
```

---

## 4. Prepare normalization statistics

You need **`meta/stats.json`** (raw) and **`meta/temporal_stats.json`** (action normalization for training). The repo script runs both steps:

```bash
cd "$REPO_ROOT"

bash open_h/prepare_datasets.sh \
  --embodiment-tag medbot \
  --modality-json open_h/embodiments/medbot/modality.json \
  /path/to/train_lerobot

# Repeat for test data if you evaluate on a separate folder:
bash open_h/prepare_datasets.sh \
  --embodiment-tag medbot \
  --modality-json open_h/embodiments/medbot/modality.json \
  /path/to/test_lerobot
```

`prepare_datasets.sh` copies the modality file, runs `gr00t/data/stats.py`, then you still need **temporal** stats via `launch_finetune.py` (single GPU is enough):

```bash
uv run torchrun --nproc_per_node=1 --master_port=29501 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-H \
  --dataset-path /path/to/train_lerobot \
  --embodiment-tag MEDBOT \
  --calculate-norm-stats
```

Repeat `--calculate-norm-stats` for `/path/to/test_lerobot` if that directory is used at inference time (so `temporal_stats.json` exists there too).

**Train/test splits:** if `meta/info.json` defines `splits`, add to finetune:

- `--include-splits train` (or your split name) for training-only stats and optimization
- For eval, use `idm_lerobot_eval.py` with `--include-splits test` (below)

---

## 5. Finetuning (Medbot, LeRobot)

Example: 8 GPUs, batch size 32, 20k steps (adjust to your hardware):

```bash
uv run torchrun --nproc_per_node=8 --master_port=29500 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-H \
  --dataset-path /path/to/train_lerobot \
  --embodiment-tag MEDBOT \
  --num-gpus 8 \
  --global-batch-size 32 \
  --max-steps 20000 \
  --learning-rate 1e-4 \
  --output-dir /path/to/medbot_gr00t_h_ft \
  --include-splits train
```

Notes:

- **`--embodiment-tag MEDBOT`** uses the registered config in `open_h/embodiments/medbot/medbot_config.py` (no `--modality-config-path` needed).
- Omit `--include-splits` if the dataset has no split metadata or you use a train-only folder.
- Tune `--global-batch-size`, `--max-steps`, and `--num-gpus` to match your cluster.
- For **vision-only** style training (state dropped at the model), see `open_h/gr00t_h_config.yaml` patterns and `state_dropout_prob_per_embodiment` in upstream docs — the default finetune script uses standard state conditioning unless you change those fields in code/config.

Checkpoints appear under `--output-dir` as `checkpoint-<step>/`.

---

## 6. Inference and benchmark (IDM-style MSE / MAE)

This matches the spirit of:

```bash
# GR00T-Dreams tutorial (reference)
python scripts/idm_inference_simple.py \
  --checkpoint $OUTPUT/checkpoint-10000 \
  --dataset $HOME/dataset/test_lerobot \
  --output-dir $OUTPUT \
  --data-config medbot \
  --embodiment-tag new_embodiment \
  --batch-size 16 \
  --observation-indices 0 8
```

**GR00T-H equivalent** (same policy path as `open_loop_eval.py`: `Gr00tPolicy` + `LeRobotEpisodeLoader`):

```bash
uv run python gr00t/eval/idm_lerobot_eval.py \
  --checkpoint /path/to/medbot_gr00t_h_ft/checkpoint-20000 \
  --dataset /path/to/test_lerobot \
  --output-dir /path/to/medbot_gr00t_h_ft/idm_eval_test \
  --embodiment-tag MEDBOT \
  --include-splits test \
  --action-horizon 16 \
  --no-plots
```

Outputs:

- **`idm_eval_test/inference_summary.json`** — average and per-episode **MSE** / **MAE** in **unnormalized dataset space** (same convention as `gr00t/eval/open_loop_eval.py`).
- Without `--no-plots`, saves one JPEG per trajectory under `--output-dir`.

**Evaluate all episodes** in a test-only folder (no `splits` in `info.json`):

```bash
uv run python gr00t/eval/idm_lerobot_eval.py \
  --checkpoint /path/to/checkpoint-20000 \
  --dataset /path/to/test_lerobot \
  --output-dir /path/to/idm_eval \
  --embodiment-tag MEDBOT \
  --max-episodes 50
```

**Explicit trajectory IDs:**

```bash
uv run python gr00t/eval/idm_lerobot_eval.py \
  --checkpoint /path/to/checkpoint-20000 \
  --dataset /path/to/test_lerobot \
  --output-dir /path/to/idm_eval \
  --embodiment-tag MEDBOT \
  --traj-ids 0 1 2 3
```

There is **no** `--observation-indices` flag here: video/state time offsets are defined by the **checkpoint’s processor** and the **MEDBOT** modality config (`OPEN_H_VIDEO_DELTA_INDICES` = `[0, 16]`). To change them you would edit `open_h/embodiments/temporal_layout.py` and retrain.

---

## 7. File map

| Item | Role |
|------|------|
| `open_h/embodiments/medbot/medbot_config.py` | Registers `EmbodimentTag.MEDBOT` |
| `open_h/embodiments/medbot/modality.json` | LeRobot column layout for Medbot tutorial data |
| `open_h/embodiments/temporal_layout.py` | Shared 16-step action horizon and video deltas |
| `gr00t/experiment/launch_finetune.py` | Finetuning + `--calculate-norm-stats` |
| `gr00t/eval/idm_lerobot_eval.py` | Aggregated IDM-style metrics + optional plots |
| `gr00t/eval/open_loop_eval.py` | Lower-level single-trajectory eval (used internally) |

---

## 8. Troubleshooting

- **`Embodiment 'medbot' not found in MODALITY_CONFIGS`** — ensure `import open_h.embodiments` runs before training ( `launch_finetune.py` already imports it).
- **Missing `stats.json` / `temporal_stats.json`** — run `prepare_datasets.sh` and `launch_finetune.py --calculate-norm-stats` for that dataset path.
- **Language / task errors** — `modality.json` must define `annotation.task` → `task_index` as in this repo’s `medbot/modality.json`.
- **Video backend** — default in `idm_lerobot_eval.py` is `torchcodec`; if decoding fails, try `--video-backend decord` where supported.

For the original **GR00T-Dreams** IDM (SigLIP, separate checkpoint family), use the **GR00T-Dreams** repository and its Medbot / `idm_inference_simple.py` scripts (same ideas as the surgical synthetic-data tutorial).
