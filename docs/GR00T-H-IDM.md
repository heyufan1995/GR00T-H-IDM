# GR00T-H-IDM: Downstream LeRobot finetuning and IDM-style evaluation

This guide mirrors the **surgical robotic video generator** workflow (GR00T-Dreams Medbot LeRobot prep and IDM scripts, as in common internal tutorials) for **GR00T N1.6 / GR00T-H** (Eagle VLM + flow-matching action head): finetune on a **Medbot-style LeRobot** dataset, then measure **open-loop action error** (MSE / MAE) on a held-out test split — similar to `GR00T-Dreams/scripts/idm_training.py` + `idm_inference_simple.py`, but reusing this repo’s training and eval stack.

**§3–§6** use one **worked path layout** (`REPO_ROOT`, `DATASET`, `TEST_DATASET`, `OUTPUT_DIR`). Edit those four variables to match your cluster; the commands are otherwise copy-pasteable.

---

## 1. Environment setup

This project pins **Python 3.10** in `pyproject.toml`. Use **uv** so the environment is reproducible and isolated.

### 1.1 Clone and create the project virtualenv

From a directory where you keep code:

```bash
git clone --recurse-submodules https://github.com/NVIDIA-Medtech/GR00T-H.git
cd GR00T-H   # or your fork, e.g. GR00T-H-IDM — stay in the repo root for every uv command below
```

**Create / refresh the venv and install dependencies** (`uv` writes a project env at **`.venv/`** by default):

```bash
uv sync --python 3.10
```

- **`uv sync`** installs **all default dependencies** from `pyproject.toml` / `uv.lock` **and** installs **this repository as an editable package** (`gr00t`, `open_h`). You do **not** need a separate `uv pip install -e .` for a normal setup — that line only repeats the same editable install and can make it look like something is “missing” when the real gap is **FlashAttention** (below).

**Why FlashAttention is not installed by `uv sync` alone:** in `pyproject.toml`, **`flash-attn` is not in the main `dependencies` list**. It is listed only under **`[project.optional-dependencies] gpu`** together with ONNX/TensorRT. So:

- **`uv sync --python 3.10`** → core stack + editable `gr00t`, **no** `flash_attn` unless you add extras.
- **`uv sync --python 3.10 --extra gpu`** → tries to install the whole **`gpu`** extra (FlashAttention **and** heavy packages like TensorRT). That often **fails** on generic Linux nodes if TensorRT wheels don’t match your platform.

**Recommended way to use Flash Attention 2:** after `uv sync`, install **only** `flash-attn` (see **§1.3**).

**Always run `uv run …` and `uv pip …` from the repository root** (where `pyproject.toml` lives) so commands use **this** `.venv**.

Or use the helper (same steps):

```bash
bash scripts/setup_gr00t_h_idm_env.sh
```

In **§3**, set `REPO_ROOT` to this clone path so all later commands use the same tree and `.venv`.

### 1.2 Verify the interpreter

You should see a path under **`.venv`**:

```bash
uv run python -c "import sys; print(sys.executable)"
uv run python -c "import torch; print('torch', torch.__version__)"
```

If `sys.executable` is **not** inside your clone (e.g. it points at Conda), deactivate other envs (`conda deactivate` until clean) or run commands only from the repo root with `uv run`.

### 1.3 FlashAttention 2 (recommended for training; default model path)

**Order matters:** run **`uv sync` first** so **`torch==2.7.1`** (or whatever the lockfile pins) is in **this repo’s `.venv`**. **`flash-attn` builds and links against that PyTorch**, so **`torch` must already be installed in the same environment** before you install `flash-attn`.

**Install into the same environment that runs `torchrun` / `uv run`.** From the **repository root** (where `pyproject.toml` lives):

```bash
cd /path/to/your/GR00T-H-clone
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
```

`uv pip` targets the project **`.venv`** when you run it from the repo root and no other env hijacks it. That is the **correct** pattern.

If **`uv pip install`** prints a **Conda** path (e.g. `Using Python … environment at: …/miniconda3/envs/…`), force the project venv:

```bash
uv pip install --python .venv/bin/python flash-attn==2.7.4.post1 --no-build-isolation
```

**Common mistake:** installing `flash-attn` into a **Conda** env (e.g. `conda activate gr00t` → `uv pip install …` may still report that Conda path) while **`torchrun`** is launched with **`uv run`** or **`.venv/bin/python`**. Training then uses **`.venv`**, which has **no** `flash_attn`, and Transformers raises *“FlashAttention2 has been toggled on, but … flash_attn seems to be not installed”* even though Conda can import it. **Fix:** run **`uv pip install flash-attn==2.7.4.post1 --no-build-isolation`** from the **repo root** (or use **`.venv/bin/python -m pip install flash-attn==2.7.4.post1 --no-build-isolation`** explicitly), then verify with **`uv run`** (below).

Verify with the **same interpreter** as training:

```bash
uv run python -c "import flash_attn; print('flash_attn OK', flash_attn.__version__)"
```

If that fails, `flash_attn` is not in the env your job will use.

Requirements (typical failures if missing):

- **OS:** Linux **x86_64** with NVIDIA **CUDA** (not macOS CPU).
- **Toolchain:** `nvcc` and a CUDA toolkit compatible with your PyTorch build, plus a working C++ compiler (often `ninja`).

**If you cannot build or use `flash-attn`**, the default GR00T-H / Eagle stack expects FlashAttention 2 when loading from Hub; disabling it requires changing model / checkpoint **`use_flash_attention`** (not exposed as a single `launch_finetune.py` flag in this repo). Prefer fixing the **§1.3** install in **`.venv`** first.

**Avoid expecting `uv sync` alone to install FlashAttention** unless you intentionally use **`uv sync --extra gpu`** (pulls **`flash-attn` + onnx + tensorrt**`; TensorRT often fails outside NVIDIA’s intended environments). Prefer the explicit **`uv pip install flash-attn==2.7.4.post1 --no-build-isolation`** line above after **`uv sync`**.

### 1.4 Tokens and secrets

- **Hugging Face:** `export HF_TOKEN=...` if you pull private checkpoints or datasets.
- **Weights & Biases:** `export WANDB_API_KEY=...` only if you pass `--use-wandb` to finetuning.

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

## 3. Example paths, modality layout, and `modality.json`

The rest of this guide uses **one concrete layout** you can copy verbatim or edit. Adjust the variables for your machine.

```bash
REPO_ROOT=/home/users/yufanh/idm/GR00T-H-IDM
DATASET=/home/projects/healthcareeng_monai/datasets/medbot/medbot_1027/train20_lerobot
TEST_DATASET=/home/projects/healthcareeng_monai/datasets/medbot/medbot_1027/test0_40_lerobot
OUTPUT_DIR=/home/users/yufanh/idm/outputs/medbot_gr00t_h_ft
```

- **`DATASET`** — LeRobot root used for **training** (here: 20-episode train split).
- **`TEST_DATASET`** — LeRobot root used for **IDM-style eval** (here: 40-episode test split; name may differ, e.g. `test_lerobot`).
- **`OUTPUT_DIR`** — where finetune checkpoints and `idm_lerobot_eval` summaries will go.

**Medbot layout** (defined in **`open_h/embodiments/medbot/modality.json`**):

- **Video:** `left_endo_image` ← `observation.images.left_endo`
- **State / action:** 20D vectors split into `left_cartesian` (3), `left_rotation` (6), `left_jaw` (1), and the same for the right arm
- **Language:** `task_index` in parquet → text via `meta/tasks.jsonl` (annotation key `task`)

If you already converted data with another Medbot pipeline, **replace** `meta/modality.json` with this repo’s copy when your file used a different language key (here we standardize on **`annotation.task`** for Open-H).

**Install modality into train and test datasets** (copy **from the repo** into each dataset’s `meta/` — not the other way around):

```bash
cd "$REPO_ROOT"

for D in "$DATASET" "$TEST_DATASET"; do
  mkdir -p "$D/meta"
  cp "$REPO_ROOT/open_h/embodiments/medbot/modality.json" "$D/meta/modality.json"
done
```

---

## 4. Prepare normalization statistics

You need **`meta/stats.json`** (raw parquet / feature stats) and **`meta/temporal_stats.json`** (action stats over the action chunk, used at training time). **`open_h/prepare_datasets.sh` does both in one pass** — it is not only `stats.py`:

1. Copies your `modality.json` into `meta/`
2. Runs **`gr00t/data/stats.py`** → writes **`stats.json`**
3. Runs **`gr00t/experiment/launch_finetune.py --calculate-norm-stats`** → writes **`temporal_stats.json`** (inside the script this uses `--base-model-path nvidia/GR00T-N1.6-3B`, which is enough for stats; it does not train)

So after a successful `prepare_datasets.sh` run, you do **not** need a second, separate `launch_finetune.py --calculate-norm-stats` for the same dataset unless you are regenerating temporal stats (e.g. different splits or base checkpoint).

```bash
cd "$REPO_ROOT"

bash open_h/prepare_datasets.sh \
  --embodiment-tag medbot \
  --modality-json open_h/embodiments/medbot/modality.json \
  "$DATASET"

bash open_h/prepare_datasets.sh \
  --embodiment-tag medbot \
  --modality-json open_h/embodiments/medbot/modality.json \
  "$TEST_DATASET"
```

`prepare_datasets.sh` uppercases `--embodiment-tag` before calling Python (so `medbot` and `MEDBOT` both work). If you call `gr00t/data/stats.py` or `launch_finetune.py` directly, use the enum **member** name Tyro expects (e.g. `MEDBOT`).

**Manual / extra runs (optional):**

- You only ran **`stats.py`** by hand → run **`launch_finetune.py --calculate-norm-stats`** once to create **`temporal_stats.json`**.
- You want **`--include-splits`** / **`--exclude-splits`** for stats (not passed by `prepare_datasets.sh` today) → call **`launch_finetune.py --calculate-norm-stats`** yourself with those flags (and optionally `--base-model-path nvidia/GR00T-H` if you prefer that checkpoint for processor loading).
- You want temporal stats keyed off **GR00T-H** explicitly → same manual command with `--base-model-path nvidia/GR00T-H`.

**Train/test splits:** if `meta/info.json` defines `splits`, add `--include-splits train` (or your split name) to **finetuning** and, when computing stats manually, to **`--calculate-norm-stats`**. For eval, use `idm_lerobot_eval.py` with `--include-splits test` (below).

---

## 5. Finetuning (Medbot, LeRobot)

### 5.1 How `torchrun` and `launch_finetune.py` must line up

On **one node**, these must agree (otherwise you get wrong batch math or hung rendezvous):

| What | Rule |
|------|------|
| **`torchrun --nproc_per_node=N`** | **N** = number of GPU processes you launch (usually **one per GPU**). |
| **`--num-gpus N`** (passed to `launch_finetune.py`) | **Must equal** `nproc_per_node` for single-node jobs. It is stored as `config.training.num_gpus` and used to compute **per-device** batch size. |
| **`--global-batch-size B`** | **B must be divisible by N** (`assert` in `gr00t/experiment/experiment.py`). Example: `N=8` → `B=32` works; `N=7` → use `B=224` (32×7), not `B=32`. |
| **Working directory** | Run from **repository root** (where `gr00t/` exists) so `gr00t/experiment/launch_finetune.py` resolves. |

FlashAttention is unrelated to the above: it depends on **`flash_attn`** in the venv and Hub **`use_flash_attention`** (see **§1.3**).

**Video decoding:** the stack defaults to **`torchcodec`**. If it is not installed, you will see a fallback warning. Either install **`torchcodec`** in **`.venv`**, or pass **`--video-backend ffmpeg`** (needs **`ffmpeg`** / **`ffprobe`** on `PATH`) or **`--video-backend decord`** on `launch_finetune.py`.

### 5.2 Example command (8 GPUs)

```bash
cd "$REPO_ROOT"
mkdir -p "$OUTPUT_DIR"

NUM_GPUS=8
uv run torchrun --nproc_per_node="$NUM_GPUS" --master_port=29500 \
  gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-H \
  --dataset-path "$DATASET" \
  --embodiment-tag MEDBOT \
  --num-gpus "$NUM_GPUS" \
  --global-batch-size 32 \
  --max-steps 20000 \
  --learning-rate 1e-4 \
  --output-dir "$OUTPUT_DIR" \
  --include-splits train
```

Use the **same** `NUM_GPUS` for both `torchrun` and `--num-gpus`. Change **`--global-batch-size`** whenever you change **`NUM_GPUS`** so it stays divisible.

### 5.3 `--include-splits train`

Only add **`--include-splits train`** if **`meta/info.json`** defines a **`splits`** field that includes a split named `train`. If your dataset has **no** split metadata (common when you already exported a train-only folder), **remove** this flag or finetuning / split resolution can error.

### 5.4 Notes

- **`--embodiment-tag MEDBOT`** uses `open_h/embodiments/medbot/medbot_config.py` (no `--modality-config-path` needed).
- Add **`--video-backend ffmpeg`** (or **`decord`**) if **`torchcodec`** is missing and you want to avoid noisy fallback warnings.
- Tune **`--global-batch-size`**, **`--max-steps`**, **`--save-steps`**, **`--dataloader-num-workers`** for your cluster.
- For **vision-only** style training, see `open_h/gr00t_h_config.yaml` and `state_dropout_prob_per_embodiment` in upstream docs.

Checkpoints appear under **`--output-dir`** as **`checkpoint-<step>/`**.

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
cd "$REPO_ROOT"

uv run python gr00t/eval/idm_lerobot_eval.py \
  --checkpoint "$OUTPUT_DIR/checkpoint-20000" \
  --dataset "$TEST_DATASET" \
  --output-dir "$OUTPUT_DIR/idm_eval_test" \
  --embodiment-tag MEDBOT \
  --include-splits test \
  --action-horizon 16 \
  --no-plots
```

Replace **`checkpoint-20000`** with the step folder you actually saved (e.g. `checkpoint-10000`).

Outputs:

- **`$OUTPUT_DIR/idm_eval_test/inference_summary.json`** — average and per-episode **MSE** / **MAE** in **unnormalized dataset space** (same convention as `gr00t/eval/open_loop_eval.py`).
- Omit **`--include-splits test`** if `TEST_DATASET` has no split metadata (e.g. a dedicated test-only folder); then all episodes are evaluated unless you pass **`--max-episodes`** or **`--traj-ids`**.
- Without **`--no-plots`**, saves one JPEG per trajectory under **`--output-dir`**.

**Evaluate all episodes** in a test-only folder (no `splits` in `info.json`):

```bash
cd "$REPO_ROOT"

uv run python gr00t/eval/idm_lerobot_eval.py \
  --checkpoint "$OUTPUT_DIR/checkpoint-20000" \
  --dataset "$TEST_DATASET" \
  --output-dir "$OUTPUT_DIR/idm_eval_sample" \
  --embodiment-tag MEDBOT \
  --max-episodes 50
```

**Explicit trajectory IDs:**

```bash
cd "$REPO_ROOT"

uv run python gr00t/eval/idm_lerobot_eval.py \
  --checkpoint "$OUTPUT_DIR/checkpoint-20000" \
  --dataset "$TEST_DATASET" \
  --output-dir "$OUTPUT_DIR/idm_eval_traj_subset" \
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

- **Wrong Python / missing `gr00t`** — Run `uv run python -c "import sys; print(sys.executable)"` from the **repo root**; expect `.../.venv/bin/python`. Re-run **`uv sync --python 3.10`** (that installs the repo in editable form). Avoid mixing **Conda `python`** with **`uv run`** unless you know which env you are using.
- **`ImportError: ... flash_attn` / “flash_attn seems to be not installed”** — **`flash-attn` is not installed by default** (**§1**). Install it with **`uv pip install flash-attn==2.7.4.post1 --no-build-isolation`** from the **repo root** so it lands in **`.venv`**, then confirm with **`uv run python -c "import flash_attn"`**. If you installed only into **Conda** but run training with **`uv run`**, see the **“Common mistake”** note in **§1.3**.
- **`flash-attn` install fails** — Needs **Linux + CUDA + nvcc**, and **torch already installed** in the same `.venv` (`--no-build-isolation`). On **macOS** or **CPU-only**, skip it. If the build fails with **undefined symbols** when *importing* `flash_attn`, search flash-attention issues for your **torch + CUDA** combo; you may need a different flash-attn / torch pairing.
- **`Embodiment 'medbot' not found in MODALITY_CONFIGS`** — ensure `import open_h.embodiments` runs before training ( `launch_finetune.py` already imports it).
- **Missing `stats.json` / `temporal_stats.json`** — run `prepare_datasets.sh` and `launch_finetune.py --calculate-norm-stats` for that dataset path.
- **`invalid choice: 'medbot'` from `stats.py` / `launch_finetune.py`** — Tyro expects the enum member name (`MEDBOT`). Use `prepare_datasets.sh` (it uppercases the tag) or pass `--embodiment-tag MEDBOT` on the CLI.
- **Language / task errors** — `modality.json` must define `annotation.task` → `task_index` as in this repo’s `medbot/modality.json`.
- **Video backend** — finetuning defaults to `torchcodec` in `DataConfig`. If you see *“torchcodec is not available, falling back …”*, install **`torchcodec`** in **`.venv`** or run **`launch_finetune.py` with `--video-backend ffmpeg`** or **`--video-backend decord`**. For **`idm_lerobot_eval.py`**, use **`--video-backend`** there as well if needed.

For the original **GR00T-Dreams** IDM (SigLIP, separate checkpoint family), use the **GR00T-Dreams** repository and its Medbot / `idm_inference_simple.py` scripts (same ideas as the surgical synthetic-data tutorial).
