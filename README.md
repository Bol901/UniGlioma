# UniGlioma

**A Generative Foundation Model for Instruction-Guided Processing and Controlled Generation of Brain Tumor MRI**

UniGlioma uses a shared 3D diffusion transformer for instruction-conditioned MRI restoration, completion, and segmentation. Co-registered references are fused at corresponding spatial locations before global attention. The current configuration accepts zero to four reference slots.

## Code-only release

**Production model weights are not publicly distributed because of potential patient-information disclosure risks.** This repository contains source code, generic task instructions, example configurations, fabricated schema examples, and synthetic tests. It contains no patient images, reports, data manifests, cached features, training outputs, or production checkpoints. There are no pretrained-weight download links or automatic model downloads.

Following publication, model weights may be made available for research use upon request, subject to institutional approval and execution of a Material Transfer Agreement (MTA). Researchers interested in obtaining access should contact the authors; contact details will be added following publication.

Full training requires your own authorized data, a compatible pretrained 3D VAE, and a local text encoder. Inference additionally requires a DiT checkpoint that you trained or are authorized to use. The VAE architecture is included; VAE pretraining is outside this repository. The example configuration is a starting point, not an archived configuration that reproduces the paper's private checkpoint. The generic segmentation prompt catalog is newly authored for this release; rebuild text caches before training with it.

## Install and check the source

Use Python 3.10 or later and install a CUDA-compatible PyTorch build for GPU work, then:

```bash
pip install -r requirements-dev.txt
export PYTHONDONTWRITEBYTECODE=1
python -m pytest -q -p no:cacheprovider tests
python train.py --help
python infer.py --help
python scripts/check_release.py
```

The tests use tiny randomly initialized models and fabricated arrays on CPU. They do not require MRI data, pretrained weights, a text encoder, or a GPU. Dependencies were checked locally with Python 3.12, PyTorch 2.13, NumPy 2.0, SciPy 1.15, h5py 3.16, and nibabel 5.4; this is not a complete compatibility matrix. Transformer loading and full GPU training were not exercised for the source release.

## Task interface

| Task | Input | Output |
|---|---|---|
| `sr` | One low-resolution MRI | Restored MRI |
| `deblur` | One MRI with blur | Restored MRI |
| `dealias` | One MRI with aliasing | Restored MRI |
| `motion` | One MRI with motion artifacts | Restored MRI |
| `inpaint` | MRI and region defined from a label volume | Completed MRI |
| `missing` | Available modalities, excluding the target | Target MRI modality |
| `seg` | One to four MRI modalities | Generated tumor label volume |

`modality_only` is available for modality-conditioned pretraining. The core also retains `mask_guide` for mask-conditioned generation and `whole_brain` for field-of-view completion; these are not enabled in the default training mix. Text/report-to-image and longitudinal prediction tasks are not part of this release.

The shared slot budget includes spatial mask conditions. Zero-reference generation is not zero-input diagnostic segmentation. `seg` produces a generated label map; the optional auxiliary segmentation head is a separate readout and is not an independent evaluator of generated MRI.

## Prepare caller-owned assets

Run commands from this repository root. Keep all local data and generated prompt caches under the ignored `local_data/` directory or outside the repository. See [the data contract](docs/DATA_FORMAT.md) before creating inputs.

```bash
export HF_HOME=/working/huggingface_cache
export UNIGLIOMA_TEXT_MODEL="$HF_HOME/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/<revision>"
export UNIGLIOMA_VAE_CHECKPOINT="$HF_HOME/hub/<your-compatible-vae>/model.pt"
```

These paths must already exist; the repository does not fetch weights. The inherited text-cache fingerprint expects an unsharded `model.safetensors` file in the encoder snapshot. It uses file size and encoding settings, not a cryptographic identity of the weights, so do not reuse caches across different encoder weights even when their size matches. A container should mount the same host cache and set its `HF_HOME` to the mount location.

Create `local_data/train.json`, `local_data/val.json`, and `local_data/acquisition.json` using the documented schema, with patient-disjoint splits. The files in `examples/` contain invented paths and are not a runnable dataset.

```bash
# Collect caller-owned image paths, including full-head siblings used by artifact tasks.
python scripts/list_volumes.py \
  --records local_data/train.json local_data/val.json \
  --include-full-head --out local_data/volumes.json

# These two commands write latent caches into the caller-owned H5 files.
python scripts/precompute_latents.py --config configs/train.yaml \
  --json local_data/volumes.json --modes iso1mm --device cuda:0
python scripts/precompute_mask_latents.py --config configs/train.yaml \
  --json local_data/train.json local_data/val.json --device cuda:0

# Only generic instructions are enumerated; no clinical notes are used.
python scripts/build_prompts.py --out local_data/prompts.json
python scripts/encode_prompts.py --prompts local_data/prompts.json \
  --qwen3vl "$UNIGLIOMA_TEXT_MODEL" --out local_data/prompt_features \
  --max-length 256
```

The autoencoder encodes image and label representations; regenerating a cache after changing the VAE is mandatory. Encoding scripts write only to the user-specified data/cache locations. For label caches, remap classes explicitly before encoding: raw dataset labels must not be silently clipped into the configured class range.

## Train the DiT

```bash
# Optional modality-conditioned stage; then initialize the full stage from its checkpoint.
torchrun --standalone --nproc_per_node=1 train.py \
  --config configs/train.yaml --curriculum-preset foundation

torchrun --standalone --nproc_per_node=1 train.py \
  --config configs/train.yaml --curriculum-preset full \
  --init-checkpoint /path/to/your/pretraining-checkpoint.pt
```

Omit `--init-checkpoint` to initialize the DiT randomly. Increase `--nproc_per_node` to use more GPUs; configure `parallel: fsdp` and the `fsdp` block when appropriate. The 2B example needs substantial GPU memory even with a batch size of one; no hardware-fit claim is made. SDPA is the default attention implementation. Flash attention is optional. Training uses flow matching, structured conditioning dropout, EMA, and optional auxiliary segmentation supervision.

Checkpoints and resolved run configurations default to `${HF_HOME}/hub/uniglioma-training/`. Resume a caller-owned run with:

```bash
torchrun --standalone --nproc_per_node=1 train.py --config configs/train.yaml \
  --resume latest --resume-exp /path/to/your/run
```

W&B and periodic image validation are disabled by default. Training logs, evaluation outputs, checkpoints, resolved configurations, and input path sidecars may carry sensitive local information; they are runtime artifacts and are not release contents.

## Inference

```bash
export UNIGLIOMA_DIT_CHECKPOINT="$HF_HOME/hub/<your-uniglioma-run>/ckpts/model.pt"

# Actual thick-slice input, already prepared on the required H5 grid.
python infer.py --config configs/infer.yaml --task sr --native-input \
  --ref-image local_data/h5/example/scan_fla_bet.h5 --target-modality 3

# Synthesize a withheld FLAIR modality from available T1/T1c/T2.
python infer.py --config configs/infer.yaml --task missing \
  --ref-image local_data/h5/example/scan_t1_bet.h5 \
              local_data/h5/example/scan_t1c_bet.h5 \
              local_data/h5/example/scan_t2_bet.h5 \
  --ref-modality 0 1 2 --target-modality 3

python infer.py --config configs/infer.yaml --task seg \
  --ref-image local_data/h5/example/scan_t1_bet.h5 \
              local_data/h5/example/scan_fla_bet.h5 \
  --ref-modality 0 3
```

Pass `--prompt "your complete task instruction"` to override the task's default instruction. A prompt does not change the required task-specific input preparation. Outputs are written to ignored `outputs/infer/` by default.

**Artifact-task input semantics:** the inherited `deblur`, `dealias`, and `motion` command paths generate synthetic corruption from the supplied source before restoration. They are evaluation paths, not a raw clinical artifact-removal interface. `sr --native-input` bypasses synthetic SR degradation; ordinary `sr` applies a synthetic degradation. `inpaint` constructs its missing region from the provided label. See the data contract for full-head siblings, masks, and alignment. Do not interpret these synthetic tests as clinical validation.

## Repository layout

```text
train.py / infer.py       training and inference entry points
models/                  DiT, aligned reference fusion, VAE, local text encoder, FSDP
data/                    H5 loading, task construction, collation
utils/                   prompts, losses/support code, geometry and degradation helpers
configs/                 portable training and inference examples
assets/                  generic segmentation prompt catalog
scripts/                 cache preparation and source-release checks
examples/                invented schema examples
tests/                   CPU synthetic checks
docs/DATA_FORMAT.md       input and cache contract
```

Before publication, run `python scripts/check_release.py` and upload only this repository directory. The check screens file types, symlinks, obvious private paths and credential patterns; it cannot establish that arbitrary later user-added text or identifiers are non-sensitive. A code license and final anonymous repository URL are to be specified by the authors. No license for third-party model weights is granted by this source release.
