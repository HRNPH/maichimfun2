#!/usr/bin/env bash
# Fine-tune Needle 2 on the Thai tool-calling dataset (run on the GPU node).
#
# Usage:
#   bash scripts/finetune.sh                 # CUDA GPU node (default)
#   ACCEL=metal bash scripts/finetune.sh    # Apple Silicon
#
# Requires on the node: this repo's pyproject.toml + uv.lock + scripts/ +
# data/dataset/. Env overrides: EPOCHS, LORA_RANK, LORA_ALPHA, LR, BATCH,
# MAX_LEN, BITS, ACCEL.
#
# Hyperparameter rationale (from needle doc/finetuning.md):
#   - ~900 examples is a small set → 10-30 epochs (default here: 25)
#   - argument-grounding-heavy task → --lora-rank 32 ("doubled adapter capacity")
#   - lora-alpha kept at 2x rank (same ratio as the 32/16 default)
#   - val-split left at needle's default 0.1 (we concatenate our train+val
#     and let needle re-split; a fixed external val file isn't in the CLI)
set -euo pipefail
cd "$(dirname "$0")/.."

ACCEL="${ACCEL:-gpu}"                  # gpu | metal (dependency group in pyproject.toml)
EPOCHS="${EPOCHS:-25}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LR="${LR:-1e-4}"
BATCH="${BATCH:-16}"
MAX_LEN="${MAX_LEN:-1024}"
BITS="${BITS:-2}"                      # export quantization (2 matches the shipped 14MB .cact)
DATA_ALL="data/dataset/needle_thai_all.jsonl"
BASE="checkpoints/needle2.pkl"
ADAPTER="checkpoints/needle_lora.pkl"
OUT_CACT="out/needle_thai.cact"

mkdir -p out logs

# JAX preallocates most of the device by default; on unified memory (metal)
# that can eat all system RAM, and on shared CUDA nodes it hogs the GPU.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# --- 1. environment (uv + dependency group; base weights auto-download in step 3)
if ! command -v uv >/dev/null 2>&1; then
    echo "== installing uv =="
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "== uv sync --group ${ACCEL} =="
uv sync --group "${ACCEL}"

# --- 2. data ------------------------------------------------------------------
# needle finetune takes ONE file and carves its own 10% val split, so fold our
# pre-split val back in rather than wasting it.
echo "== preparing ${DATA_ALL} =="
cat data/dataset/needle_thai_train.jsonl data/dataset/needle_thai_val.jsonl > "${DATA_ALL}"
wc -l "${DATA_ALL}"

# --- 3. train (downloads ${BASE} from Hugging Face on first run) ---------------
echo "== fine-tuning (epochs=${EPOCHS} rank=${LORA_RANK} alpha=${LORA_ALPHA} lr=${LR}) =="
.venv/bin/needle finetune "${DATA_ALL}" \
    --epochs "${EPOCHS}" \
    --lora-rank "${LORA_RANK}" \
    --lora-alpha "${LORA_ALPHA}" \
    --lr "${LR}" \
    --batch-size "${BATCH}" \
    --max-len "${MAX_LEN}" \
    2>&1 | tee "logs/finetune_$(date +%Y%m%d_%H%M%S).log"

[ -f "${ADAPTER}" ] || { echo "adapter ${ADAPTER} not produced — check the log above"; exit 1; }

# --- 4. build the .cact (merge LoRA into the frozen base) ----------------------
[ -f "${BASE}" ] || { echo "base ${BASE} missing after training — unexpected"; exit 1; }
echo "== building ${OUT_CACT} (bits=${BITS}) =="
.venv/bin/needle build "${BASE}" \
    --lora "${ADAPTER}" \
    --out "${OUT_CACT}" \
    --bits "${BITS}"

ls -lh "${OUT_CACT}"

# --- 5. sanity eval -----------------------------------------------------------
echo "== running eval sample =="
.venv/bin/python scripts/eval_thai.py --weights "${OUT_CACT}" | tee "logs/eval_$(date +%Y%m%d_%H%M%S).log"

echo "DONE — tuned model at ${OUT_CACT}"
echo "Use it with: needle.Needle(weights=\"${OUT_CACT}\")"
