# maichimfun2 — Thai synthetic data for Needle 2

Synthetic Thai fine-tuning data for [Needle 2](https://github.com/cactus-compute/needle)
(45M-param tool-calling model): text tool-call examples in the exact
`needle finetune` JSONL format, plus parallel multi-speaker Thai speech
generated with FastThaiG2P + Kokoro.

## Layout

```
data/raw/*.jsonl           10 domain shards × 90 lines (subagent-generated)
voicepacks/*.npy           10 blended Thai speaker voicepacks
data/audio/wavs/           one 24 kHz WAV per query line
data/audio/manifest.jsonl  audio ↔ example mapping (speaker, duration, text)
data/dataset/
  needle_thai_{train,val}.jsonl       text dataset (needle finetune input)
  needle_thai_asr_{train,val}.jsonl   speech dataset (audio + needle fields)
  dataset_stats.json
scripts/
  make_voicepacks.py       blend thai_som with upstream Kokoro voices
  synthesize_audio.py      text → multi-speaker Thai speech + manifest
  build_dataset.py         validate + dedupe + shuffle + split + aggregate
```

## Data format (per Needle's doc/finetuning.md)

One JSON object per line: `query` (Thai utterance), `tools` (JSON-Schema
catalogue), `answers` (`[{name, arguments}]`, `[]` for off-topic), optional
`reasoning` (arg-derivation line — loss is computed on it + the call).
Values are grounded in the query; no placeholders. Off-topic rate ≈ 18 %
(doc suggests ≥ 1/8).

Domains: smart-home lights/AC, security, appliances/scenes, music/media,
weather/alarms/timers, calendar/comms, structured extraction, utility tools,
ambiguous-resolution, chitchat/rejection.

## Pipeline

Dependencies are managed with uv (`pyproject.toml` + `uv.lock`):
`fastthaig2p[tts]` is a main dep; `cactus-needle[metal]` / `[gpu]` live in
the `metal` / `gpu` dependency groups (gpu is linux-gated).

```bash
uv sync --group metal          # this machine (generation pipeline)
# uv sync --group gpu         # on the training node

.venv/bin/python scripts/make_voicepacks.py      # 10 speakers
.venv/bin/python scripts/synthesize_audio.py     # ~900 WAVs (resume-safe)
.venv/bin/python scripts/build_dataset.py        # unified datasets
```

### Multi-speaker hack

FastThaiG2P ships one Thai voice (`thai_som`, a `[510, 1, 256]` style table
indexed by phoneme count; voice fixed at construction). The Thai model was
fine-tuned from Kokoro-82M, so its style space is compatible with upstream
Kokoro voicepacks (`[511, 1, 256]`, trimmed to 510). `make_voicepacks.py`
blends `alpha · thai_som + (1 − alpha) · upstream` (alpha 0.70–0.80, from
`onnx-community/Kokoro-82M-v1.0-ONNX`) into 9 new speakers; keeping alpha ≥
0.7 preserves Thai intelligibility while shifting timbre/gender/pitch.
`synthesize_audio.py` keeps one ONNX session and hot-swaps `tts._voice`
per line, with deterministic ±10 % resample speed jitter.

Latin tokens in queries (goodnight, Taylor Swift, AB-4521, …) are
transliterated to Thai **for TTS input only** (word dict + letter/digit
spelling); dataset text keeps the originals.

## Fine-tuning

On the GPU node (copy the project over — only `data/dataset/` + `scripts/`
are needed, ~2 MB):

```bash
bash scripts/finetune.sh                # CUDA node
ACCEL=metal bash scripts/finetune.sh    # Apple Silicon
```

The script runs `uv sync --group gpu|metal`, concatenates train+val
(`needle finetune` carves its own 10 % split from a single file), trains
LoRA (rank 32 / alpha 64 — grounding-heavy task per the docs, 25 epochs),
merges the adapter into a 2-bit `out/needle_thai.cact`, then runs
`scripts/eval_thai.py` on val samples. Tune via env: `EPOCHS LORA_RANK
LORA_ALPHA LR BATCH MAX_LEN BITS`. `XLA_PYTHON_CLIENT_PREALLOCATE=false`
is set by the script (jax-metal preallocation can otherwise consume all
unified memory on 16 GB Macs).

Note: ASR variants carry `audio` paths for speech pipelines (e.g. Whisper
transcription → Needle); `needle finetune` itself consumes the text files.
