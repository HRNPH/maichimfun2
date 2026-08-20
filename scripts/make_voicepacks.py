#!/usr/bin/env python3
"""Blend the Thai Kokoro voicepack (thai_som) with upstream Kokoro-82M voicepacks
to synthesize multiple distinct Thai-capable speakers.

FastThaiG2P ships exactly one Thai voice (thai_som, [511, 1, 256] style vectors
indexed by phoneme-count). Upstream Kokoro voicepacks live in the same style
space (the Thai model was fine-tuned from Kokoro), so a convex combination
`alpha * thai_som + (1 - alpha) * upstream` keeps Thai intelligibility of som
while shifting timbre/gender/pitch toward the upstream speaker.

Usage: .venv/bin/python scripts/make_voicepacks.py [--alpha 0.75]
"""
import argparse
import urllib.request
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".cache" / "fastthaig2p"
VOICE_DIR = Path(__file__).resolve().parent.parent / "voicepacks"
HF_BASE = "https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main/voices"

# name -> (upstream voice file, blend weight for thai_som)
# alphas >= 0.7 keep Thai segmental quality; upstream contributes timbre.
SPEAKERS = {
    "th_som":      (None,          1.00),  # original female Thai voice
    "th_aara":     ("af_heart",    0.75),  # warmer female
    "th_bua":      ("af_bella",    0.75),  # bright female
    "th_chaniya":  ("if_sara",     0.70),  # soft female
    "th_dara":     ("bf_emma",     0.75),  # british-leaning female
    "th_ekkachai": ("am_adam",     0.75),  # male
    "th_fah":      ("af_nicole",   0.80),  # quiet female
    "th_git":      ("bm_george",   0.70),  # deep male
    "th_hatai":    ("im_nicola",   0.75),  # female
    "ithima":      ("am_michael",  0.75),  # male
}


def load_voicepack(path: Path) -> np.ndarray:
    try:
        arr = np.load(path)
    except ValueError:
        raw = np.fromfile(path, dtype=np.float32)
        arr = raw.reshape(-1, 1, 256)
    # thai_som is (510, 1, 256); upstream Kokoro packs ship (511, 1, 256).
    # The engine indexes voice[len(phonemes) - 1] with phonemes <= 510, so
    # 510 rows is what the model can ever touch — align everything to that.
    arr = arr[:510]
    assert arr.shape == (510, 1, 256), f"{path}: unexpected shape {arr.shape}"
    return arr.astype(np.float32)


def fetch_upstream(name: str) -> Path:
    dest = VOICE_DIR / f"{name}.bin"
    if dest.exists():
        return dest
    url = f"{HF_BASE}/{name}.bin"
    print(f"downloading {url}")
    tmp = dest.with_suffix(".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=None,
                    help="override the per-speaker thai weight for all blends")
    args = ap.parse_args()

    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    thai_path = CACHE / "thai_som.npy"
    if not thai_path.exists():
        raise SystemExit(
            f"{thai_path} not found — run a TTS() once first so the "
            "default model + voicepack download to the cache."
        )
    thai = load_voicepack(thai_path)

    for out_name, (upstream, alpha) in SPEAKERS.items():
        if args.alpha is not None and upstream is not None:
            alpha = args.alpha
        out = VOICE_DIR / f"{out_name}.npy"
        if upstream is None:
            blended = thai.copy()
        else:
            up = load_voicepack(fetch_upstream(upstream))
            blended = alpha * thai + (1.0 - alpha) * up
        np.save(out, blended.astype(np.float32))
        print(f"{out_name}:  {out.name}  (thai={alpha:.2f}, upstream={upstream})")
    print(f"\n{len(SPEAKERS)} voicepacks in {VOICE_DIR}")


if __name__ == "__main__":
    main()
