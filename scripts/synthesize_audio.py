#!/usr/bin/env python3
"""Synthesize multi-speaker Thai speech for every query in the raw JSONL shards.

Pipeline: query -> (Latin words transliterated to Thai) -> FastThaiG2P G2P
-> Kokoro-82M Thai ONNX -> 24 kHz mono WAV, one file per example.

Speaker hack: FastThaiG2P's TTS fixes the voice at construction
(`ref_s = self._voice[len(phonemes) - 1]`), but that voice is just a
[511, 1, 256] numpy style table. We keep ONE model session and hot-swap
`tts._voice` with the blended voicepacks from make_voicepacks.py, plus a
deterministic ±10% resample jitter per line for prosody variety.

Resume-safe: existing WAVs are skipped; the manifest is rewritten whole.

Usage: .venv/bin/python scripts/synthesize_audio.py [--limit N] [--shard NAME]
"""
import argparse
import hashlib
import json
import re
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
WAV_DIR = ROOT / "data" / "audio" / "wavs"
MANIFEST = ROOT / "data" / "audio" / "manifest.jsonl"
VOICE_DIR = ROOT / "voicepacks"
SR = 24000
MAX_PHONEMES = 510

LATIN_WORDS = {
    "goodnight": "กูดไนท์", "movie": "มูฟวี่", "morning": "มอร์นิง",
    "away": "อะเวย์", "dinner": "ดินเนอร์", "planning": "แพลนนิ่ง",
    "spotify": "สปอติฟาย", "youtube": "ยูทูบ", "netflix": "เน็ตฟลิกซ์",
    "wifi": "ไวไฟ", "wi-fi": "ไวไฟ", "online": "ออนไลน์", "cctv": "ซีซีทีวี",
    "radio": "เรดิโอ", "podcast": "พอดแคสต์", "playlist": "เพลย์ลิสต์",
    "music": "มิวสิก", "news": "นิวส์", "line": "ไลน์", "email": "อีเมล",
    "message": "เมสเสจ", "video": "วิดีโอ", "call": "คอล", "app": "แอป",
    "tv": "ทีวี", "kitchen": "คิเชน", "quick": "ควิก", "normal": "นอร์มัล",
    "heavy": "เฮฟวี่", "mode": "โหมด", "auto": "ออโต้", "cool": "คูล",
    "dry": "ไดร์", "fan": "แฟน", "dock": "ดอก", "pause": "พอส",
    "start": "สตาร์ท", "stop": "สต็อป", "play": "เพลย์", "mute": "มิวท์",
    "en": "อังกฤษ", "ja": "ญี่ปุ่น", "zh": "จีน", "ko": "เกาหลี",
    "kg": "กิโลกรัม", "km": "กิโลเมตร", "mile": "ไมล์", "thb": "บาท",
    "usd": "ดอลลาร์", "jpy": "เยน", "eur": "ยูโร",
    "taylor swift": "เทย์เลอร์ สวิฟท์", "bts": "บีทีเอส",
    "blackpink": "แบล็กพิงก์", "one ok rock": "วัน โอเค ร็อก",
}
LETTERS = {
    "a": "เอ", "b": "บี", "c": "ซี", "d": "ดี", "e": "อี", "f": "เอฟ",
    "g": "จี", "h": "เอช", "i": "ไอ", "j": "เจ", "k": "เค", "l": "แอล",
    "m": "เอ็ม", "n": "เอ็น", "o": "โอ", "p": "พี", "q": "คิว", "r": "แอร์",
    "s": "เอส", "t": "ที", "u": "ยู", "v": "วี", "w": "ดับเบิลยู",
    "x": "เอกซ์", "y": "วาย", "z": "ซี",
}
DIGITS = ["ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า"]

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*|\d+")


def latin_to_thai(text: str) -> str:
    """Rewrite Latin tokens/digits to Thai so the Thai G2P can phonemize them.
    The original text stays untouched in the dataset — this is TTS-input only."""
    def repl(m: re.Match) -> str:
        tok = m.group(0)
        low = tok.lower()
        if low in LATIN_WORDS:
            return LATIN_WORDS[low]
        if tok.isdigit():
            return " ".join(DIGITS[int(d)] for d in tok)
        return " ".join(
            DIGITS[int(c)] if c.isdigit() else LETTERS.get(c, "")
            for c in low if c.isalnum()
        ).strip()
    return TOKEN_RE.sub(repl, text)


def resample_speed(audio: np.ndarray, factor: float) -> np.ndarray:
    idx = np.arange(0, len(audio) - 1, factor)
    return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)


def split_for_phonemes(text: str, g2p) -> list[str]:
    """Split long text so each chunk stays under the 510-phoneme model limit."""
    if len(g2p.convert(text)) <= MAX_PHONEMES:
        return [text]
    parts, cur = [], ""
    for w in text.split():
        cand = (cur + " " + w).strip()
        if cur and len(g2p.convert(cand)) > MAX_PHONEMES:
            parts.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        parts.append(cur)
    return parts


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only synthesize N lines")
    ap.add_argument("--shard", type=str, default="", help="only this shard name")
    args = ap.parse_args()

    from fastthaig2p import TTS

    speakers = sorted(p.stem for p in VOICE_DIR.glob("*.npy"))
    if not speakers:
        raise SystemExit("no voicepacks — run scripts/make_voicepacks.py first")
    print(f"{len(speakers)} speakers: {', '.join(speakers)}")

    tts = TTS()  # default model + thai_som; session reused across speakers
    voices = {s: np.load(VOICE_DIR / f"{s}.npy") for s in speakers}

    def set_speaker(name: str) -> None:
        tts._voice = voices[name]

    # smoke-test voice swap on the default instance
    set_speaker(speakers[0])

    g2p = tts.g2p if hasattr(tts, "g2p") else tts._g2p

    WAV_DIR.mkdir(parents=True, exist_ok=True)
    manifest, failures = [], []
    items = []
    for shard in sorted(RAW_DIR.glob("*.jsonl")):
        if args.shard and shard.stem != args.shard:
            continue
        for i, line in enumerate(shard.read_text(encoding="utf-8").splitlines()):
            if line.strip():
                items.append((shard.stem, i, json.loads(line)))
    if args.limit:
        items = items[: args.limit]
    if not items:
        raise SystemExit(
            "no raw shards in data/raw/ — nothing to synthesize; "
            "manifest left untouched"
        )
    print(f"{len(items)} examples to synthesize")

    for n, (shard, idx, ex) in enumerate(items, 1):
        stem = f"{shard}__{idx:04d}"
        h = int(hashlib.md5(ex["query"].encode()).hexdigest(), 16)
        speaker = speakers[h % len(speakers)]
        speed = 0.90 + (h >> 8) % 21 / 100.0
        out_path = WAV_DIR / f"{stem}__{speaker}.wav"
        if out_path.exists():
            manifest.append((out_path, speaker, ex))
            continue
        tts_text = latin_to_thai(ex["query"])
        try:
            set_speaker(speaker)
            chunks = split_for_phonemes(tts_text, g2p)
            audio = np.concatenate([tts.generate(c) for c in chunks])
            audio = resample_speed(audio, speed)
            audio = np.clip(audio, -1.0, 1.0)
            with wave.open(str(out_path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes((audio * 32767).astype(np.int16).tobytes())
            manifest.append((out_path, speaker, ex))
        except Exception as e:  # noqa: BLE001 — keep going, report at end
            failures.append((stem, repr(e)))
        if n % 25 == 0:
            print(f"  {n}/{len(items)} done", flush=True)

    with open(MANIFEST, "w", encoding="utf-8") as f:
        for path, speaker, ex in manifest:
            rec = {
                "id": path.stem,
                "audio": str(path.relative_to(ROOT)),
                "duration": round(wav_duration(path), 3),
                "speaker": speaker,
                "text": ex["query"],
                "tools": ex["tools"],
                "answers": ex["answers"],
                "reasoning": ex["reasoning"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nmanifest: {MANIFEST} ({len(manifest)} entries)")
    if failures:
        print(f"{len(failures)} FAILURES:")
        for stem, err in failures[:20]:
            print(f"  {stem}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
    # onnxruntime's macOS teardown races its thread pool and aborts after
    # everything is already flushed; skip interpreter shutdown entirely.
    import os
    os._exit(0)
