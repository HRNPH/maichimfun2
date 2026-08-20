#!/usr/bin/env python3
"""Validate, dedupe, shuffle and aggregate all raw shards into the unified
Needle-2 fine-tuning dataset (text) and the speech dataset (audio + manifest).

Outputs, under data/dataset/:
  needle_thai_train.jsonl      — text-only, exact Needle format
                                  (query/tools/answers/reasoning)
  needle_thai_val.jsonl        — 10 % held out (matches `needle finetune`
                                  default --val-split 0.1)
  needle_thai_asr_train.jsonl  — same split, speech version: audio path,
                                  duration, speaker + the needle fields
  dataset_stats.json            — counts per shard/speaker/tool

Usage: .venv/bin/python scripts/build_dataset.py [--val-split 0.1]
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "audio" / "manifest.jsonl"
OUT_DIR = ROOT / "data" / "dataset"

# Rough ceiling: Needle's --max-len 1024 tokens ≈ 3 bytes/token of rendered
# text; queries this long would be silently truncated, so reject instead.
MAX_QUERY_CHARS = 2200


def validate(ex: dict, src: str) -> list[str]:
    errs = []
    for k in ("query", "tools", "answers", "reasoning"):
        if k not in ex:
            errs.append(f"missing field {k}")
    if errs:
        return errs
    if not ex["query"].strip():
        errs.append("empty query")
    if not isinstance(ex["tools"], list) or not ex["tools"]:
        errs.append("no tools")
    names = set()
    for t in ex["tools"]:
        if "name" not in t or "parameters" not in t:
            errs.append(f"bad tool shape: {t}")
            continue
        if t["name"] in names:
            errs.append(f"duplicate tool {t['name']}")
        names.add(t["name"])
    if len(ex["query"]) > MAX_QUERY_CHARS:
        errs.append(f"query too long ({len(ex['query'])} chars)")
    for a in ex["answers"]:
        if a.get("name") not in names:
            errs.append(f"answer tool {a.get('name')} not in catalogue")
        for k, v in a.get("arguments", {}).items():
            if isinstance(v, str) and not v.strip():
                errs.append(f"empty-string arg {k} in {a.get('name')}")
        params = next((t["parameters"] for t in ex["tools"]
                       if t["name"] == a.get("name")), {})
        for req in params.get("required", []):
            if req not in a.get("arguments", {}):
                errs.append(f"{a.get('name')}: missing required arg {req}")
    return [f"{src}: {e}" for e in errs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    examples, errors = [], []
    seen, dupes = set(), 0
    for shard in sorted(RAW_DIR.glob("*.jsonl")):
        for i, line in enumerate(shard.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"{shard.stem}:{i}: JSON error {e}")
                continue
            errors += validate(ex, f"{shard.stem}:{i}")
            key = (ex["query"], json.dumps(ex.get("answers", []), sort_keys=True))
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            ex["_shard"] = shard.stem
            examples.append(ex)

    if not examples:
        raise SystemExit(
            "no examples found in data/raw/ (missing or empty?) — refusing to "
            "overwrite the existing dataset in data/dataset/ with empty files"
        )

    if errors:
        print(f"{len(errors)} validation problems — inspect and fix:")
        for e in errors[:30]:
            print(" ", e)
        raise SystemExit(1)

    audio = {}
    if MANIFEST.exists():
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            audio[rec["text"]] = rec

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n_val = round(len(examples) * args.val_split)
    val, train = examples[:n_val], examples[n_val:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", train), ("val", val)):
        matched = 0
        with open(OUT_DIR / f"needle_thai_{split}.jsonl", "w", encoding="utf-8") as f:
            for ex in rows:
                f.write(json.dumps(
                    {k: ex[k] for k in ("query", "tools", "answers", "reasoning")
                     if k in ex and ex[k] is not None},
                    ensure_ascii=False) + "\n")
        matched = 0
        with open(OUT_DIR / f"needle_thai_asr_{split}.jsonl", "w", encoding="utf-8") as f:
            for ex in rows:
                rec = audio.get(ex["query"])
                if rec is None:
                    continue
                matched += 1
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {split}: {len(rows)} text rows, {matched} with audio")

    tool_counts = Counter(
        a["name"] for ex in examples for a in ex["answers"])
    stats = {
        "total": len(examples),
        "deduped_removed": dupes,
        "on_topic": sum(1 for e in examples if e["answers"]),
        "off_topic": sum(1 for e in examples if not e["answers"]),
        "train": len(train),
        "val": len(val),
        "audio_matched": len(audio),
        "per_shard": dict(Counter(e["_shard"] for e in examples)),
        "top_tools": tool_counts.most_common(20),
    }
    (OUT_DIR / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nwrote {OUT_DIR}/needle_thai_{{train,val}}.jsonl"
          f" (+ asr variants; {matched}/{len(train)} train rows have audio)")


if __name__ == "__main__":
    main()
