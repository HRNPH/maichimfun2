#!/usr/bin/env python3
"""Post-training sanity eval: run tuned Needle on a sample of the val split
and compare emitted calls against the expected answers.

Verified against the installed cactus-needle API: schema dicts are accepted
directly by Needle(tools=[...]), and complete() returns a dict whose
"function_calls" hold {"name", "arguments"} without executing anything.

Usage: python scripts/eval_thai.py --weights out/needle_thai.cact [--n 25]
"""
import argparse
import json
import random
from pathlib import Path

import needle
from needle import Needle

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            (ROOT / "data/dataset/needle_thai_val.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    sample = random.Random(args.seed).sample(rows, min(args.n, len(rows)))

    exact = tool_ok = 0
    for ex in sample:
        # one agent per example: the tool catalogue differs per row
        agent = Needle(tools=ex["tools"], weights=args.weights)
        res = agent.complete(ex["query"])
        calls = res.get("function_calls") or []
        got = [(c.get("name"), c.get("arguments") or {}) for c in calls]
        want = [(a["name"], a["arguments"]) for a in ex["answers"]]
        names_match = [n for n, _ in got] == [n for n, _ in want]
        if names_match and got == want:
            exact += 1
        if names_match:
            tool_ok += 1
        print(f"Q: {ex['query'][:48]}")
        print(f"   want={want}")
        print(f"   got ={got}")

    print(f"\n{len(sample)} sampled | exact match {exact}/{len(sample)}"
          f" | right tool {tool_ok}/{len(sample)}")
    print("note: confidence is reported as None for tuned weights (expected)")


if __name__ == "__main__":
    main()
