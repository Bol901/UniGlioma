"""Enumerate generic task instructions; no reports or patient records are read."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.prompts import enumerate_taskless_prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="local_data/prompts.json")
    parser.add_argument("--seg-catalog", default="assets/seg_prompts.json")
    args = parser.parse_args()
    prompts = sorted(set(enumerate_taskless_prompts(seg_catalog_path=args.seg_catalog)))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prompts, indent=2) + "\n")
    print(f"Wrote {len(prompts)} generic task instructions to {path}")


if __name__ == "__main__":
    main()
