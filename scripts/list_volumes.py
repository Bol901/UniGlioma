"""Extract unique MRI paths from caller-owned training manifests for VAE caching."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-full-head", action="store_true")
    args = parser.parse_args()
    paths = set()
    for name in args.records:
        for row in json.loads(Path(name).read_text()):
            for image in row["images"]:
                path = str(image["path"])
                if not path.endswith(".h5"):
                    raise ValueError("Expected explicit .h5 paths")
                paths.add(path)
                if args.include_full_head and path.endswith("_bet.h5"):
                    paths.add(path[: -len("_bet.h5")] + ".h5")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sorted(paths), indent=2) + "\n")
    print(f"Wrote {len(paths)} paths to {out}")


if __name__ == "__main__":
    main()
