"""Helpers shared by the scripts."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--quick", action="store_true",
                   help="short run to check that the script works (results are not publication quality)")
    return p.parse_args()


def dump(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    print(f"wrote {path}")
