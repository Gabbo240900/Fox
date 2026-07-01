"""Command-line interface: `fox predict --tgl file.tgl` / `python -m fox ...`."""

import argparse
import json
import sys

from .core import predict_tgl
from .encoding import EVENT_NAMES


def _add_predict_parser(sub):
    p = sub.add_parser("predict", help="Predict cophylogenetic event frequencies.")
    p.add_argument("--tgl", required=True, help="Well-formatted .tgl bundle.")
    p.add_argument("--ckpt", default=None,
                   help="Checkpoint path (default: $FOX_CKPT or repo fox.ckpt).")
    p.add_argument("--device", default="cpu", help="cpu | cuda | mps.")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    return p


def main(argv=None):
    parser = argparse.ArgumentParser(prog="fox", description="Fox cophylogenetics model.")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_predict_parser(sub)
    args = parser.parse_args(argv)

    if args.command == "predict":
        result = predict_tgl(
            args.tgl,
            ckpt_path=args.ckpt,
            device=args.device,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            dominant = max(result, key=result.get)
            print(f"{'Event':12s}  {'Frequency':>10s}")
            print("-" * 24)
            for e in EVENT_NAMES:
                print(f"{e:12s}  {result[e]:10.4f}")
            print(f"\nDominant: {dominant} ({result[dominant]:.1%})")
        return 0


if __name__ == "__main__":
    sys.exit(main())
