#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SU2.opt.progressive_ffd_split import split_bootstrap_ffd_box


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Replace one 2D bootstrap FFD box with independent curved "
            "upper/lower boxes."
        )
    )
    parser.add_argument("--mesh-in", required=True)
    parser.add_argument("--mesh-out", required=True)
    parser.add_argument("--bootstrap-tag", required=True)
    parser.add_argument("--marker", required=True)
    parser.add_argument("--upper-tag", default="UPPER_BOX")
    parser.add_argument("--lower-tag", default="LOWER_BOX")
    parser.add_argument("--upper-offset-chord", type=float, required=True)
    parser.add_argument("--lower-offset-chord", type=float, required=True)
    parser.add_argument("--x-le", type=float, default=None)
    parser.add_argument("--x-te", type=float, default=None)
    parser.add_argument("--diagnostics-csv", default=None)
    parser.add_argument(
        "--output-blending",
        choices=("BEZIER", "BSPLINE_UNIFORM"),
        default="BEZIER",
    )
    parser.add_argument("--bspline-order-i", type=int, default=2)
    parser.add_argument("--bspline-order-j", type=int, default=2)
    parser.add_argument("--bspline-order-k", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return split_bootstrap_ffd_box(
        args.mesh_in,
        args.mesh_out,
        bootstrap_tag=args.bootstrap_tag,
        marker=args.marker,
        upper_offset_chord=args.upper_offset_chord,
        lower_offset_chord=args.lower_offset_chord,
        upper_tag=args.upper_tag,
        lower_tag=args.lower_tag,
        x_le=args.x_le,
        x_te=args.x_te,
        diagnostics_csv=args.diagnostics_csv,
        overwrite=args.overwrite,
        output_blending=args.output_blending,
        bspline_orders=(
            args.bspline_order_i,
            args.bspline_order_j,
            args.bspline_order_k,
        ),
    )


if __name__ == "__main__":
    main()
