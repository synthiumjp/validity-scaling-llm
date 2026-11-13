"""
Command-line interface for validity-screen.

    validity-screen run --data results.csv --correct-col is_correct --confidence-col keep
    validity-screen run --data results.csv --confidence-col confidence --threshold 90
    validity-screen plan --accuracy 0.92 --L 1.0
    validity-screen version
"""

import argparse
import csv
import json
import sys

import numpy as np

TRUE = ("true", "1", "1.0", "yes", "keep", "bet", "high")


def _parse_conf(vals):
    """Return (array, is_binary). Numeric strings become floats; words become 0/1."""
    out, numeric = [], True
    for v in vals:
        s = str(v).strip().lower()
        try:
            out.append(float(s))
        except ValueError:
            numeric = False
            out.append(1.0 if s in TRUE else 0.0)
    arr = np.array(out, dtype=float)
    return arr, (not numeric) or bool(np.all(np.isin(arr, [0.0, 1.0])))


def _load(args):
    if args.data:
        with open(args.data, newline="") as f:
            rows = list(csv.DictReader(f))
        correct = np.array([str(r[args.correct_col]).strip().lower() in TRUE for r in rows])
        conf_raw = [r[args.confidence_col] for r in rows]
        answered = None
        if args.answered_col:
            answered = np.array([str(r[args.answered_col]).strip().lower() in TRUE for r in rows])
        return correct, conf_raw, answered
    if args.correct and args.confidence:
        def col(p):
            with open(p) as f:
                return [ln.strip() for ln in f if ln.strip()]
        return np.array([v.lower() in TRUE for v in col(args.correct)]), col(args.confidence), None
    raise SystemExit("Error: provide --data, or both --correct and --confidence.")


def main(argv=None):
    p = argparse.ArgumentParser(prog="validity-screen",
                                description="Validity screen for LLM confidence signals.")
    sub = p.add_subparsers(dest="command")

    r = sub.add_parser("run", help="Screen one model's data.")
    r.add_argument("--data", help="CSV with correctness and confidence columns.")
    r.add_argument("--correct", help="File of correctness labels, one per line.")
    r.add_argument("--confidence", help="File of confidence values, one per line.")
    r.add_argument("--correct-col", default="correct")
    r.add_argument("--confidence-col", default="confidence")
    r.add_argument("--answered-col", default=None,
                   help="Optional column; rows that are not true (declines, parse failures, "
                        "pipeline defaults) are excluded before scoring.")
    r.add_argument("--threshold", type=float, default=None,
                   help="Cut point for continuous confidence (confidence >= threshold is high). "
                        "Required for continuous confidence; choose it in advance.")
    r.add_argument("--model-name", default="")
    r.add_argument("--benchmark", default="")
    r.add_argument("--v02", action="store_true",
                   help="Use the version 0.2 rule (Valid = no flag fired).")
    r.add_argument("--json", action="store_true", help="JSON output instead of the VRS Table.")

    pl = sub.add_parser("plan", help="Items/errors needed to certify blanket confidence.")
    pl.add_argument("--accuracy", type=float, required=True)
    pl.add_argument("--L", type=float, default=1.0, help="Observed L to certify (default 1.0).")

    sub.add_parser("version", help="Print version.")
    args = p.parse_args(argv)

    if args.command == "version":
        from validity_screen import __version__
        print(f"validity-screen {__version__}")
        return 0
    if args.command == "plan":
        from validity_screen import errors_needed, items_needed
        e, n = errors_needed(args.L), items_needed(args.accuracy, args.L)
        if e is None:
            print(f"L = {args.L} cannot be certified above the .90 guard.")
        else:
            print(f"To certify L = {args.L} (Wilson lower bound > .90): {e} errors, "
                  f"about {n} items at accuracy {args.accuracy}.")
        return 0
    if args.command != "run":
        p.print_help()
        return 0

    from validity_screen import screen, binarise
    try:
        correct, conf_raw, answered = _load(args)
        conf, is_binary = _parse_conf(conf_raw)
        if is_binary:
            confidence, cut = conf.astype(bool), "N/A (binary)"
        else:
            if args.threshold is None:
                raise SystemExit("Error: confidence is continuous; pass --threshold (a cut point chosen "
                                 "in advance, ideally the deployment threshold). Median splits are not "
                                 "supported because their verdict depends on tie handling.")
            confidence, cut = binarise(conf, args.threshold), f">= {args.threshold:g}"
        res = screen(correct, confidence, answered=answered, model_name=args.model_name,
                     benchmark_name=args.benchmark, binarisation_threshold=cut,
                     require_signal=not args.v02)
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(res.to_dict(), indent=2) if args.json else res.vrs_table())
    return 0


if __name__ == "__main__":
    sys.exit(main())
