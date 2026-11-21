"""
Command-line interface for validity-screen.

    validity-screen run --data results.csv --correct-col is_correct --confidence-col keep
    validity-screen run --data results.csv --confidence-col confidence --cut 90
    (--correct/--confidence: one value per line; use --header if the first line is a column
     name, which is detected automatically when every other line is a number)
    validity-screen plan --accuracy 0.92 --true-L 0.97
    validity-screen version

Cells are read strictly. A blank or unreadable cell is never guessed: the row is
excluded and counted, and the count is reported.
"""

import argparse
import csv
import json
import math
import sys

import numpy as np

TRUE_WORDS = {"true", "yes", "keep", "kept", "bet", "high", "correct"}
FALSE_WORDS = {"false", "no", "withdraw", "withdrawn", "no bet", "no_bet", "low", "incorrect", "wrong"}


def _cell(v):
    """Return (value, kind): kind is 'num', 'word' or 'missing'."""
    s = "" if v is None else str(v).strip()
    if s.endswith("%"):
        s = s[:-1].strip()
    if s == "" or s.lower() in {"na", "nan", "none", "null"}:
        return math.nan, "missing"
    try:
        return float(s), "num"
    except ValueError:
        low = s.lower()
        if low in TRUE_WORDS:
            return 1.0, "word"
        if low in FALSE_WORDS:
            return 0.0, "word"
        return math.nan, "missing"


def _column(vals):
    """Parse a column. Words count as 0/1 only in a binary column; in a numeric column
    with values other than 0 and 1, a word is unreadable (never silently 0 or 1)."""
    out = [_cell(v) for v in vals]
    nums = np.array([x for x, k in out if k == "num"], dtype=float)
    continuous = bool(len(nums)) and not np.all(np.isin(nums, [0.0, 1.0]))
    arr = np.array([(np.nan if (k == "word" and continuous) else x) for x, k in out], dtype=float)
    unreadable = sum(1 for (x, k), v in zip(out, vals)
                     if (k == "missing" and str(v).strip() != "") or (k == "word" and continuous))
    return arr, unreadable


def _load(args):
    if args.data:
        with open(args.data, newline="") as f:
            rows = list(csv.DictReader(f))
        for col in [args.correct_col, args.confidence_col] + ([args.answered_col] if args.answered_col else []):
            if rows and col not in rows[0]:
                raise KeyError(f"column '{col}' not found; columns are: {', '.join(rows[0].keys())}")
        correct = [r[args.correct_col] for r in rows]
        conf = [r[args.confidence_col] for r in rows]
        answered = [r[args.answered_col] for r in rows] if args.answered_col else None
        return correct, conf, answered
    if args.correct and args.confidence:
        def col(p):
            with open(p) as f:
                return [ln.rstrip("\n") for ln in f]
        c, k = col(args.correct), col(args.confidence)
        if len(c) != len(k):
            raise SystemExit("Error: --correct and --confidence files have different numbers of lines.")
        def looks_like_header(lines):
            rest = [x for x in lines[1:] if x.strip()]
            return (bool(lines) and _cell(lines[0])[1] != "num" and lines[0].strip() != ""
                    and bool(rest) and all(_cell(x)[1] == "num" for x in rest))
        if args.header or (looks_like_header(c) and looks_like_header(k)):
            c, k = c[1:], k[1:]
        return c, k, None
    raise SystemExit("Error: provide --data, or both --correct and --confidence.")


def main(argv=None):
    p = argparse.ArgumentParser(prog="validity-screen",
                                description="Validity screen for LLM confidence signals.")
    sub = p.add_subparsers(dest="command")

    r = sub.add_parser("run", help="Screen one model's data.")
    r.add_argument("--data", help="CSV with correctness and confidence columns.")
    r.add_argument("--correct", help="File of correctness labels, one per line.")
    r.add_argument("--confidence", help="File of confidence values, one per line.")
    r.add_argument("--header", action="store_true",
                   help="With --correct/--confidence: the first line of each file is a header. "
                        "Detected automatically when it is a word and every other line is a number.")
    r.add_argument("--correct-col", default="correct")
    r.add_argument("--confidence-col", default="confidence")
    r.add_argument("--answered-col", default=None,
                   help="Optional column; rows that are not true (declines, parse failures, "
                        "pipeline defaults) are excluded before scoring.")
    r.add_argument("--cut", "--threshold", dest="cut", type=float, default=None,
                   help="Cut point for continuous confidence (confidence >= cut is kept). Required for "
                        "continuous confidence; choose it in advance from the threshold you intend to use.")
    r.add_argument("--model-name", default="")
    r.add_argument("--benchmark", default="")
    r.add_argument("--min-j", type=float, default=0.0,
                   help="Optional smallest item sensitivity of interest (default 0); choose it in advance.")
    r.add_argument("--json", action="store_true", help="JSON output instead of the report table.")

    pl = sub.add_parser("plan", help="Errors and items needed to certify blanket confidence.")
    pl.add_argument("--accuracy", type=float, required=True)
    pl.add_argument("--true-L", type=float, default=1.0,
                    help="Assumed true kept-error rate (default 1.0).")
    pl.add_argument("--power", type=float, default=0.90)
    pl.add_argument("--min-j", type=float, default=None, help=argparse.SUPPRESS)

    sub.add_parser("version", help="Print version.")
    args = p.parse_args(argv)

    if args.command == "version":
        from validity_screen import __version__
        print(f"validity-screen {__version__}")
        return 0
    if args.command == "plan":
        from validity_screen import items_needed, errors_needed
        for nm, v, ok in [("--accuracy", args.accuracy, 0 <= args.accuracy < 1),
                          ("--true-L", args.true_L, 0 <= args.true_L <= 1),
                          ("--power", args.power, 0 < args.power < 1)]:
            if not ok:
                print(f"Error: {nm} = {v} is out of range", file=sys.stderr)
                return 1
        n = items_needed(args.accuracy, args.true_L, args.power)
        print(f"Minimum errors to certify from an observed table: {errors_needed(0)} if every error is kept, "
              f"{errors_needed(1)} if one is withdrawn, {errors_needed(2)} if two are.")
        if n is None:
            print(f"With a true kept-error rate of {args.true_L}, a {args.power:.0%} chance of certifying "
                  f"blanket confidence cannot be reached at any sample size.")
        else:
            print(f"For a {args.power:.0%} chance of certifying blanket confidence at a true kept-error rate "
                  f"of {args.true_L}: about {n} items at accuracy {args.accuracy} "
                  f"({int(round(n * (1 - args.accuracy)))} errors).")
        return 0
    if args.command != "run":
        p.print_help()
        return 0

    from validity_screen import screen
    try:
        correct_raw, conf_raw, ans_raw = _load(args)
        correct, bad_c = _column(correct_raw)
        conf, bad_k = _column(conf_raw)
        if len(conf) == 0 or np.all(np.isnan(conf)):
            raise ValueError("no confidence cell could be read (expected numbers, or words such as KEEP/WITHDRAW)")
        if np.all(np.isnan(correct)):
            raise ValueError("no correctness cell could be read (expected 0/1 or True/False)")
        answered = np.ones(len(correct), dtype=bool)
        if ans_raw is not None:
            a, _ = _column(ans_raw)
            answered = a == 1.0
        missing_correct = np.isnan(correct) & answered
        answered &= ~np.isnan(correct)
        is_binary = bool(np.all(np.isin(conf[~np.isnan(conf)], [0.0, 1.0])))
        if is_binary and args.cut is None:
            missing_conf = np.isnan(conf) & answered
            answered &= ~np.isnan(conf)
            res = screen(np.nan_to_num(correct), np.nan_to_num(conf), answered=answered,
                         model_name=args.model_name, benchmark_name=args.benchmark, min_j=args.min_j)
            res.n_missing = int(missing_conf.sum() + missing_correct.sum())
        else:
            if args.cut is None:
                raise SystemExit("Error: confidence is continuous; pass --cut (a cut point chosen in "
                                 "advance from the threshold you intend to use).")
            res = screen(np.nan_to_num(correct), conf, answered=answered, cut=args.cut,
                         model_name=args.model_name, benchmark_name=args.benchmark, min_j=args.min_j)
            res.n_missing += int(missing_correct.sum())
        res.notes = [n for n in res.notes if "missing values" not in n]
        if res.n_missing:
            res.notes.append(f"{res.n_missing} rows excluded for blank or unreadable cells "
                             f"({bad_c + bad_k} unreadable)")
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(res.to_dict(), indent=2) if args.json else res.vrs_table())
    return 0


if __name__ == "__main__":
    sys.exit(main())
