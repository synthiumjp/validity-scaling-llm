import json

import numpy as np
import pytest

from validity_screen import (screen, binarise, errors_needed, items_needed, power_invalid, power_valid,
                             screen_batch, summary_table, wilson_ci, newcombe_diff_ci, midp_signal, MIN_N)
from validity_screen.cli import main as cli_main


def make(nc, ni, keep_c, keep_i):
    """nc correct (keep_c kept), ni wrong (keep_i kept)."""
    correct = np.r_[np.ones(nc, int), np.zeros(ni, int)]
    keep = np.r_[np.ones(keep_c), np.zeros(nc - keep_c), np.ones(keep_i), np.zeros(ni - keep_i)].astype(int)
    return correct, keep


# ---------------- tiers (D1-D4) ----------------

def test_blanket_confidence_invalid():
    r = screen(*make(460, 40, 460, 40))
    assert r.tier == "Invalid" and r.reason_code == "blanket_confidence"


def test_blanket_withdrawal_invalid():
    r = screen(*make(460, 40, 0, 0))
    assert r.tier == "Invalid" and r.reason_code == "blanket_withdrawal"


def test_inversion_invalid():
    r = screen(*make(460, 40, 92, 32))          # keep 20% of correct, 80% of errors
    assert r.tier == "Invalid" and r.reason_code == "inversion"


def test_good_monitor_valid():
    r = screen(*make(460, 40, 368, 16))         # keep 80% of correct, 40% of errors
    assert r.tier == "Valid" and r.reason_code == "signal_detected" and r.flags == []


def test_conservative_but_informative_is_valid_with_low_coverage():
    # Fp = .55, L = 0: strong discrimination at a conservative operating point (v0.3 called this Invalid)
    r = screen(*make(400, 100, 180, 0))
    assert r.tier == "Valid" and "low_coverage" in r.flags
    assert abs(r.J.value - 0.45) < 1e-12


def test_uninformative_indeterminate():
    r = screen(*make(466, 58, 373, 46))         # keep 80% of both
    assert r.tier == "Indeterminate" and r.reason_code == "no_detectable_sensitivity"


def test_inversion_suspected():
    r = screen(*make(466, 58, 373, 48))         # keeps errors slightly more, not significant
    assert r.tier == "Indeterminate" and r.reason_code == "inversion_suspected"


def test_high_L_few_errors_suspected():
    r = screen(*make(476, 24, 457, 23))         # L = .958 on 24 errors
    assert r.tier == "Indeterminate" and r.reason_code == "blanket_confidence_suspected"
    assert r.L.flag == "suspected"


def test_high_Fp_few_correct_suspected():
    r = screen(*make(24, 476, 1, 10))           # Fp = .958 on 24 correct; not inverted
    assert r.tier == "Indeterminate" and r.reason_code == "blanket_withdrawal_suspected"


def test_r1_like_is_uncertifiable_with_low_coverage():
    r = screen(*make(446, 6, 24, 0))            # 6 errors, Fp = .946 (below .95)
    assert r.tier == "Uncertifiable" and r.reason_code == "too_few_errors"
    assert "low_coverage" in r.flags


def test_blanket_withdrawal_certified_even_with_few_errors():
    r = screen(*make(446, 6, 10, 0))            # Fp = .978, LB > .90, only 6 errors
    assert r.tier == "Invalid" and r.reason_code == "blanket_withdrawal"


def test_too_few_correct():
    r = screen(*make(8, 300, 4, 100))
    assert r.tier == "Uncertifiable" and r.reason_code == "too_few_correct"


def test_response_set_not_excluded_flag():
    # Valid, but the L interval reaches .95 (few errors, high L)
    r = screen(*make(470, 30, 420, 26))
    assert r.tier == "Valid" or r.tier == "Indeterminate"
    if r.tier == "Valid":
        assert ("response_set_not_excluded" in r.flags) == (r.L.hi >= 0.95 or r.Fp.hi >= 0.95)


# ---------------- boundaries ----------------

@pytest.mark.parametrize("ni,expect_estimable", [(MIN_N - 1, False), (MIN_N, True)])
def test_min_n_boundary_errors(ni, expect_estimable):
    r = screen(*make(400, ni, 300, ni // 2))
    assert (r.J.flag != "not_estimable") == expect_estimable
    if not expect_estimable:
        assert r.tier == "Uncertifiable"


def test_response_set_threshold_boundary():
    # exactly 95% kept on 140 errors with 7 withdrawn: certifiable (errors_needed(7) == 140)
    r = screen(*make(400, 140, 380, 133))
    assert r.L.value == pytest.approx(0.95) and r.tier == "Invalid"
    # 139 errors with 7 withdrawn: L = .9496 < .95 -> not a response set
    r = screen(*make(400, 139, 380, 132))
    assert r.L.value < 0.95 and r.L.flag == "ok"


def test_zero_cells():
    for args in [(100, 20, 100, 0), (100, 20, 0, 20), (100, 20, 50, 0), (100, 20, 100, 20)]:
        r = screen(*make(*args))
        assert r.tier in {"Valid", "Invalid", "Indeterminate", "Uncertifiable"}


def test_empty_input():
    r = screen([], [])
    assert r.tier == "Uncertifiable" and r.n_scored == 0


# ---------------- numerics ----------------

def test_newcombe_reference_value():
    # Newcombe (1998), example 56/70 vs 48/80: interval .0524 to .3339
    lo, hi = newcombe_diff_ci(56, 70, 48, 80)
    assert lo == pytest.approx(0.0524, abs=1e-4) and hi == pytest.approx(0.3339, abs=1e-4)


def test_wilson_reference_value():
    lo, hi = wilson_ci(81, 263)        # Wilson interval for 81/263: .2553 to .3662
    assert lo == pytest.approx(0.2553, abs=1e-4) and hi == pytest.approx(0.3662, abs=1e-4)


def test_midp_symmetry_and_range():
    ps, pi = midp_signal(30, 10, 10, 30)
    assert 0 <= ps <= 1 and 0 <= pi <= 1 and ps + pi == pytest.approx(1.0)
    assert ps < 0.001


def test_identities():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for _ in range(100):
        cor = (rng.random(400) < rng.uniform(.6, .95)).astype(int)
        conf = np.where(rng.random(400) < .7, cor, 1 - cor)
        conf = np.where(rng.random(400) < .5, 1, conf)
        r = screen(cor, conf)
        if np.isnan(r.J.value):
            continue
        assert abs(r.auroc_binary - sk.roc_auc_score(cor, conf)) < 1e-12
        assert abs(r.youden_j - ((1 - r.Fp.value) - r.L.value)) < 1e-12
        assert abs(r.rbs.value + r.J.value) < 1e-12


# ---------------- input checks (D8) ----------------

def test_rejects_raw_continuous_confidence():
    c = np.r_[np.ones(450), np.zeros(50)]
    conf = np.full(500, 90.0)
    with pytest.raises(ValueError, match="cut="):
        screen(c, conf)


def test_rejects_nan_confidence_on_scored_rows():
    c, k = make(450, 50, 400, 20)
    k = k.astype(float); k[3] = np.nan
    with pytest.raises(ValueError, match="missing"):
        screen(c, k)
    ans = np.ones(500, bool); ans[3] = False
    r = screen(c, k, answered=ans)
    assert r.n_scored == 499


def test_rejects_nan_correct():
    c, k = make(450, 50, 400, 20)
    c = c.astype(float); c[0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        screen(c, k)


def test_cut_excludes_and_counts_missing():
    c = np.r_[np.ones(450), np.zeros(50)]
    conf = np.r_[np.full(450, 95.0), np.full(50, 60.0)]
    conf[:5] = np.nan
    r = screen(c, conf, cut=90)
    assert r.n_missing == 5 and r.n_scored == 495 and r.binarisation_threshold == ">= 90"


def test_cut_on_binary_rejected():
    c, k = make(450, 50, 400, 20)
    with pytest.raises(ValueError, match="looks binary"):
        screen(c, k, cut=90)


def test_binarise_requires_threshold_and_rejects_nan():
    x = np.array([10, 50, 90, 100])
    assert binarise(x, 90).tolist() == [False, False, True, True]
    with pytest.raises(TypeError):
        binarise(x)
    with pytest.raises(ValueError):
        binarise(np.array([10, np.nan]), 50)


def test_exclusions():
    c, k = make(460, 40, 368, 16)
    ans = np.ones(500, int); ans[:10] = 0
    r = screen(c, k, answered=ans)
    assert r.n_excluded == 10 and r.n_scored == 490


# ---------------- planning ----------------

def test_errors_needed_integer_table():
    assert [errors_needed(k) for k in range(8)] == [35, 53, 69, 84, 99, 113, 127, 140]
    with pytest.raises(TypeError):
        errors_needed(0.97)


def test_power_invalid_values():
    assert power_invalid(1.0, 35) == pytest.approx(1.0)
    assert power_invalid(0.97, 71) == pytest.approx(0.641, abs=0.001)
    assert max(power_invalid(0.95, n) for n in (200, 500, 1000)) < 0.6
    assert items_needed(0.92, 1.0) == 438
    assert items_needed(0.92, 0.95) is None


def test_false_valid_rate_held():
    for acc, keep in [(.92, .95), (.95, .90), (.97, .99)]:
        assert power_valid(452, acc, keep, keep) <= 0.025


# ---------------- batch, report, CLI ----------------

def test_batch_summary_report():
    c, k = make(460, 40, 368, 16)
    res = screen_batch({"a": {"correct": c, "confidence": k}, "b": {"correct": c, "confidence": np.ones(500)}})
    s = summary_table(res)
    assert "Invalid" in s and "Valid" in s
    assert "VALIDITY REPORT" in res[0].vrs_table()
    json.dumps(res[0].to_dict())


def _csv(tmp_path, rows, name="d.csv"):
    p = tmp_path / name
    p.write_text("correct,confidence\n" + "\n".join(f"{a},{b}" for a, b in rows) + "\n")
    return str(p)


def test_cli_continuous_with_blank_cell(tmp_path, capsys):
    rows = [(1, 95)] * 400 + [(1, 60)] * 50 + [(0, 95)] * 10 + [(0, 50)] * 40
    rows[0] = (1, "")                          # one blank confidence cell
    p = _csv(tmp_path, rows)
    assert cli_main(["run", "--data", p, "--cut", "90", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_missing"] == 1 and out["n_scored"] == 499 and out["tier"] == "Valid"


def test_cli_continuous_requires_cut(tmp_path, capsys):
    p = _csv(tmp_path, [(1, 95), (0, 40)] * 50)
    with pytest.raises(SystemExit):
        cli_main(["run", "--data", p])


def test_cli_binary_words(tmp_path, capsys):
    rows = [("True", "KEEP")] * 400 + [("True", "WITHDRAW")] * 50 + [("False", "KEEP")] * 10 + \
           [("False", "WITHDRAW")] * 40
    p = _csv(tmp_path, rows)
    assert cli_main(["run", "--data", p, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["tier"] == "Valid" and out["a"] == 400 and out["d"] == 40


def test_cli_unreadable_cell_excluded(tmp_path, capsys):
    rows = [("True", "KEEP")] * 400 + [("True", "WITHDRAW")] * 50 + [("False", "KEEP")] * 10 + \
           [("False", "WITHDRAW")] * 40
    rows[5] = ("True", "maybe")
    p = _csv(tmp_path, rows)
    assert cli_main(["run", "--data", p, "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_missing"] == 1 and out["n_scored"] == 499


def test_cli_plan(capsys):
    assert cli_main(["plan", "--accuracy", "0.92", "--true-L", "0.95"]) == 0
    assert "cannot be reached" in capsys.readouterr().out


# ---------------- 0.4.1 regressions (independent review, defects P1-P13) ----------------

def _cont(n_c=450, n_w=50, hi=95.0, lo=60.0, scale=1.0):
    c = np.r_[np.ones(n_c), np.zeros(n_w)]
    conf = np.r_[np.full(n_c - 50, hi), np.full(50, lo), np.full(10, hi), np.full(n_w - 10, lo)] * scale
    return c, conf


def test_P1_unit_scale_mismatch_0_1_with_cut_90():
    c, conf = _cont(scale=0.01)
    with pytest.raises(ValueError, match="0-1 scale"):
        screen(c, conf, cut=90)


def test_P2_unit_scale_mismatch_0_100_with_cut_09():
    c, conf = _cont()
    with pytest.raises(ValueError, match="0-100 scale"):
        screen(c, conf, cut=0.9)


def test_P3_nan_cut():
    c, conf = _cont()
    with pytest.raises(ValueError):
        screen(c, conf, cut=float("nan"))


def test_P4_items_needed_integer_ceiling():
    assert items_needed(0.9, 1.0, 0.9) == 350
    assert items_needed(0.8, 1.0, 0.9) == 175
    assert items_needed(0.9, 0.97, 0.9) == 1600


def test_P5_flag_independent_of_alpha():
    c, k = make(425, 30, 422, 27)
    flags = [screen(c, k, alpha=a).flags for a in (0.05, 0.5)]
    assert flags[0] == flags[1]


@pytest.mark.parametrize("a", [0, -1, 2, 1])
def test_P6_alpha_range(a):
    c, k = make(450, 50, 400, 20)
    with pytest.raises(ValueError):
        screen(c, k, alpha=a)


def test_P7_argument_validation():
    with pytest.raises(ValueError):
        power_valid(452.5, .9, .95, .85)
    with pytest.raises(ValueError):
        power_valid(452, .9, 1.2, .5)
    with pytest.raises(ValueError):
        power_invalid(1.2, 50)
    assert cli_main(["plan", "--accuracy", "0.9", "--true-L", "1.5"]) == 1
    assert cli_main(["plan", "--accuracy", "0.9", "--power", "1.5"]) == 1


def test_P8_unreadable_confidence_column(tmp_path, capsys):
    p = tmp_path / "d.csv"
    p.write_text("correct,confidence\n" + "\n".join("1,n/a" for _ in range(20)) + "\n")
    assert cli_main(["run", "--data", str(p), "--cut", "90"]) == 1
    assert "no confidence cell could be read" in capsys.readouterr().err


def test_P8_percent_strings_are_read(tmp_path, capsys):
    rows = ["1,95%"] * 400 + ["1,60%"] * 50 + ["0,95%"] * 10 + ["0,50%"] * 40
    p = tmp_path / "d.csv"
    p.write_text("correct,confidence\n" + "\n".join(rows) + "\n")
    assert cli_main(["run", "--data", str(p), "--cut", "90", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["n_scored"] == 500


def test_P9_words_in_numeric_column_are_unreadable(tmp_path, capsys):
    rows = ["1,95"] * 400 + ["1,60"] * 50 + ["0,95"] * 10 + ["0,50"] * 40
    rows[0] = "1,high"
    p = tmp_path / "d.csv"
    p.write_text("correct,confidence\n" + "\n".join(rows) + "\n")
    assert cli_main(["run", "--data", str(p), "--cut", "90", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_missing"] == 1 and out["n_scored"] == 499


def test_P10_no_scored_rows():
    r = screen([1, 0, 1], [1, 1, 0], answered=[0, 0, 0])
    assert r.tier == "Uncertifiable" and r.reason_code == "no_scored_rows"


def test_P11_shapes_and_lengths():
    c, k = make(450, 50, 400, 20)
    with pytest.raises(ValueError):
        screen(c, k, answered=np.ones(10))
    with pytest.raises(ValueError):
        screen(c.reshape(10, 50), k.reshape(10, 50))


def test_P12_header_lines_skipped(tmp_path, capsys):
    pc, pk = tmp_path / "c.txt", tmp_path / "k.txt"
    pc.write_text("correct\n" + "\n".join(["1"] * 450 + ["0"] * 50) + "\n")
    pk.write_text("keep\n" + "\n".join(["1"] * 400 + ["0"] * 50 + ["1"] * 10 + ["0"] * 40) + "\n")
    assert cli_main(["run", "--correct", str(pc), "--confidence", str(pk), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_items"] == 500 and out["n_missing"] == 0


def test_P13_reason_code_vs_reason():
    r = screen(*make(460, 40, 368, 16))
    assert r.reason_code == "signal_detected" and r.reason.startswith("J > 0")


def test_min_j_option():
    c, k = make(4500, 500, 4300, 450)          # J ~ .055, highly significant
    assert screen(c, k).tier == "Valid"
    r = screen(c, k, min_j=0.10)
    assert r.tier == "Indeterminate" and r.reason_code == "sensitivity_below_minimum"
