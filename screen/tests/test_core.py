import numpy as np
import pytest

from validity_screen import screen, binarise, errors_needed, items_needed, screen_batch, summary_table


def make(nc, ni, keep_c, keep_i):
    correct = np.r_[np.ones(nc, int), np.zeros(ni, int)]
    keep = np.r_[np.ones(keep_c), np.zeros(nc - keep_c), np.ones(keep_i), np.zeros(ni - keep_i)].astype(int)
    return correct, keep


def test_always_keep_invalid():
    c, k = make(460, 40, 460, 40)
    assert screen(c, k).tier == "Invalid"


def test_always_withdraw_invalid():
    c, k = make(460, 40, 0, 0)
    assert screen(c, k).tier == "Invalid"


def test_inverted_invalid():
    c, k = make(460, 40, 92, 32)          # keep 20% of correct, 80% of errors
    assert screen(c, k).tier == "Invalid"


def test_good_monitor_valid():
    c, k = make(460, 40, 368, 16)         # keep 80% of correct, 40% of errors
    r = screen(c, k)
    assert r.tier == "Valid" and r.tier_v02 == "Valid"


def test_uninformative_not_valid_under_v03():
    c, k = make(466, 58, 373, 46)         # keep 80% of both
    r = screen(c, k)
    assert r.tier == "Indeterminate"
    assert r.tier_v02 == "Valid"
    assert screen(c, k, require_signal=False).tier == "Valid"


def test_high_L_few_errors_indeterminate():
    c, k = make(476, 24, 457, 23)         # L = .958 on 24 errors
    r = screen(c, k)
    assert r.tier == "Indeterminate" and r.L.flag == "indeterminate"


def test_not_estimable_with_few_errors():
    c, k = make(446, 6, 24, 0)            # R1-like: 6 errors, Fp = .946
    r = screen(c, k)
    assert r.L.flag == "not_estimable" and r.rbs.flag == "not_estimable"
    assert r.tier == "Invalid"            # decided by Fp


def test_exclusions():
    c, k = make(460, 40, 368, 16)
    ans = np.ones(500, int); ans[:10] = 0
    r = screen(c, k, answered=ans)
    assert r.n_excluded == 10 and r.n_scored == 490


def test_identities():
    rng = np.random.default_rng(0)
    from sklearn.metrics import roc_auc_score
    for _ in range(100):
        cor = (rng.random(400) < rng.uniform(.6, .95)).astype(int)
        conf = np.where(rng.random(400) < .7, cor, 1 - cor)
        conf = np.where(rng.random(400) < .5, 1, conf)
        r = screen(cor, conf)
        if np.isnan(r.rbs.value):
            continue
        assert abs(r.auroc_binary - roc_auc_score(cor, conf)) < 1e-12
        assert abs(r.youden_j + r.rbs.value) < 1e-12


def test_errors_needed():
    assert [errors_needed(x) for x in (1.0, .97, .96, .95)] == [35, 71, 97, 139]
    assert errors_needed(.90) is None
    assert items_needed(0.92, 1.0) == 438


def test_binarise_and_median_deprecated():
    x = np.array([10, 50, 90, 100])
    assert binarise(x, 90).tolist() == [False, False, True, True]
    with pytest.warns(DeprecationWarning):
        binarise(x, method="median")


def test_batch_and_summary():
    c, k = make(460, 40, 368, 16)
    res = screen_batch({"a": {"correct": c, "confidence": k}, "b": {"correct": c, "confidence": np.ones(500)}})
    s = summary_table(res)
    assert "Invalid" in s and "Valid" in s
    assert "VRS TABLE" in res[0].vrs_table()
