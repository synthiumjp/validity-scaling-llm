"""Data loaders for the v0.4 analysis. Every loader returns one row per scored or
excluded response; nothing here decides a tier."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_PATTERNS = [
    ("GPT-5.4 nano", "GPT-5.4 nano"), ("GPT-5.4 mini", "GPT-5.4 mini"), ("GPT-5.4", "GPT-5.4"),
    ("Claude Opus 4.6", "Opus 4.6"), ("Claude Sonnet 4.6", "Sonnet 4.6"),
    ("Claude Haiku 4.5", "Claude Haiku 4.5"), ("DeepSeek-R1", "DeepSeek-R1"),
    ("DeepSeek V3.2", "DeepSeek V3.2"), ("Gemini 3.1 Pro", "Gemini 3.1 Pro"),
    ("Gemini 2.5 Pro", "Gemini 2.5 Pro"), ("Gemini 2.5 Flash", "Gemini 2.5 Flash"),
    ("Gemini 3 Flash", "Gemini 3 Flash"), ("Gemma 3 1B", "Gemma 3 1B"),
    ("Gemma 3 12B", "Gemma 3 12B"), ("Gemma 3 27B", "Gemma 3 27B"), ("GLM-5", "GLM-5"), ("GLM 5", "GLM-5"), ("haiku", "Claude Haiku 4.5"), ("Opus 4.6", "Opus 4.6"), ("Sonnet 4.6", "Sonnet 4.6"), ("Gemini Flash 3", "Gemini 3 Flash"),
    ("Qwen 3 235B", "Qwen 235B"), ("Qwen 3 Coder 480B", "Qwen Coder 480B"),
    ("Qwen 3 Next 80B Thinking", "Qwen 80B Think"), ("Qwen 3 Next 80B Think", "Qwen 80B Think"), ("Qwen 3 Next 80B Instruct", "Qwen 80B Inst"), ("Qwen 3 Next 80B Inst", "Qwen 80B Inst"), ("Gemini 3 Flash Preview", "Gemini 3 Flash"), ("R1", "DeepSeek-R1"),
]
TRACKS = [("attention", "attention"), ("exec", "executive"), ("metacog", "metacognition"),
          ("overhyp", "learning"), ("socialcog", "social"), ("soccog", "social"), ("prospective", "prospective")]
FAMILY = {"Claude Haiku 4.5": "Anthropic", "Opus 4.6": "Anthropic", "Sonnet 4.6": "Anthropic",
          "GPT-5.4": "OpenAI", "GPT-5.4 mini": "OpenAI", "GPT-5.4 nano": "OpenAI",
          "Gemini 2.5 Flash": "Gemini", "Gemini 2.5 Pro": "Gemini", "Gemini 3 Flash": "Gemini",
          "Gemini 3.1 Pro": "Gemini", "Gemma 3 1B": "Gemma", "Gemma 3 12B": "Gemma",
          "Gemma 3 27B": "Gemma", "Qwen 235B": "Qwen", "Qwen 80B Inst": "Qwen",
          "Qwen 80B Think": "Qwen", "Qwen Coder 480B": "Qwen", "DeepSeek V3.2": "DeepSeek",
          "DeepSeek-R1": "DeepSeek", "GLM-5": "Zhipu"}
PUBLISHED_INVALID = ["DeepSeek-R1", "Gemini 3.1 Pro", "Qwen 80B Think"]




def model_of(stem):
    s = stem.lower()
    for pat, canon in MODEL_PATTERNS:
        if pat.lower() in s:
            return canon
    raise ValueError(stem)


def track_of(stem):
    s = stem.lower().replace(" ", "")
    for k, t in TRACKS:
        if k in s:
            return t
    raise ValueError(stem)


def is_nonanswer(answer, correct_answer) -> bool:
    """Step 1 rule (frozen): a decline marker, or a bare option letter (A-D) where the key
    is not a letter (the harness captured a path choice rather than an answer)."""
    a = str(answer).strip()
    ca = str(correct_answer).strip()
    if "DECLIN" in a.upper():
        return True
    bare = len(a.rstrip(".)")) == 1 and a.rstrip(".)").upper() in "ABCD"
    ca_is_letter = len(ca) == 1 and ca.upper() in "ABCD"
    return bare and not ca_is_letter


def load_battery(folder, drop_tracks=("prospective",)):
    rows = []
    for f in sorted(Path(folder).glob("*.csv")):
        df = pd.read_csv(f)
        cc = "correct" if "correct" in df.columns else "is_correct"
        o = pd.DataFrame({
            "model": model_of(f.stem), "track": track_of(f.stem), "pos": np.arange(len(df)),
            "correct": df[cc].map(lambda x: str(x).strip().lower() in ["true", "1", "1.0"]).astype(int),
            "keep": (df["keep_withdraw"].astype(str).str.strip().str.upper() == "KEEP").astype(int),
            "bet": (df["bet_nobet"].astype(str).str.strip().str.upper() == "BET").astype(int),
        })
        if "answer" in df.columns and "correct_answer" in df.columns:
            o["answered"] = [int(not is_nonanswer(a, ca)) for a, ca in zip(df["answer"], df["correct_answer"])]
        else:
            o["answered"] = 1
        rows.append(o)
    D = pd.concat(rows, ignore_index=True)
    D = D[~D.track.isin(list(drop_tracks))].reset_index(drop=True)
    D["item"] = D.track + ":" + D.pos.astype(str)
    D["ordinal"] = D.keep * 2 + D.bet
    return D


def load_mmlu(folder, imputed=None):
    """Atlas / MMLU-500 item files. `imputed` maps a model-name substring to the parser
    default value; rows with that value in that model are marked not answered (Step 1)."""
    imputed = imputed or {}
    out = []
    for f in sorted(Path(folder).glob("*.csv")):
        d = pd.read_csv(f).drop_duplicates(subset=["model", "item_id"], keep="first")
        name = str(d.model.iloc[0])
        ans_ok = ~d.answer.astype(str).str.strip().str.upper().isin(["X", "", "NAN", "NONE"])
        conf = pd.to_numeric(d.confidence, errors="coerce")
        imp = pd.Series(False, index=d.index)
        for key, val in imputed.items():
            if key in name:
                imp = conf == val
        out.append(pd.DataFrame({
            "model": name, "item": d.item_id.values,
            "correct": d.is_correct.astype(str).str.lower().isin(["true", "1", "1.0"]).astype(int).values,
            "confidence": conf.values,
            "answered": (ans_ok & conf.notna() & ~imp).astype(int).values,
            "imputed": imp.astype(int).values, "unparsed_answer": (~ans_ok).astype(int).values}))
    return pd.concat(out, ignore_index=True)


def load_yang(base, tcut=0.5):
    """Yang, Tsai and Yamada (2024) released predictions: all benchmarks, models, methods.
    Confidence 0-1 is rescaled to 0-100. Fractional correctness is binarised at `tcut`."""
    base = Path(base)
    out = []
    for fp in sorted((base / "y_pred").glob("*/*/*.npy")):
        bench, model, method = fp.parts[-3], fp.parts[-2], fp.stem
        ft = base / "y_true" / bench / model / f"{method}.npy"
        if not ft.exists():
            continue
        p = np.load(fp, allow_pickle=False).astype(float)
        t = np.load(ft, allow_pickle=False).astype(float)
        if p.shape != t.shape:
            continue
        ok = np.isfinite(p) & np.isfinite(t)
        out.append(pd.DataFrame({"bench": bench, "model": model, "method": method,
                                 "correct": (np.nan_to_num(t) >= tcut).astype(int),
                                 "confidence": np.where(np.isfinite(p), p * 100, np.nan),
                                 "answered": ok.astype(int)}))
    return pd.concat(out, ignore_index=True)
