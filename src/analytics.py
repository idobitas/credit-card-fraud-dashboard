"""The statistics engine: binning, lift, WoE / Information Value, and rule scoring.

No model library is used. Everything here is a closed-form statistic over the
contingency table of a binned feature against the fraud label, which keeps every
number on screen traceable to an arithmetic the user can check by hand.

Sample-size discipline matters more than usual: the file contains 10,000 rows but
only 151 frauds. A bin holding 3 frauds produces a wild-looking lift that is pure
noise, so every bin carries a Wilson confidence interval and bins below
MIN_FRAUDS are flagged and, by default, kept out of the Information Value sum.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from .data import FEATURES, TARGET, Feature

MIN_FRAUDS = 5           # below this a bin is statistically untrustworthy
HALDANE = 0.5            # cell correction so an empty bin cannot yield log(0)
Z = 1.959963985          # 95% normal quantile


# --------------------------------------------------------------------------- #
# Binning
# --------------------------------------------------------------------------- #
def bin_series(df: pd.DataFrame, name: str, edges: list[float] | None = None) -> pd.Series:
    """Return the feature as ordered, labeled bins.

    Numeric features are cut on the registry's domain-meaningful edges (or the
    caller's override); binary and categorical features are used as-is.
    """
    feat: Feature = FEATURES[name]
    col = df[name]

    if feat.kind == "binary":
        return pd.Categorical(
            np.where(col == 1, "Yes", "No"), categories=["No", "Yes"], ordered=True
        )
    if feat.kind == "categorical":
        cats = sorted(col.dropna().unique().tolist())
        return pd.Categorical(col, categories=cats, ordered=False)

    cuts = list(edges or feat.edges or np.linspace(col.min(), col.max(), 6))
    # Widen the outer edges so no observation falls outside the bins.
    cuts[0] = min(cuts[0], float(col.min()) - 1e-9)
    cuts[-1] = max(cuts[-1], float(col.max()) + 1e-9)
    return pd.cut(col, bins=cuts, include_lowest=True, precision=0)


def wilson_ci(k: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """95% Wilson score interval for a proportion.

    Preferred over the normal approximation because fraud rates sit near zero,
    where the normal interval runs below 0 and understates uncertainty.
    """
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n > 0, k / n, 0.0)
        denom = 1 + Z**2 / n
        center = (p + Z**2 / (2 * n)) / denom
        half = (Z / denom) * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2))
    lo = np.clip(np.where(n > 0, center - half, 0.0), 0, 1)
    hi = np.clip(np.where(n > 0, center + half, 0.0), 0, 1)
    return lo, hi


# --------------------------------------------------------------------------- #
# Per-feature signal table
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def bin_table(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Contingency statistics for every bin of one feature.

    Columns: bin, n, frauds, legit, share, fraud_rate, ci_lo, ci_hi, lift, woe,
    iv_part, reliable.
    """
    bins = bin_series(df, name)
    g = df.assign(_bin=bins).groupby("_bin", observed=True)[TARGET].agg(["count", "sum"])
    g.columns = ["n", "frauds"]
    g = g[g["n"] > 0].reset_index().rename(columns={"_bin": "bin"})
    g["bin"] = g["bin"].astype(str)
    g["legit"] = g["n"] - g["frauds"]

    total_n = int(df.shape[0])
    base_rate = float(df[TARGET].mean()) if total_n else 0.0

    g["share"] = g["n"] / total_n if total_n else 0.0
    g["fraud_rate"] = np.where(g["n"] > 0, g["frauds"] / g["n"], 0.0)
    g["ci_lo"], g["ci_hi"] = wilson_ci(g["frauds"].to_numpy(), g["n"].to_numpy())
    g["lift"] = g["fraud_rate"] / base_rate if base_rate > 0 else np.nan

    # Haldane-corrected distributions keep WoE finite when a bin has no fraud.
    f_adj = g["frauds"] + HALDANE
    l_adj = g["legit"] + HALDANE
    p_f = f_adj / f_adj.sum()
    p_l = l_adj / l_adj.sum()
    g["woe"] = np.log(p_f / p_l)
    g["iv_part"] = (p_f - p_l) * g["woe"]

    g["reliable"] = g["frauds"] >= MIN_FRAUDS
    return g


def information_value(bt: pd.DataFrame, strict: bool = True) -> float:
    """Sum the per-bin IV contributions.

    strict=True (the default) drops bins with fewer than MIN_FRAUDS frauds, so a
    feature is never credited with predictive power that rests on three rows.
    """
    rows = bt[bt["reliable"]] if strict else bt
    return float(rows["iv_part"].sum())


def iv_band(iv: float) -> str:
    """Conventional credit-scoring interpretation bands for Information Value.

    On real portfolio data an IV above 0.5 usually means target leakage rather
    than a great predictor, and is worth investigating. This file is synthetic
    with signal planted deliberately, so several features land there honestly -
    hence "EXTREME" rather than the textbook label "suspicious".
    """
    if iv < 0.02:
        return "USELESS"
    if iv < 0.10:
        return "WEAK"
    if iv < 0.30:
        return "MEDIUM"
    if iv < 0.50:
        return "STRONG"
    return "EXTREME"


def direction(bt: pd.DataFrame, name: str) -> str:
    """Whether risk rises or falls across the ordered bins of a numeric feature."""
    if FEATURES[name].kind != "numeric" or len(bt) < 3:
        return "-"
    rows = bt[bt["reliable"]]
    if len(rows) < 3:
        return "insufficient data"
    r = np.corrcoef(np.arange(len(rows)), rows["fraud_rate"].to_numpy())[0, 1]
    if r > 0.6:
        return "higher = riskier"
    if r < -0.6:
        return "lower = riskier"
    return "non-monotonic"


@st.cache_data(show_spinner=False)
def scan_features(df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Rank every registry feature by Information Value - the scanner core table."""
    rows = []
    for name, feat in FEATURES.items():
        bt = bin_table(df, name)
        pool = bt[bt["reliable"]] if strict else bt
        best = pool.loc[pool["lift"].idxmax()] if len(pool) else None
        iv = information_value(bt, strict)
        rows.append(
            {
                "feature": name,
                "label": feat.label,
                "kind": feat.kind,
                "iv": iv,
                "strength": iv_band(iv),
                "max_lift": float(best["lift"]) if best is not None else np.nan,
                "best_bin": str(best["bin"]) if best is not None else "-",
                "best_bin_rate": float(best["fraud_rate"]) if best is not None else np.nan,
                "best_bin_n": int(best["n"]) if best is not None else 0,
                "direction": direction(bt, name),
                "bins_flagged": int((~bt["reliable"]).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("iv", ascending=False, ignore_index=True)


# --------------------------------------------------------------------------- #
# Rule evaluation
# --------------------------------------------------------------------------- #
def rule_metrics(df: pd.DataFrame, mask: np.ndarray) -> dict:
    """Score a candidate detection rule as a binary classifier.

    Precision is what an investigations team feels (how many alerts are real),
    recall is what the loss ledger feels (how much fraud is caught). Lift says
    whether the rule beats flagging transactions at random.
    """
    y = df[TARGET].to_numpy().astype(bool)
    m = np.asarray(mask, dtype=bool)

    tp = int((m & y).sum())
    fp = int((m & ~y).sum())
    fn = int((~m & y).sum())
    tn = int((~m & ~y).sum())
    flagged = tp + fp
    total = len(df)
    base = float(y.mean()) if total else 0.0
    precision = tp / flagged if flagged else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "flagged": flagged,
        "coverage": flagged / total if total else 0.0,
        "precision": precision,
        "recall": tp / int(y.sum()) if y.sum() else 0.0,
        "lift": precision / base if base > 0 else 0.0,
        "alerts_per_1k": 1000 * flagged / total if total else 0.0,
        "f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0,
        "base_rate": base,
    }


# --------------------------------------------------------------------------- #
# WoE scorecard - the statistics-only stand-in for a model
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def woe_scorecard(df: pd.DataFrame, features: tuple[str, ...]) -> pd.Series:
    """Additive log-odds risk score per transaction.

    Each feature contributes the WoE of the bin the transaction falls into.
    Summing WoE across features is naive-Bayes: it assumes the features are
    conditionally independent given the label. That assumption is imperfect
    here - foreign_transaction and location_mismatch plainly travel together -
    so read the score as a ranking, not a calibrated probability.

    Bins are always learned on the full dataset so a transaction score does
    not change when the user moves a filter.
    """
    score = pd.Series(0.0, index=df.index)
    for name in features:
        lookup = bin_table(df, name).set_index("bin")["woe"]
        binned = pd.Series(bin_series(df, name), index=df.index).astype(str)
        score += binned.map(lookup).fillna(0.0)
    return score


@st.cache_data(show_spinner=False)
def correlations(df: pd.DataFrame) -> pd.Series:
    """Point-biserial correlation of each numeric/binary feature with the label.

    Linear-only by construction: transaction_hour scores low here despite being
    a top predictor, because its risk is a 00-05 block rather than a trend. That
    contrast is the reason the scanner ranks on IV and not on correlation.
    """
    cols = [n for n, f in FEATURES.items() if f.kind in ("numeric", "binary")]
    return df[cols + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET).sort_values()


@st.cache_data(show_spinner=False)
def interaction_grid(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """Fraud rate for every cell of a two-feature grid, long format.

    Two weak signals can combine into a strong one, which a per-feature ranking
    can never reveal on its own.
    """
    def _ordered(name: str) -> pd.Categorical:
        """Bin as strings but keep the bin order, so the grid axes read in
        domain order rather than alphabetically ("(12, 18]" before "(6, 12]")."""
        binned = pd.Categorical(bin_series(df, name))
        cats = [str(c) for c in binned.categories]
        return pd.Categorical(binned.astype(str), categories=cats, ordered=True)

    d = df.assign(_x=_ordered(x), _y=_ordered(y))
    g = d.groupby(["_y", "_x"], observed=True)[TARGET].agg(["count", "sum"]).reset_index()
    g.columns = ["y", "x", "n", "frauds"]
    g["fraud_rate"] = g["frauds"] / g["n"]
    base = float(df[TARGET].mean())
    g["lift"] = g["fraud_rate"] / base if base else np.nan
    g["reliable"] = g["frauds"] >= MIN_FRAUDS
    return g
