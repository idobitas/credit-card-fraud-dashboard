"""Data loading, the feature registry, and the single shared filter function.

The FEATURES registry is the one place a column is described. The sidebar, the
signal scanner, the interaction grid and the scorecard all read from it, so
adding a column to the dataset means editing exactly one dict entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import streamlit as st

CSV_PATH = Path(__file__).resolve().parent.parent / "credit_card_fraud_10k.csv"

TARGET = "is_fraud"
ID_COL = "transaction_id"


@dataclass(frozen=True)
class Feature:
    """Describes one predictor column.

    kind:  "numeric" | "binary" | "categorical" - drives widget and binning choice.
    edges: explicit bin edges for numeric features. Domain-meaningful cut points
           beat equal-width bins here: 00-05 is the fraud window, and device
           trust collapses in the low band, so both get hand-chosen edges.
    """

    name: str
    label: str
    kind: str
    unit: str = ""
    edges: list[float] | None = field(default=None)


FEATURES: dict[str, Feature] = {
    "amount": Feature("amount", "Amount", "numeric", "$", [0, 50, 100, 200, 400, 800, 1e9]),
    "transaction_hour": Feature(
        "transaction_hour", "Hour of day", "numeric", "h", [-0.5, 5.5, 11.5, 17.5, 23.5]
    ),
    "merchant_category": Feature("merchant_category", "Merchant category", "categorical"),
    "foreign_transaction": Feature("foreign_transaction", "Foreign transaction", "binary"),
    "location_mismatch": Feature("location_mismatch", "Location mismatch", "binary"),
    "device_trust_score": Feature(
        "device_trust_score", "Device trust score", "numeric", "", [0, 20, 40, 60, 80, 100]
    ),
    "velocity_last_24h": Feature(
        "velocity_last_24h", "Velocity (24h)", "numeric", "txn", [-0.5, 0.5, 1.5, 2.5, 3.5, 100]
    ),
    "cardholder_age": Feature(
        "cardholder_age", "Cardholder age", "numeric", "y", [0, 25, 35, 45, 55, 120]
    ),
}

NUMERIC = [f.name for f in FEATURES.values() if f.kind == "numeric"]
BINARY = [f.name for f in FEATURES.values() if f.kind == "binary"]
CATEGORICAL = [f.name for f in FEATURES.values() if f.kind == "categorical"]


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Read the transaction file. Path is resolved from __file__, not the CWD,
    so the app runs the same from any working directory."""
    df = pd.read_csv(CSV_PATH)
    missing = ({TARGET, ID_COL} | set(FEATURES)) - set(df.columns)
    if missing:
        raise ValueError(f"{CSV_PATH.name} is missing columns: {sorted(missing)}")
    return df


def bounds(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Min/max per numeric feature, used to initialise and reset the sliders."""
    return {c: (float(df[c].min()), float(df[c].max())) for c in NUMERIC}


def apply_filters(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    """Reduce the frame to the sidebar selection.

    Every tab reads its data through this one function, so a filter can never
    apply on one tab and silently not on another.
    """
    mask = pd.Series(True, index=df.index)

    for col in NUMERIC:
        lo, hi = state[col]
        mask &= df[col].between(lo, hi)

    cats = state.get("merchant_category")
    if cats:
        mask &= df["merchant_category"].isin(cats)

    for col in BINARY:
        choice = state.get(col, "All")
        if choice == "Yes":
            mask &= df[col] == 1
        elif choice == "No":
            mask &= df[col] == 0

    outcome = state.get("outcome", "All")
    if outcome == "Fraud only":
        mask &= df[TARGET] == 1
    elif outcome == "Legit only":
        mask &= df[TARGET] == 0

    return df[mask]


def rule_mask(df: pd.DataFrame, state: dict) -> pd.Series:
    """Boolean mask for the filter treated as a detection rule.

    Deliberately ignores the outcome filter: a rule may only reference features
    that are knowable at authorisation time, never the label it is predicting.
    """
    return df.index.isin(apply_filters(df, {**state, "outcome": "All"}).index)


def describe_rule(state: dict, full_bounds: dict[str, tuple[float, float]]) -> str:
    """Human-readable rule text, listing only the conditions that actually narrow
    the data - an untouched slider is not a condition."""
    parts: list[str] = []
    for col in NUMERIC:
        lo, hi = state[col]
        flo, fhi = full_bounds[col]
        if lo > flo or hi < fhi:
            parts.append(f"{col} in [{lo:g}, {hi:g}]")
    cats = state.get("merchant_category")
    if cats and len(cats) < 5:
        parts.append(f"category in {{{', '.join(sorted(cats))}}}")
    for col in BINARY:
        if state.get(col, "All") != "All":
            parts.append(f"{col} = {1 if state[col] == 'Yes' else 0}")
    return " AND ".join(parts) if parts else "ALL TRANSACTIONS (no conditions)"
