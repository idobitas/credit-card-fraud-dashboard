"""FRAUD SIGNAL CONSOLE - an interactive console for finding the transaction
characteristics that predict credit-card fraud.

Run with:  streamlit run app.py

Chart conventions used throughout (see src/theme.py for the validated palette):
  * Fraud is always red, legitimate always cyan, and the pair is validated for
    colorblind separation against the dark surface.
  * No chart uses two y-axes. Where a rate and a volume both matter, the rate is
    plotted and the volume is carried in the hover and the labels.
  * A bin whose fraud count is below the reliability threshold is marked in the
    label with a warning glyph, never by color alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import analytics as A
from src import data as D
from src import theme as T

st.set_page_config(
    page_title="Fraud Signal Console",
    page_icon="\N{SHIELD}",
    layout="wide",
    initial_sidebar_state="expanded",
)
T.inject_css(st)

df = D.load_data()
FULL_BOUNDS = D.bounds(df)
ALL_CATEGORIES = sorted(df["merchant_category"].unique().tolist())
BASE_RATE = float(df[D.TARGET].mean())
TOTAL_FRAUDS = int(df[D.TARGET].sum())

# The ramp stops short of its lightest step so a single light text colour stays
# readable on every cell - Plotly cannot set text colour per cell.
_RAMP = T.SEQ_GREEN[:-1]
SEQ_SCALE = [[i / (len(_RAMP) - 1), c] for i, c in enumerate(_RAMP)]


# --------------------------------------------------------------------------- #
# Sidebar - filters
# --------------------------------------------------------------------------- #
def reset_filters() -> None:
    """Drop every filter widget key so each control falls back to its default."""
    for key in [k for k in st.session_state if k.startswith("f_")]:
        del st.session_state[key]


with st.sidebar:
    st.markdown('<div class="term-head">&gt; FILTERS</div>', unsafe_allow_html=True)

    state: dict = {}
    for col in D.NUMERIC:
        lo, hi = FULL_BOUNDS[col]
        whole = df[col].dtype.kind in "iu"
        state[col] = st.slider(
            D.FEATURES[col].label,
            min_value=lo, max_value=hi, value=(lo, hi),
            step=1.0 if whole else 0.01,
            format="%d" if whole else "$%.2f",
            key=f"f_{col}",
        )
        sel_lo, sel_hi = state[col]
        fmt = (lambda v: f"{v:,.0f}") if whole else (lambda v: f"${v:,.2f}")
        st.markdown(
            f'<div class="range-note">{fmt(sel_lo)} &rarr; {fmt(sel_hi)}'
            f'{" (full range)" if (sel_lo, sel_hi) == (lo, hi) else ""}</div>',
            unsafe_allow_html=True,
        )

    state["merchant_category"] = st.multiselect(
        "Merchant category", ALL_CATEGORIES, default=ALL_CATEGORIES, key="f_cat"
    )

    for col in D.BINARY:
        state[col] = st.radio(
            D.FEATURES[col].label, ["All", "Yes", "No"], horizontal=True, key=f"f_{col}"
        )

    state["outcome"] = st.radio(
        "Outcome", ["All", "Fraud only", "Legit only"], horizontal=True, key="f_outcome"
    )

    st.button("RESET FILTERS", on_click=reset_filters, width="stretch")

view = D.apply_filters(df, state)
n_view, f_view = len(view), int(view[D.TARGET].sum())

with st.sidebar:
    st.markdown(
        f'<div class="term-sub">{n_view:,} / {len(df):,} records &middot; '
        f'{f_view} frauds</div>',
        unsafe_allow_html=True,
    )

st.markdown("# FRAUD SIGNAL CONSOLE")
st.markdown(
    f'<div class="term-sub">credit_card_fraud_10k.csv &middot; {len(df):,} transactions '
    f'&middot; {TOTAL_FRAUDS} confirmed frauds &middot; base rate {BASE_RATE:.2%} '
    f'&middot; statistics-only engine (no model library)</div>',
    unsafe_allow_html=True,
)

tab_over, tab_scan, tab_rule, tab_inter, tab_txn = st.tabs(
    ["OVERVIEW", "SIGNAL SCANNER", "RULE BUILDER", "INTERACTIONS", "TRANSACTIONS"]
)

EMPTY_MSG = "No transactions match the current filter. Widen it or press RESET FILTERS."


# --------------------------------------------------------------------------- #
# Tab 1 - Overview
# --------------------------------------------------------------------------- #
with tab_over:
    T.head(st, "OVERVIEW", "Where fraud sits in the filtered slice, against the "
                           f"{BASE_RATE:.2%} portfolio baseline.")

    if view.empty:
        st.warning(EMPTY_MSG)
    else:
        rate = f_view / n_view
        cols = st.columns(6)
        with cols[0]:
            T.kpi(st, "Transactions", f"{n_view:,}", f"{n_view / len(df):.1%} of file")
        with cols[1]:
            T.kpi(st, "Frauds", f"{f_view}", f"{f_view}/{TOTAL_FRAUDS} of all fraud",
                  tone="alert")
        with cols[2]:
            T.kpi(st, "Fraud rate", f"{rate:.2%}", f"baseline {BASE_RATE:.2%}",
                  tone="alert" if rate > BASE_RATE else "calm")
        with cols[3]:
            T.kpi(st, "Lift vs baseline", f"{rate / BASE_RATE:.2f}x",
                  "1.00x = no better than random")
        with cols[4]:
            T.kpi(st, "Fraud amount", f"${view.loc[view[D.TARGET] == 1, 'amount'].sum():,.0f}",
                  "value exposed in this slice", tone="alert")
        with cols[5]:
            T.kpi(st, "Fraud recall", f"{f_view / TOTAL_FRAUDS:.1%}",
                  "share of all fraud captured")

        st.markdown("")
        left, right = st.columns([3, 2])

        with left:
            hourly = (
                view.groupby("transaction_hour")[D.TARGET]
                .agg(["count", "sum"]).reindex(range(24), fill_value=0).reset_index()
            )
            hourly["rate"] = np.where(hourly["count"] > 0,
                                      hourly["sum"] / hourly["count"], np.nan)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hourly["transaction_hour"], y=hourly["rate"],
                mode="lines+markers", name="Fraud rate",
                line=dict(color=T.FRAUD, width=2),
                marker=dict(size=8, color=T.FRAUD,
                            line=dict(color=T.PANEL, width=2)),
                customdata=np.stack([hourly["sum"], hourly["count"]], axis=-1),
                hovertemplate=("<b>%{x:02d}:00</b><br>fraud rate %{y:.2%}"
                               "<br>%{customdata[0]} of %{customdata[1]} txns"
                               "<extra></extra>"),
            ))
            fig.add_hline(y=BASE_RATE, line=dict(color=T.MUTED, width=1, dash="dot"),
                          annotation_text=f"baseline {BASE_RATE:.2%}",
                          annotation_font=dict(color=T.MUTED, size=10))
            if hourly["rate"].notna().any():
                peak = hourly.loc[hourly["rate"].idxmax()]
                fig.add_annotation(
                    x=peak["transaction_hour"], y=peak["rate"],
                    text=f"peak {peak['rate']:.1%} @ {int(peak['transaction_hour']):02d}:00",
                    showarrow=True, arrowcolor=T.MUTED, arrowhead=0, ay=-28,
                    font=dict(color=T.TEXT, size=11),
                )
            T.style_fig(fig, height=330, legend=False,
                        title="Fraud rate by hour of day", hovermode="x unified")
            fig.update_xaxes(title_text="hour", dtick=2, range=[-0.5, 23.5])
            fig.update_yaxes(title_text="fraud rate", tickformat=".1%", rangemode="tozero")
            st.plotly_chart(fig, width="stretch", theme=None)
            st.caption("The overnight window is the single clearest split in this file: "
                       "risk collapses once normal trading hours begin.")

        with right:
            cat = (
                view.groupby("merchant_category", observed=True)[D.TARGET]
                .agg(["count", "sum"]).reset_index()
            )
            cat["rate"] = cat["sum"] / cat["count"]
            cat = cat.sort_values("rate")
            fig = go.Figure(go.Bar(
                x=cat["rate"], y=cat["merchant_category"], orientation="h",
                marker=dict(color=T.FRAUD, line=dict(color=T.PANEL, width=2)),
                text=[f"{r:.2%}  (n={n:,})" for r, n in zip(cat["rate"], cat["count"])],
                textposition="outside", textfont=dict(color=T.TEXT, size=11),
                customdata=np.stack([cat["sum"], cat["count"]], axis=-1),
                hovertemplate=("<b>%{y}</b><br>fraud rate %{x:.2%}"
                               "<br>%{customdata[0]} of %{customdata[1]} txns"
                               "<extra></extra>"),
            ))
            fig.add_vline(x=BASE_RATE, line=dict(color=T.MUTED, width=1, dash="dot"))
            T.style_fig(fig, height=330, legend=False,
                        title="Fraud rate by merchant category")
            # Headroom on the right so the outside value labels are not clipped.
            fig.update_xaxes(title_text="fraud rate", tickformat=".1%",
                             range=[0, max(cat["rate"].max() * 2.1, BASE_RATE * 2)])
            st.plotly_chart(fig, width="stretch", theme=None)
            st.caption("Volume is in the labels rather than on a second axis - "
                       "the spread here is small enough to be mostly sampling noise.")

        fraud_amt = view.loc[view[D.TARGET] == 1, "amount"]
        legit_amt = view.loc[view[D.TARGET] == 0, "amount"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=legit_amt, name="Legitimate", histnorm="percent", nbinsx=45,
            marker=dict(color=T.LEGIT, line=dict(color=T.PANEL, width=1)),
            opacity=0.85,
            hovertemplate="<b>Legitimate</b><br>$%{x}<br>%{y:.1f}% of legit<extra></extra>",
        ))
        fig.add_trace(go.Histogram(
            x=fraud_amt, name="Fraud", histnorm="percent", nbinsx=45,
            marker=dict(color=T.FRAUD, line=dict(color=T.PANEL, width=1),
                        pattern=dict(shape="/", fgcolor=T.PANEL, size=4)),
            opacity=0.85,
            hovertemplate="<b>Fraud</b><br>$%{x}<br>%{y:.1f}% of fraud<extra></extra>",
        ))
        T.style_fig(fig, height=300, title="Amount distribution, each class to 100%",
                    barmode="overlay", bargap=0.04)
        fig.update_xaxes(title_text="amount ($)")
        fig.update_yaxes(title_text="% within class")
        st.plotly_chart(fig, width="stretch", theme=None)
        st.caption("Each series is normalised to its own 100% because fraud is only "
                   "1.5% of rows - on raw counts the red series would be invisible. "
                   "Fraud is hatched as well as red.")


# --------------------------------------------------------------------------- #
# Tab 2 - Signal scanner
# --------------------------------------------------------------------------- #
with tab_scan:
    T.head(st, "SIGNAL_SCANNER",
           "Every feature ranked by Information Value - how much knowing it moves "
           "the odds of fraud. Select a row to open its bin-level detail.")

    if view.empty:
        st.warning(EMPTY_MSG)
    else:
        c1, c2 = st.columns([1, 3])
        with c1:
            strict = st.toggle(
                "Hide low-confidence bins", value=True, key="scan_strict",
                help=f"Exclude bins holding fewer than {A.MIN_FRAUDS} frauds from the "
                     "IV total and from the best-bin pick. With only 151 frauds in the "
                     "file, thin bins produce impressive-looking lift that is noise.",
            )
        with c2:
            st.markdown(
                '<div class="term-sub">IV bands: &lt;0.02 useless &middot; 0.02-0.10 weak '
                '&middot; 0.10-0.30 medium &middot; 0.30-0.50 strong &middot; &gt;0.50 '
                'extreme (on real data, check for leakage; here the signal is planted '
                'by design).</div>', unsafe_allow_html=True)

        scan = A.scan_features(view, strict)
        # NumberColumn's "%%" format appends a percent sign without rescaling, so
        # fractions have to be converted to percent units before display.
        shown = scan.assign(
            best_bin_rate=scan["best_bin_rate"] * 100,
            flagged=np.where(scan["bins_flagged"] > 0,
                             scan["bins_flagged"].astype(str) + " low-n", "-"),
        )
        event = st.dataframe(
            shown[["label", "kind", "iv", "strength", "max_lift", "best_bin",
                   "best_bin_rate", "best_bin_n", "direction", "flagged"]],
            column_config={
                "label": st.column_config.TextColumn("Feature", width="medium"),
                "kind": st.column_config.TextColumn("Type"),
                "iv": st.column_config.NumberColumn("Info value", format="%.3f"),
                "strength": st.column_config.TextColumn("Strength"),
                "max_lift": st.column_config.NumberColumn("Best lift", format="%.2fx"),
                "best_bin": st.column_config.TextColumn("Riskiest bin", width="medium"),
                "best_bin_rate": st.column_config.NumberColumn("Bin rate", format="%.2f%%"),
                "best_bin_n": st.column_config.NumberColumn("Bin n", format="%d"),
                "direction": st.column_config.TextColumn("Direction"),
                "flagged": st.column_config.TextColumn("Warnings"),
            },
            hide_index=True, width="stretch", height=330,
            on_select="rerun", selection_mode="single-row", key="scan_table",
        )
        st.caption("Click any column header to re-sort. Ranked on IV rather than "
                   "correlation because correlation only sees straight lines - see "
                   "the INTERACTIONS tab for the contrast.")

        rows = event.selection.rows if event and event.selection else []
        sel = scan.iloc[rows[0]]["feature"] if rows else scan.iloc[0]["feature"]
        bt = A.bin_table(view, sel)
        label = D.FEATURES[sel].label

        st.markdown("---")
        T.head(st, f"DETAIL :: {sel}",
               f"{label} broken into bins. Error bars are 95% Wilson intervals - "
               "where they overlap, the difference between two bins is not established.")

        bt = bt.copy()
        bt["tag"] = np.where(bt["reliable"], bt["bin"], "⚠ " + bt["bin"])
        d1, d2 = st.columns(2)

        with d1:
            fig = go.Figure(go.Bar(
                x=bt["tag"], y=bt["fraud_rate"],
                marker=dict(color=np.where(bt["reliable"], T.FRAUD, T.AMBER),
                            line=dict(color=T.PANEL, width=2)),
                error_y=dict(type="data", symmetric=False,
                             array=bt["ci_hi"] - bt["fraud_rate"],
                             arrayminus=bt["fraud_rate"] - bt["ci_lo"],
                             color=T.MUTED, thickness=1, width=6),
                customdata=np.stack([bt["frauds"], bt["n"], bt["ci_lo"], bt["ci_hi"]], -1),
                hovertemplate=("<b>%{x}</b><br>fraud rate %{y:.2%}"
                               "<br>%{customdata[0]} of %{customdata[1]} txns"
                               "<br>95% CI %{customdata[2]:.2%} - %{customdata[3]:.2%}"
                               "<extra></extra>"),
                showlegend=False,
            ))
            fig.add_hline(y=BASE_RATE, line=dict(color=T.MUTED, width=1, dash="dot"),
                          annotation_text="baseline",
                          annotation_font=dict(color=T.MUTED, size=10))
            T.style_fig(fig, height=320, legend=False, title="Fraud rate per bin")
            fig.update_yaxes(tickformat=".1%", rangemode="tozero")
            st.plotly_chart(fig, width="stretch", theme=None)
            if (~bt["reliable"]).any():
                st.caption(f"⚠ marks bins with fewer than {A.MIN_FRAUDS} frauds - "
                           "amber and flagged in the label, so the warning survives "
                           "in greyscale.")

        with d2:
            fig = go.Figure(go.Bar(
                x=bt["tag"], y=bt["woe"],
                marker=dict(color=np.where(bt["woe"] >= 0, T.FRAUD, T.LEGIT),
                            line=dict(color=T.PANEL, width=2)),
                text=[f"{v:+.2f}" for v in bt["woe"]],
                textposition="outside", textfont=dict(color=T.TEXT, size=11),
                hovertemplate="<b>%{x}</b><br>WoE %{y:+.3f}<extra></extra>",
                showlegend=False,
            ))
            fig.add_hline(y=0, line=dict(color=T.MUTED, width=1))
            T.style_fig(fig, height=320, legend=False,
                        title="Weight of evidence (log-odds shift vs the average)")
            st.plotly_chart(fig, width="stretch", theme=None)
            st.caption("Above zero the bin carries more fraud than its share of "
                       "volume; below zero it is safer than average. These are the "
                       "weights the risk score on the TRANSACTIONS tab adds up.")


# --------------------------------------------------------------------------- #
# Tab 3 - Rule builder
# --------------------------------------------------------------------------- #
with tab_rule:
    T.head(st, "RULE_BUILDER",
           "The sidebar filter read as a detection rule and scored against the whole "
           "file. The Outcome filter is ignored here - a rule may not reference the "
           "label it is trying to predict.")

    mask = D.rule_mask(df, state)
    m = A.rule_metrics(df, mask)
    rule_text = D.describe_rule(state, FULL_BOUNDS)

    st.code(f"FLAG IF  {rule_text}", language="sql")

    k = st.columns(6)
    with k[0]:
        T.kpi(st, "Precision", f"{m['precision']:.1%}",
              f"{m['tp']} real of {m['flagged']:,} alerts",
              tone="alert" if m["precision"] < BASE_RATE else "")
    with k[1]:
        T.kpi(st, "Recall", f"{m['recall']:.1%}", f"{m['tp']} of {TOTAL_FRAUDS} frauds caught")
    with k[2]:
        T.kpi(st, "Lift", f"{m['lift']:.2f}x", "vs flagging at random")
    with k[3]:
        T.kpi(st, "Alert load", f"{m['alerts_per_1k']:.0f}", "alerts per 1,000 txns")
    with k[4]:
        T.kpi(st, "Missed fraud", f"{m['fn']}", "false negatives", tone="alert")
    with k[5]:
        T.kpi(st, "F1", f"{m['f1']:.3f}", "precision/recall balance")

    st.markdown("")
    cm, pin = st.columns([2, 3])

    with cm:
        st.markdown(
            f"""<table style="width:100%;border-collapse:collapse;font-size:.8rem">
<tr><td></td>
    <th style="color:{T.MUTED};padding:.4rem">actually fraud</th>
    <th style="color:{T.MUTED};padding:.4rem">actually legit</th></tr>
<tr><th style="color:{T.MUTED};padding:.4rem;text-align:right">flagged</th>
    <td style="border:1px solid {T.BORDER};padding:.6rem;text-align:center;
        color:{T.NEON};font-size:1.1rem">{m['tp']}<br>
        <span style="color:{T.MUTED};font-size:.65rem">caught</span></td>
    <td style="border:1px solid {T.BORDER};padding:.6rem;text-align:center;
        color:{T.AMBER};font-size:1.1rem">{m['fp']:,}<br>
        <span style="color:{T.MUTED};font-size:.65rem">false alarms</span></td></tr>
<tr><th style="color:{T.MUTED};padding:.4rem;text-align:right">not flagged</th>
    <td style="border:1px solid {T.BORDER};padding:.6rem;text-align:center;
        color:{T.FRAUD};font-size:1.1rem">{m['fn']}<br>
        <span style="color:{T.MUTED};font-size:.65rem">missed</span></td>
    <td style="border:1px solid {T.BORDER};padding:.6rem;text-align:center;
        color:{T.MUTED};font-size:1.1rem">{m['tn']:,}<br>
        <span style="color:{T.MUTED};font-size:.65rem">correctly cleared</span></td></tr>
</table>""",
            unsafe_allow_html=True,
        )
        st.caption("Every rule trades these four cells against each other. Tightening "
                   "for precision moves fraud from 'caught' into 'missed'.")

    with pin:
        st.session_state.setdefault("pinned", [])
        b1, b2 = st.columns(2)
        with b1:
            if st.button("PIN THIS RULE", width="stretch"):
                st.session_state["pinned"].append({
                    "rule": rule_text, "precision": m["precision"], "recall": m["recall"],
                    "lift": m["lift"], "alerts/1k": m["alerts_per_1k"],
                    "caught": m["tp"], "missed": m["fn"], "f1": m["f1"],
                })
        with b2:
            if st.button("CLEAR PINNED", width="stretch"):
                st.session_state["pinned"] = []

        if st.session_state["pinned"]:
            pinned = pd.DataFrame(st.session_state["pinned"])
            pinned["precision"] *= 100
            pinned["recall"] *= 100
            st.dataframe(
                pinned,
                column_config={
                    "rule": st.column_config.TextColumn("Rule", width="large"),
                    "precision": st.column_config.NumberColumn("Prec.", format="%.1f%%"),
                    "recall": st.column_config.NumberColumn("Recall", format="%.1f%%"),
                    "lift": st.column_config.NumberColumn("Lift", format="%.2fx"),
                    "alerts/1k": st.column_config.NumberColumn("Alerts/1k", format="%.0f"),
                    "caught": st.column_config.NumberColumn("Caught", format="%d"),
                    "missed": st.column_config.NumberColumn("Missed", format="%d"),
                    "f1": st.column_config.NumberColumn("F1", format="%.3f"),
                },
                hide_index=True, width="stretch", height=210,
            )
        else:
            st.info("Pin two or three rules to compare them side by side. Try "
                    "`hour <= 5` alone, then add `device_trust_score <= 40`.")

    st.markdown("---")
    T.head(st, "COMBINED SCORECARD",
           "Instead of one rule, add up the weight of evidence of every feature and "
           "flag the riskiest transactions by score.")

    score = A.woe_scorecard(df, tuple(D.FEATURES))
    lo, hi = float(score.min()), float(score.max())
    thr = st.slider("Flag transactions scoring above", lo, hi, float(np.quantile(score, 0.95)),
                    step=(hi - lo) / 200, key="score_thr")
    sm = A.rule_metrics(df, (score > thr).to_numpy())

    s1, s2 = st.columns([2, 3])
    with s1:
        sk = st.columns(2)
        with sk[0]:
            T.kpi(st, "Precision", f"{sm['precision']:.1%}",
                  f"{sm['tp']} real of {sm['flagged']:,} alerts")
            T.kpi(st, "Lift", f"{sm['lift']:.2f}x", "vs random")
        with sk[1]:
            T.kpi(st, "Recall", f"{sm['recall']:.1%}",
                  f"{sm['tp']} of {TOTAL_FRAUDS} frauds")
            T.kpi(st, "Alert load", f"{sm['alerts_per_1k']:.0f}", "per 1,000 txns")
    with s2:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=score[df[D.TARGET] == 0], name="Legitimate", histnorm="percent", nbinsx=50,
            marker=dict(color=T.LEGIT, line=dict(color=T.PANEL, width=1)), opacity=0.85,
            hovertemplate="<b>Legitimate</b><br>score %{x:.1f}<br>%{y:.1f}%<extra></extra>",
        ))
        fig.add_trace(go.Histogram(
            x=score[df[D.TARGET] == 1], name="Fraud", histnorm="percent", nbinsx=50,
            marker=dict(color=T.FRAUD, line=dict(color=T.PANEL, width=1),
                        pattern=dict(shape="/", fgcolor=T.PANEL, size=4)), opacity=0.85,
            hovertemplate="<b>Fraud</b><br>score %{x:.1f}<br>%{y:.1f}%<extra></extra>",
        ))
        fig.add_vline(x=thr, line=dict(color=T.NEON, width=2),
                      annotation_text="threshold",
                      annotation_font=dict(color=T.NEON, size=10))
        T.style_fig(fig, height=270, title="Risk score by outcome",
                    barmode="overlay", bargap=0.04)
        fig.update_xaxes(title_text="summed weight of evidence (log-odds)")
        fig.update_yaxes(title_text="% within class")
        st.plotly_chart(fig, width="stretch", theme=None)

    st.caption("Adding WoE across features assumes they are independent given the "
               "outcome, which foreign_transaction and location_mismatch clearly are "
               "not - they fire together. Read the score as a ranking, not a "
               "probability.")


# --------------------------------------------------------------------------- #
# Tab 4 - Interactions
# --------------------------------------------------------------------------- #
with tab_inter:
    T.head(st, "INTERACTIONS",
           "Two weak features can be strong together. A per-feature ranking cannot "
           "show that; this grid can.")

    if view.empty:
        st.warning(EMPTY_MSG)
    else:
        i1, i2 = st.columns([2, 3])

        with i1:
            corr = A.correlations(view)
            fig = go.Figure(go.Bar(
                x=corr.to_numpy(), y=[D.FEATURES[c].label for c in corr.index],
                orientation="h",
                marker=dict(color=np.where(corr.to_numpy() >= 0, T.FRAUD, T.LEGIT),
                            line=dict(color=T.PANEL, width=2)),
                text=[f"{v:+.3f}" for v in corr], textposition="outside",
                textfont=dict(color=T.TEXT, size=11),
                hovertemplate="<b>%{y}</b><br>r = %{x:+.3f}<extra></extra>",
                showlegend=False,
            ))
            fig.add_vline(x=0, line=dict(color=T.MUTED, width=1))
            T.style_fig(fig, height=380, legend=False,
                        title="Linear correlation with is_fraud")
            fig.update_xaxes(range=[min(corr.min() * 2.2, -0.05), max(corr.max() * 1.7, 0.05)])
            st.plotly_chart(fig, width="stretch", theme=None)
            st.caption("Red = more fraud as the value rises, cyan = less. Note how low "
                       "transaction_hour scores despite topping the scanner: its risk "
                       "is a block of hours, not a trend, and correlation cannot see "
                       "shapes like that.")

        with i2:
            names = list(D.FEATURES)
            labels = {n: D.FEATURES[n].label for n in names}
            g1, g2 = st.columns(2)
            with g1:
                xf = st.selectbox("Columns", names, index=names.index("transaction_hour"),
                                  format_func=labels.get, key="ix")
            with g2:
                yf = st.selectbox("Rows", names, index=names.index("device_trust_score"),
                                  format_func=labels.get, key="iy")

            if xf == yf:
                st.info("Pick two different features to see how they combine.")
            else:
                grid = A.interaction_grid(view, xf, yf)
                piv = grid.pivot(index="y", columns="x", values="fraud_rate")
                n_piv = grid.pivot(index="y", columns="x", values="n")
                rel = grid.pivot(index="y", columns="x", values="reliable")
                text = np.where(
                    rel.fillna(False).to_numpy(),
                    np.vectorize(lambda v: "-" if pd.isna(v) else f"{v:.1%}")(piv.to_numpy()),
                    "⚠",
                )
                fig = go.Figure(go.Heatmap(
                    z=piv.to_numpy(), x=[str(c) for c in piv.columns],
                    y=[str(i) for i in piv.index],
                    colorscale=SEQ_SCALE, colorbar=dict(
                        title=dict(text="fraud rate", font=dict(color=T.MUTED, size=11)),
                        tickfont=dict(color=T.MUTED, size=10), tickformat=".1%",
                        outlinecolor=T.BORDER, thickness=12),
                    xgap=2, ygap=2,
                    text=text, texttemplate="%{text}",
                    textfont=dict(family=T.FONT, size=11, color=T.TEXT),
                    customdata=np.dstack([n_piv.to_numpy()]),
                    hovertemplate=("%{y} &times; %{x}<br>fraud rate %{z:.2%}"
                                   "<br>n = %{customdata[0]}<extra></extra>"),
                ))
                T.style_fig(fig, height=380, legend=False,
                            title=f"Fraud rate: {labels[yf]} against {labels[xf]}")
                fig.update_xaxes(title_text=labels[xf])
                fig.update_yaxes(title_text=labels[yf])
                st.plotly_chart(fig, width="stretch", theme=None)
                st.caption(f"⚠ replaces the rate in cells holding fewer than "
                           f"{A.MIN_FRAUDS} frauds. The default pairing shows the "
                           "sharpest combination in the file: an overnight "
                           "transaction from a low-trust device.")


# --------------------------------------------------------------------------- #
# Tab 5 - Transactions
# --------------------------------------------------------------------------- #
with tab_txn:
    T.head(st, "TRANSACTIONS",
           "The filtered rows, with the weight-of-evidence risk score attached. "
           "Click any column header to sort.")

    if view.empty:
        st.warning(EMPTY_MSG)
    else:
        score_all = A.woe_scorecard(df, tuple(D.FEATURES))
        out = view.copy()
        out.insert(1, "risk_score", score_all.reindex(out.index).round(2))
        # np.where returns an ndarray, so build the joined string as a Series.
        out["flags"] = pd.Series(
            np.where(out["transaction_hour"] <= 5, "NIGHT ", "")
            + np.where(out["foreign_transaction"] == 1, "FOREIGN ", "")
            + np.where(out["location_mismatch"] == 1, "GEO ", "")
            + np.where(out["device_trust_score"] < 40, "LOWTRUST ", "")
            + np.where(out["velocity_last_24h"] >= 4, "VELOCITY", ""),
            index=out.index,
        ).str.strip()
        out["outcome"] = np.where(out[D.TARGET] == 1, "FRAUD", "legit")

        sort_col = st.selectbox(
            "Sort by", ["risk_score", "amount", "device_trust_score",
                        "velocity_last_24h", "transaction_hour", "transaction_id"],
            key="txn_sort",
        )
        out = out.sort_values(sort_col, ascending=False)

        st.dataframe(
            out.drop(columns=[D.TARGET]),
            column_config={
                "transaction_id": st.column_config.NumberColumn("ID", format="%d"),
                "risk_score": st.column_config.NumberColumn("Risk score", format="%.2f"),
                "amount": st.column_config.NumberColumn("Amount", format="$%.2f"),
                "transaction_hour": st.column_config.NumberColumn("Hour", format="%d"),
                "merchant_category": st.column_config.TextColumn("Category"),
                "foreign_transaction": st.column_config.NumberColumn("Foreign", format="%d"),
                "location_mismatch": st.column_config.NumberColumn("Geo mismatch", format="%d"),
                "device_trust_score": st.column_config.NumberColumn("Device trust", format="%d"),
                "velocity_last_24h": st.column_config.NumberColumn("Velocity", format="%d"),
                "cardholder_age": st.column_config.NumberColumn("Age", format="%d"),
                "flags": st.column_config.TextColumn("Triggered flags", width="medium"),
                "outcome": st.column_config.TextColumn("Outcome"),
            },
            hide_index=True, width="stretch", height=460,
        )
        st.download_button(
            "DOWNLOAD THIS SLICE (CSV)",
            out.to_csv(index=False).encode("utf-8"),
            file_name="fraud_slice.csv", mime="text/csv",
        )
