"""Hacker-terminal visual theme: color tokens, global CSS, and Plotly styling.

Everything visual is defined once here so the app never hand-codes a hex value.

Color policy (validated with the dataviz palette validator against the #0F1512
chart surface):
  * Chart series use CATEGORICAL = red / cyan / violet - all six checks pass.
  * Neon green is UI chrome only (text, borders, headers). Green and red are
    3.4 delta-E apart under deuteranopia, so they are never two series in one chart.
  * AMBER is a reserved status color for low-confidence bins and always ships
    alongside a text marker, never as color alone.
"""

from __future__ import annotations

# --- UI chrome tokens -------------------------------------------------------
BG = "#0B0F0B"       # app background
PANEL = "#0F1512"    # cards, sidebar, chart surface
BORDER = "#1F3D2B"   # 1px panel borders / gridlines
NEON = "#00FF41"     # primary accent: headers, KPI numbers
TEXT = "#C8F5D0"     # body text (softer than pure neon)
MUTED = "#5A7A63"    # captions, axis labels

# --- Data series tokens (validated) ----------------------------------------
FRAUD = "#FF3B3B"    # slot 1
LEGIT = "#00A0BF"    # slot 2
ACCENT = "#9B6BFF"   # slot 3 - highlights the active selection
CATEGORICAL = [FRAUD, LEGIT, ACCENT]

# --- Reserved status --------------------------------------------------------
AMBER = "#C08512"    # low-confidence / insufficient sample

# --- Sequential ramp (one hue, dark -> light) for magnitude heatmaps --------
SEQ_GREEN = ["#0C2415", "#12401F", "#17662B", "#1D9139", "#2FBC4D", "#63E683"]
# Diverging pair for signed correlations: cyan <- neutral gray -> red
DIVERGING = [LEGIT, "#3A4A40", FRAUD]

FONT = "'JetBrains Mono','Fira Code',Consolas,'Courier New',monospace"

# Rendered once per session by inject_css().
_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="st-"], button, input, textarea, select {{
    font-family: {FONT} !important;
}}
/* Force LTR: the UI is English, and an RTL browser locale otherwise flips
   the two value labels on every range slider so the max reads on the left. */
.stApp, section[data-testid="stSidebar"] {{ direction: ltr; }}
.stApp {{ background: {BG}; color: {TEXT}; }}

/* CRT scanlines - decorative only, never intercepts clicks. */
.stApp::after {{
    content: ""; position: fixed; inset: 0; z-index: 9999; pointer-events: none;
    background: repeating-linear-gradient(
        180deg, rgba(0,255,65,.035) 0 1px, transparent 1px 3px);
}}

section[data-testid="stSidebar"] {{
    background: {PANEL}; border-right: 1px solid {BORDER};
}}
header[data-testid="stHeader"] {{ background: transparent; }}

h1, h2, h3, h4 {{
    color: {NEON} !important;
    letter-spacing: .06em;
    text-shadow: 0 0 8px rgba(0,255,65,.45);
}}
h1 {{ font-size: 1.6rem !important; }}

/* Prompt-style section header, e.g. "> SIGNAL_SCANNER". */
.term-head {{
    color: {NEON}; font-size: .95rem; font-weight: 700; letter-spacing: .12em;
    text-shadow: 0 0 8px rgba(0,255,65,.45);
    border-bottom: 1px solid {BORDER};
    padding: 0 0 .35rem 0; margin: .2rem 0 .9rem 0;
}}
.term-sub {{ color: {MUTED}; font-size: .78rem; margin: -.5rem 0 .9rem 0; }}

/* KPI card: label above a large mono number, neon rule on the left. */
.kpi {{
    background: {PANEL}; border: 1px solid {BORDER}; border-left: 3px solid {NEON};
    padding: .7rem .85rem; height: 100%;
}}
.kpi-label {{
    color: {MUTED}; font-size: .68rem; letter-spacing: .14em; text-transform: uppercase;
}}
.kpi-value {{
    color: {NEON}; font-size: 1.5rem; font-weight: 700; line-height: 1.25;
    text-shadow: 0 0 10px rgba(0,255,65,.35);
}}
.kpi-value.alert {{ color: {FRAUD}; text-shadow: 0 0 10px rgba(255,59,59,.35); }}
.kpi-value.calm  {{ color: {LEGIT}; text-shadow: none; }}
.kpi-sub {{ color: {MUTED}; font-size: .7rem; }}

.stTabs [data-baseweb="tab-list"] {{ gap: .25rem; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{
    background: {PANEL}; border: 1px solid {BORDER}; border-bottom: none;
    color: {MUTED}; letter-spacing: .1em; font-size: .78rem; padding: .35rem 1rem;
}}
.stTabs [aria-selected="true"] {{ color: {NEON} !important; border-color: {NEON}; }}

div[data-testid="stMetricValue"] {{ color: {NEON}; }}
.stDataFrame, div[data-testid="stDataFrameResizable"] {{ border: 1px solid {BORDER}; }}
.stSlider label, .stMultiSelect label, .stRadio label, .stSelectbox label {{
    color: {TEXT} !important; font-size: .78rem; letter-spacing: .05em;
}}
.stButton > button, .stDownloadButton > button {{
    background: {PANEL}; color: {NEON}; border: 1px solid {NEON};
    letter-spacing: .1em; font-size: .75rem;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background: rgba(0,255,65,.12); border-color: {NEON}; color: {NEON};
}}
/* Streamlit 1.63 paints the two range-slider thumb labels in the wrong order
   (the maximum appears above the left thumb). Verified upstream - it reproduces
   with this stylesheet disabled. Hide them and print the range ourselves. */
div[data-testid="stSliderThumbValue"] {{ display: none; }}
.range-note {{
    color: {MUTED}; font-size: .7rem; letter-spacing: .04em; margin: -1.1rem 0 .6rem 0;
}}
hr {{ border-color: {BORDER}; }}
code {{ color: {NEON}; background: {PANEL}; }}
</style>
"""


def inject_css(st) -> None:
    """Apply the global stylesheet. Call once, immediately after set_page_config."""
    st.markdown(_CSS, unsafe_allow_html=True)


def head(st, title: str, subtitle: str = "") -> None:
    """Render a prompt-style section header with an optional caption."""
    st.markdown(f'<div class="term-head">&gt; {title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="term-sub">{subtitle}</div>', unsafe_allow_html=True)


def kpi(st, label: str, value: str, sub: str = "", tone: str = "") -> None:
    """Render one KPI card. tone: "" (neon), "alert" (red), or "calm" (cyan)."""
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value {tone}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height: int = 340, legend: bool = True, **layout):
    """Apply the terminal look to a Plotly figure.

    Charts default to a transparent surface so they inherit the page background
    instead of Plotly's white default - the most common dark-theme failure.
    """
    # Callers pass title= as a plain string; merge it into the styled title dict
    # rather than letting it collide with the one set here.
    title = layout.pop("title", None)
    title_cfg = dict(font=dict(color=NEON, size=13), x=0, xanchor="left")
    if isinstance(title, str):
        title_cfg["text"] = title
    elif isinstance(title, dict):
        title_cfg.update(title)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=TEXT),
        title=title_cfg,
        height=height,
        # Small fixed margins plus automargin on the axes below: the axes grow
        # the margin to fit their own labels, so category names and tick text
        # are never clipped whatever the label lengths turn out to be.
        margin=dict(l=8, r=8, t=46, b=8),
        hoverlabel=dict(
            bgcolor=PANEL, bordercolor=NEON, font=dict(family=FONT, color=TEXT, size=12)
        ),
        showlegend=legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=11),
        ),
        **layout,
    )
    axis = dict(
        gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER,
        tickfont=dict(color=MUTED, size=11),
        title=dict(font=dict(color=MUTED, size=11)),
        automargin=True,
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig
