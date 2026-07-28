"""
BMDS2003 Data Science - Deployment Prototype
Malaysia Housing Median Price Estimator (market-assisted)

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud

VISUAL REDESIGN NOTE
--------------------
This file was redesigned for a coordinated LIGHT theme (interface, CSS, layout,
copy hierarchy and chart styling only). No data-science logic was changed:
the reference-matching hierarchy, similarity weighting, model files, model
inputs, metrics and the township-level framing are exactly as before.

Paths are resolved from the application directory, so the app works no matter
what the current working directory is.

The trained pipeline receives exactly five features:
    State, Tenure, Primary_Type, Median_PSF, Transactions
Area and Township are LOOKUP CONTROLS ONLY and are never passed to the model.
"""

from __future__ import annotations

from pathlib import Path
import base64

import joblib
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Malaysia Housing Median Price Estimator",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
RESULTS_PATH = APP_DIR / "model_results.csv"
DATA_PATH = APP_DIR / "malaysia_house_price_cleaned.csv"
MODELS_DIR = APP_DIR / "models"
FIGURES_DIR = APP_DIR / "figures"
ASSETS_DIR = APP_DIR / "assets"
BANNER_PATH = ASSETS_DIR / "malaysia_housing_banner.png"

# --- Light design-system palette (single source of truth for Python + CSS) ---
BG = "#F5F7FB"          # page background
CARD = "#FFFFFF"        # card surface
SURFACE = "#F8FAFC"     # subtle neutral surface
BORDER = "#DDE4EE"      # hairline border
TEXT = "#172033"        # primary text
MUTED = "#5F6B7A"       # secondary text
PRIMARY = "#2563EB"     # primary blue - actions, navigation, information
NAVY = "#153153"        # dark navy - headings, hero
SUCCESS = "#168A5B"     # green  - successful estimate, recommended model
WARNING = "#C77700"     # amber  - caution, limited reference quality
DANGER = "#C73A4A"      # red    - errors, extrapolation, serious limitations
HIGHLIGHT = "#EAF2FF"   # soft blue highlight
NEUTRAL = "#64748B"     # grey   - supporting series in charts

# Fixed model colours, identical in every figure of the app.
MODEL_COLOURS = {
    "Decision Tree (Baseline)": NEUTRAL,
    "Decision Tree": NEUTRAL,
    "Random Forest": PRIMARY,
    "XGBoost": WARNING,
    "LightGBM": SUCCESS,
}

MODEL_FEATURES = ["State", "Tenure", "Primary_Type", "Median_PSF", "Transactions"]
ALL_AREAS = "All areas in this state"

# Similarity weighting. This is a PRESENTATION CHOICE, not an optimised
# parameter: Median_PSF carries far more predictive evidence than Transactions
# (see the correlation and permutation-importance figures), so it is weighted
# more heavily. The weights have not been statistically validated.
W_PSF, W_TXN = 0.80, 0.20


# ---------------------------------------------------------------------------
# STYLING  -  one coordinated light theme
#   1. tokens and page shell
#   2. typography
#   3. hero
#   4. cards, steps and badges
#   5. result card and notes
#   6. figure cards
#   7. Streamlit widget alignment (stable selectors only)
#   8. responsive behaviour
# ---------------------------------------------------------------------------
LIGHT_CSS = """
<style>
/* ---- 1. tokens and page shell ------------------------------------- */
:root {
  --mh-bg:#F5F7FB; --mh-card:#FFFFFF; --mh-surface:#F8FAFC;
  --mh-border:#DDE4EE; --mh-text:#172033; --mh-muted:#5F6B7A;
  --mh-primary:#2563EB; --mh-navy:#153153; --mh-success:#168A5B;
  --mh-warning:#C77700; --mh-danger:#C73A4A; --mh-highlight:#EAF2FF;
  --mh-font: Inter, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --mh-radius:14px;
  --mh-shadow:0 1px 2px rgba(21,49,83,0.05), 0 8px 24px rgba(21,49,83,0.06);
}
.stApp { background: var(--mh-bg); }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1180px; }

/* ---- 2. typography ------------------------------------------------ */
html, body, .stApp, [class*="css"] { font-family: var(--mh-font); }
.stApp, .stApp p, .stApp li, .stApp label { color: var(--mh-text); }
.block-container p, .block-container li { font-size: 0.98rem; line-height: 1.6; }
.block-container h1, .block-container h2, .block-container h3,
.block-container h4, .block-container h5 { color: var(--mh-navy); font-weight: 650; }
.block-container h4 { font-size: 1.18rem; margin-bottom: 0.2rem; }
.block-container h5 { font-size: 1.03rem; }

/* ---- 3. hero ------------------------------------------------------ */
.mh-hero { position:relative; border-radius:18px; overflow:hidden;
  padding:2.5rem 2.1rem; margin-bottom:1.2rem; border:1px solid var(--mh-border);
  background-size:cover; background-position:center; box-shadow:var(--mh-shadow); }
.mh-hero h1 { margin:0.1rem 0 0.5rem 0; font-size:2.05rem; line-height:1.16;
  font-weight:700; color:#FFFFFF; }
.mh-hero p.lede { margin:0; max-width:44rem; font-size:1.02rem; line-height:1.55;
  color:#E8EEF7; }
.mh-eyebrow { display:inline-block; font-size:0.8rem; font-weight:600;
  letter-spacing:0.1em; text-transform:uppercase; color:#FFFFFF;
  background:rgba(255,255,255,0.16); border:1px solid rgba(255,255,255,0.4);
  padding:0.3rem 0.7rem; border-radius:8px; }
.mh-chips { display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:1.1rem; }
.mh-chip { display:inline-flex; align-items:center; gap:0.35rem;
  font-size:0.85rem; font-weight:600; color:#123; background:rgba(255,255,255,0.92);
  border-radius:999px; padding:0.34rem 0.8rem; }
.mh-chip.ghost { background:rgba(255,255,255,0.14); color:#FFFFFF;
  border:1px solid rgba(255,255,255,0.45); font-weight:500; }

/* ---- 4. cards, steps, badges -------------------------------------- */
.mh-card { background:var(--mh-card); border:1px solid var(--mh-border);
  border-radius:var(--mh-radius); padding:1.15rem 1.25rem; box-shadow:var(--mh-shadow); }
.mh-stats { display:grid; gap:0.75rem;
  grid-template-columns:repeat(auto-fit, minmax(168px, 1fr)); margin-bottom:0.5rem; }
.mh-stat { background:var(--mh-card); border:1px solid var(--mh-border);
  border-radius:12px; padding:0.85rem 0.95rem; box-shadow:var(--mh-shadow); }
.mh-stat .k { font-size:0.8rem; font-weight:600; letter-spacing:0.04em;
  text-transform:uppercase; color:var(--mh-muted); }
.mh-stat .v { margin-top:0.3rem; font-size:1.32rem; font-weight:680;
  color:var(--mh-navy); line-height:1.25; overflow-wrap:anywhere; }
.mh-stat.lead { background:var(--mh-highlight); border-color:#C7DBFB; }
.mh-stat.lead .v { color:var(--mh-primary); }
.mh-steprow { display:flex; align-items:baseline; gap:0.65rem; margin:1.5rem 0 0.5rem 0; }
.mh-stepnum { flex:0 0 auto; width:1.7rem; height:1.7rem; border-radius:50%;
  background:var(--mh-primary); color:#FFFFFF; font-size:0.9rem; font-weight:700;
  display:flex; align-items:center; justify-content:center; }
.mh-steprow .t { font-size:1.1rem; font-weight:650; color:var(--mh-navy); }
.mh-steprow .s { font-size:0.92rem; color:var(--mh-muted); }
.mh-badge { display:inline-block; font-size:0.84rem; font-weight:600;
  padding:0.22rem 0.6rem; border-radius:7px; border:1px solid transparent; }
.mh-badge.ok   { background:#E7F6EE; color:#0F6B47; border-color:#BFE5D2; }
.mh-badge.info { background:var(--mh-highlight); color:#1D4ED8; border-color:#C7DBFB; }
.mh-badge.warn { background:#FCF3E3; color:#8A5300; border-color:#EFD9AE; }
.mh-badge.bad  { background:#FBEAEC; color:#93242F; border-color:#F0C4C9; }
.mh-divider { height:1px; background:var(--mh-border); margin:1.6rem 0 0.2rem 0; }
.mh-source { color:var(--mh-muted); font-size:0.86rem; line-height:1.5;
  margin:0.1rem 0 0.8rem 0; }
.mh-footer { margin-top:2.4rem; padding-top:1rem; border-top:1px solid var(--mh-border);
  text-align:center; color:var(--mh-muted); font-size:0.88rem; line-height:1.6; }

/* ---- 5. result card and notes ------------------------------------- */
.mh-result { background:var(--mh-card); border:1px solid #BFE5D2; border-left:5px solid var(--mh-success);
  border-radius:var(--mh-radius); padding:1.5rem 1.6rem; box-shadow:var(--mh-shadow); }
.mh-result .label { font-size:0.86rem; font-weight:600; letter-spacing:0.06em;
  text-transform:uppercase; color:var(--mh-muted); }
.mh-result .value { margin-top:0.25rem; font-size:2.4rem; font-weight:700;
  color:var(--mh-navy); line-height:1.1; overflow-wrap:anywhere; }
.mh-result .sub { margin-top:0.6rem; font-size:0.98rem; color:var(--mh-text); }
.mh-result .note { margin-top:0.8rem; padding-top:0.8rem; border-top:1px solid var(--mh-border);
  font-size:0.9rem; line-height:1.55; color:var(--mh-muted); }
.mh-note { border-radius:11px; padding:0.75rem 0.95rem; font-size:0.93rem;
  line-height:1.55; margin:0.35rem 0; border:1px solid var(--mh-border);
  background:var(--mh-surface); color:var(--mh-text); }
.mh-note strong { color:var(--mh-navy); }
.mh-note.ok    { background:var(--mh-highlight); border-color:#C7DBFB; }
.mh-note.warn  { background:#FCF7EC; border-color:#EFD9AE; }
.mh-note.alert { background:#FBEEF0; border-color:#F0C4C9; }

/* ---- 6. figure cards --------------------------------------------- */
.mh-fig { background:var(--mh-card); border:1px solid var(--mh-border);
  border-radius:var(--mh-radius); padding:1.05rem 1.2rem; box-shadow:var(--mh-shadow); }
.mh-fig .num { font-size:0.82rem; font-weight:700; letter-spacing:0.07em;
  text-transform:uppercase; color:var(--mh-primary); }
.mh-fig h4 { margin:0.15rem 0 0.4rem 0; font-size:1.08rem; color:var(--mh-navy); }
.mh-fig p { margin:0.35rem 0; font-size:0.93rem; line-height:1.55; color:var(--mh-text); }
.mh-fig p.cap { color:var(--mh-muted); font-size:0.89rem; }
.mh-figframe { background:var(--mh-card); border:1px solid var(--mh-border);
  border-radius:var(--mh-radius); padding:0.5rem 0.6rem; box-shadow:var(--mh-shadow); }

/* ---- 7. Streamlit widget alignment ------------------------------- */
/* data-testid selectors are required for these internals; they degrade
   gracefully - if a future Streamlit release renames them the app still runs
   with default (light) Streamlit styling. */
section[data-testid="stSidebar"] { background:var(--mh-card);
  border-right:1px solid var(--mh-border); }
section[data-testid="stSidebar"] h3 { font-size:1.05rem; }
div[data-testid="stMetric"] { background:var(--mh-surface);
  border:1px solid var(--mh-border); border-radius:12px; padding:0.8rem 0.95rem; }
div[data-testid="stMetricLabel"] p { color:var(--mh-muted) !important;
  font-size:0.85rem !important; font-weight:600; }
div[data-testid="stMetricValue"] { color:var(--mh-navy) !important;
  font-size:1.24rem !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { background:var(--mh-card);
  border-color:var(--mh-border) !important; border-radius:var(--mh-radius);
  box-shadow:var(--mh-shadow); }
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] {
  box-shadow:none; }
.stTabs [data-baseweb="tab-list"] { gap:0.35rem; border-bottom:1px solid var(--mh-border); }
.stTabs [data-baseweb="tab"] { font-size:1rem; font-weight:600; color:var(--mh-muted);
  padding:0.6rem 1rem; }
.stTabs [aria-selected="true"] { color:var(--mh-primary) !important; }
.stButton > button { border-radius:10px; font-weight:600; font-size:0.98rem;
  padding:0.55rem 1rem; }
.stButton > button[kind="primary"] { background:var(--mh-primary); border-color:var(--mh-primary); }
.stButton > button[kind="primary"]:hover { background:#1D4ED8; border-color:#1D4ED8; }
.stButton > button[kind="secondary"] { background:var(--mh-card); color:var(--mh-navy);
  border:1px solid var(--mh-border); }
label p { font-size:0.95rem !important; font-weight:600; color:var(--mh-text) !important; }
div[data-testid="stExpander"] { border-radius:12px; border-color:var(--mh-border); }
div[data-testid="stExpander"] summary p { font-weight:600; color:var(--mh-navy) !important; }
div[data-testid="stCaptionContainer"] p { color:var(--mh-muted) !important;
  font-size:0.89rem !important; }

/* ---- 8. responsive ----------------------------------------------- */
@media (max-width: 1024px) {
  .mh-stats { grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); }
}
@media (max-width: 640px) {
  .block-container { padding-left:0.85rem; padding-right:0.85rem; }
  .mh-hero { padding:1.7rem 1.2rem; }
  .mh-hero h1 { font-size:1.55rem; }
  .mh-hero p.lede { font-size:0.96rem; }
  .mh-stats { grid-template-columns:1fr; }
  .mh-result .value { font-size:1.85rem; }
  .stTabs [data-baseweb="tab"] { font-size:0.92rem; padding:0.5rem 0.6rem; }
  .stButton > button { width:100%; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(LIGHT_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def encode_image(path_str: str) -> str | None:
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return None
    try:
        return base64.b64encode(path.read_bytes()).decode()
    except OSError:
        return None


def render_hero() -> None:
    """Banner hero. Falls back to a light-blue-to-navy gradient if the image
    is missing, so the app never crashes on a fresh checkout."""
    encoded = encode_image(str(BANNER_PATH))
    if encoded:
        # Restrained navy overlay: keeps the photograph visible, keeps text legible.
        layer = ("linear-gradient(100deg, rgba(21,49,83,0.90) 0%, "
                 "rgba(21,49,83,0.72) 55%, rgba(21,49,83,0.55) 100%), "
                 f"url('data:image/png;base64,{encoded}')")
    else:
        layer = "linear-gradient(100deg, #153153 0%, #1E4373 55%, #2563EB 100%)"
    st.markdown(
        f"""
        <div class="mh-hero" style="background-image: {layer};">
            <span class="mh-eyebrow">BMDS2003 Data Science Group Assignment</span>
            <h1>Malaysia Housing Median Price Estimator</h1>
            <p class="lede">A market-assisted decision-support prototype that
               estimates the <strong>median house price of a township</strong>
               from Malaysian housing-market records.</p>
            <div class="mh-chips">
                <span class="mh-chip">Static 2025 dataset</span>
                <span class="mh-chip">Township-level median prediction</span>
                <span class="mh-chip ghost">Academic prototype &mdash; not a formal valuation</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def step_header(number: int, title: str, subtitle: str = "") -> None:
    st.markdown(
        f'<div class="mh-steprow"><span class="mh-stepnum">{number}</span>'
        f'<span><span class="t">{title}</span><br><span class="s">{subtitle}</span></span></div>',
        unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="mh-steprow"><span><span class="t">{title}</span>'
                f'<br><span class="s">{subtitle}</span></span></div>',
                unsafe_allow_html=True)


def stat_grid(items: list[tuple[str, str, bool]]) -> None:
    """items = [(label, value, is_lead)] rendered as responsive summary cards."""
    cells = "".join(
        f'<div class="mh-stat{" lead" if lead else ""}">'
        f'<div class="k">{label}</div><div class="v">{value}</div></div>'
        for label, value, lead in items)
    st.markdown(f'<div class="mh-stats">{cells}</div>', unsafe_allow_html=True)


def note(text: str, level: str = "ok") -> None:
    st.markdown(f'<div class="mh-note {level}">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# LOADING
# ---------------------------------------------------------------------------
def model_filename(model_name: str) -> str:
    return model_name.split(" (")[0].lower().replace(" ", "_") + ".pkl"


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    """Model metrics, ordered by the official selection rule (CV RMSE)."""
    results = pd.read_csv(RESULTS_PATH)
    return results.sort_values(["CV_RMSE_mean", "CV_RMSE_std"],
                               ascending=[True, True]).reset_index(drop=True)


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_name: str):
    path = MODELS_DIR / model_filename(model_name)
    if not path.exists():
        raise FileNotFoundError(str(path))
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def load_resources():
    """Load metrics and data with friendly errors instead of a traceback."""
    missing = [p.name for p in (RESULTS_PATH, DATA_PATH) if not p.exists()]
    if missing:
        st.error(
            "Required file(s) not found: " + ", ".join(missing) + ".  \n"
            "**How to fix:** keep `streamlit_app.py`, `model_results.csv`, "
            "`malaysia_house_price_cleaned.csv` and the `models/` folder together "
            "in the same directory, then reload the app."
        )
        st.stop()
    try:
        return load_results(), load_data()
    except Exception:
        st.error(
            "The metrics file or dataset could not be read. **How to fix:** check "
            "that `model_results.csv` and `malaysia_house_price_cleaned.csv` come "
            "from the same notebook run, then reload the app."
        )
        st.stop()


# ---------------------------------------------------------------------------
# LOOKUP AND SIMILARITY (pure functions - unchanged logic)
# ---------------------------------------------------------------------------
def area_options(data: pd.DataFrame, state: str) -> list[str]:
    areas = sorted(data.loc[data["State"] == state, "Area"].dropna().unique())
    return [ALL_AREAS] + list(areas)


def confidence_label(n: int, is_exact: bool) -> tuple[str, str]:
    """Describe how much dataset support the reference values have."""
    if not is_exact:
        return "Fallback reference", "warn"
    if n >= 10:
        return "Strong reference", "ok"
    if n >= 5:
        return "Moderate reference", "ok"
    if n >= 2:
        return "Limited reference", "warn"
    return "Very limited reference", "warn"


def quality_badge(text: str, level: str) -> str:
    """Reference-quality badge. Text always states the quality, so colour is
    never the only indicator."""
    icon = {"ok": "&#10003;", "warn": "&#9888;", "alert": "&#9888;"}.get(level, "")
    css = {"ok": "ok", "warn": "warn", "alert": "bad"}.get(level, "info")
    return f'<span class="mh-badge {css}">{icon} {text}</span>'


def derive_reference(data: pd.DataFrame, state: str, area: str, ptype: str,
                     tenure: str) -> dict:
    """Representative Median_PSF / Transactions from the matching records.

    Area and Tenure are used for LOOKUP ONLY. Tenure is also a model feature,
    but it is included here so the reference values describe the same kind of
    property the user selected.

    Fallback order (first non-empty group wins):
        1. state + area + type + tenure
        2. state + area + type
        3. state + type + tenure
        4. state + type
        5. state
        6. whole dataset
    The SAME pool is reused for the group median, the similar-record table and
    the comparison chart, so every number the user sees comes from one group.
    """
    s = data["State"] == state
    t = data["Primary_Type"] == ptype
    n_ = data["Tenure"] == tenure
    candidates = []
    if area != ALL_AREAS:
        a = data["Area"] == area
        candidates.append((f"{area}, {state} · {ptype} · {tenure}", s & a & t & n_))
        candidates.append((f"{area}, {state} · {ptype} (any tenure)", s & a & t))
    candidates.append((f"{state} · {ptype} · {tenure} (all areas)", s & t & n_))
    candidates.append((f"{state} · {ptype} (any tenure, all areas)", s & t))
    candidates.append((f"{state} · all property types", s))
    candidates.append(("whole dataset", pd.Series(True, index=data.index)))

    for position, (label, mask) in enumerate(candidates):
        pool = data[mask]
        if not pool.empty:
            return {
                "psf": float(pool["Median_PSF"].median()),
                "transactions": int(round(float(pool["Transactions"].median()))),
                "price_median": float(pool["Median_Price"].median()),
                "n": int(len(pool)),
                "level": label,
                "is_exact": position == 0,
                "pool": pool,
            }
    raise ValueError("no records available")


def find_similar_records(pool: pd.DataFrame, psf: float, transactions: int,
                         top_n: int = 5) -> pd.DataFrame:
    """Rank the SAME pool used for the reference values by similarity.

    Score = 0.80 x |%diff in Median_PSF| + 0.20 x normalised |diff in Transactions|
    Lower is more similar. Weights are a documented presentation choice.
    """
    if pool.empty:
        return pool
    ranked = pool.copy()
    psf_gap = (ranked["Median_PSF"] - psf).abs() / max(psf, 1)
    span = ranked["Transactions"].max() - ranked["Transactions"].min()
    span = span if span > 0 else 1
    txn_gap = (ranked["Transactions"] - transactions).abs() / span
    ranked["Similarity_Score"] = W_PSF * psf_gap + W_TXN * txn_gap
    return ranked.sort_values("Similarity_Score").head(top_n)


def psf_reference(data: pd.DataFrame, state: str) -> dict:
    series = data.loc[data["State"] == state, "Median_PSF"]
    return {"median": float(series.median()), "min": float(series.min()),
            "max": float(series.max()), "n": int(series.size)}


def psf_status(psf: float, reference: dict, state: str) -> tuple[str, str]:
    median, low, high = reference["median"], reference["min"], reference["max"]
    deviation = (psf - median) / median if median else 0.0
    if psf < low or psf > high:
        return "alert", (
            f"<strong>&#9888; Outside the observed range.</strong> The selected PSF "
            f"of RM{psf:,.0f} is outside the range observed in {state} "
            f"(RM{low:,.0f} – RM{high:,.0f}). The model is extrapolating beyond its "
            f"training data, so treat the estimate with strong caution.")
    if abs(deviation) > 0.30:
        direction = "above" if deviation > 0 else "below"
        segment = "premium-market" if deviation > 0 else "budget-market"
        return "warn", (
            f"<strong>&#9888; Unusual for this state.</strong> The selected PSF of "
            f"RM{psf:,.0f} is substantially {direction} the {state} median of "
            f"RM{median:,.0f}. The prediction may represent a {segment} property.")
    return "ok", (
        f"<strong>&#10003; Typical for this state.</strong> The selected PSF of "
        f"RM{psf:,.0f} is reasonably close to the median value observed for "
        f"{state} (RM{median:,.0f}).")


def summarise_against_group(prediction: float, group_median: float) -> str:
    difference = prediction - group_median
    percent = (difference / group_median * 100) if group_median else 0.0
    if abs(percent) < 5:
        relation = "close to the matching-group median"
    elif difference > 0:
        relation = "above the matching-group median"
    else:
        relation = "below the matching-group median"
    return (f"The estimate of **RM {prediction:,.0f}** is {relation} "
            f"(**RM {group_median:,.0f}**) — a difference of "
            f"**RM {abs(difference):,.0f}** (**{abs(percent):.1f}%**).")


# ---------------------------------------------------------------------------
# CHART  -  coordinated light chart system
# ---------------------------------------------------------------------------
def render_comparison_chart(prediction: float, group_median: float,
                            similar: pd.DataFrame) -> None:
    """Simple 2-D bar comparison: estimate, group median, closest records.
    Light template, RM formatting, thousands separators, minimal gridlines."""
    labels = ["Your estimate", "Matching-group median"]
    values = [prediction, group_median]
    colors = [SUCCESS, PRIMARY]
    for _, row in similar.iterrows():
        name = str(row["Township"]).title()
        labels.append(name if len(name) <= 22 else name[:20] + "…")
        values.append(float(row["Median_Price"]))
        colors.append(NEUTRAL)

    if PLOTLY_AVAILABLE:
        figure = go.Figure(go.Bar(
            x=labels, y=values, marker_color=colors,
            marker_line_width=0, width=0.6,
            text=[f"RM {v/1000:,.0f}K" for v in values], textposition="outside",
            textfont=dict(color=TEXT, size=12),
            hovertemplate="%{x}<br>RM %{y:,.0f}<extra></extra>"))
        figure.update_layout(
            template="plotly_white",
            title=dict(text="Estimate against the matching comparison group",
                       font=dict(color=NAVY, size=15), x=0, xanchor="left"),
            height=430, margin=dict(l=10, r=10, t=52, b=110),
            paper_bgcolor=CARD, plot_bgcolor=CARD,
            font=dict(color=TEXT, size=13,
                      family='Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'),
            yaxis=dict(title="Median price (RM)", gridcolor=BORDER,
                       zerolinecolor=BORDER, tickformat=",.0f",
                       tickfont=dict(color=MUTED)),
            xaxis=dict(tickangle=-30, tickfont=dict(color=MUTED, size=12),
                       showgrid=False),
            showlegend=False)
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.bar_chart(pd.DataFrame({"Median price (RM)": values}, index=labels))
        st.caption("Install `plotly` for the styled version of this chart.")


def render_metric_chart(results: pd.DataFrame) -> None:
    """Test-metric comparison with the fixed model colours."""
    if not PLOTLY_AVAILABLE:
        return
    order = results["Model"].tolist()
    colours = [MODEL_COLOURS.get(m, NEUTRAL) for m in order]
    figure = go.Figure(go.Bar(
        x=[results.loc[results["Model"] == m, "RMSE_test"].iloc[0] for m in order],
        y=[m.replace(" (Baseline)", "") for m in order],
        orientation="h", marker_color=colours, marker_line_width=0,
        text=[f"RM {results.loc[results['Model'] == m, 'RMSE_test'].iloc[0]/1000:,.1f}K"
              for m in order],
        textposition="outside", textfont=dict(color=TEXT, size=12),
        hovertemplate="%{y}<br>Test RMSE RM %{x:,.0f}<extra></extra>"))
    figure.update_layout(
        template="plotly_white",
        title=dict(text="Test RMSE by model — lower is better",
                   font=dict(color=NAVY, size=15), x=0, xanchor="left"),
        height=300, margin=dict(l=10, r=70, t=52, b=40),
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=TEXT, size=13,
                  family='Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'),
        xaxis=dict(title="Test RMSE (RM)", gridcolor=BORDER, tickformat=",.0f",
                   tickfont=dict(color=MUTED)),
        yaxis=dict(autorange="reversed", showgrid=False,
                   tickfont=dict(color=TEXT, size=12)),
        showlegend=False)
    st.plotly_chart(figure, use_container_width=True)
    st.caption("Colours are fixed for every model across the whole app: "
               "Decision Tree grey · Random Forest blue · XGBoost amber · "
               "LightGBM green. Test metrics are reported, never used for selection.")


# ---------------------------------------------------------------------------
# FIGURE GALLERY
# ---------------------------------------------------------------------------
# Presentation notes for every figure shipped in figures/.
FIGURE_NOTES = {
    "fig01_raw_target_distribution.png": (
        "Figure 1 — Distribution of the raw target variable",
        "Is the median price suitable for modelling as-is?",
        "The raw distribution is extremely right-skewed (skewness ≈ 8.8) with a "
        "tail reaching RM11.4 million.",
        "Skew of this size distorts distance-based fences, which is why outliers "
        "were assessed on the log scale.",
        "The log panel is a viewing transformation only; no record is deleted here."),
    "fig02_raw_numeric_boxplots.png": (
        "Figure 2 — Raw numeric attributes (log scale)",
        "How extreme are the values in each numeric column?",
        "All three columns are strongly right-skewed; a log axis is required to "
        "see the middle 50% at all.",
        "It shows why raw-scale IQR fences flag so many legitimate high-value records.",
        "Points beyond the whiskers are statistical flags, not proven data errors."),
    "fig03_raw_state_counts.png": (
        "Figure 3 — Raw record count by state",
        "How evenly is the dataset spread across Malaysia?",
        "Selangor and Johor dominate; several territories have fewer than five records.",
        "The model will generalise best to well-represented states.",
        "Small-sample states cannot support reliable state-level conclusions."),
    "fig04_raw_tenure_type.png": (
        "Figure 4 — Raw tenure and property-type labels",
        "What cleaning do the categorical columns need?",
        "Tenure contains the same pair written in two orders, and Type holds 46 "
        "multi-value combination strings.",
        "Both need standardising before they can be encoded.",
        "Primary_Type takes the first listed type, which is an operational assumption."),
    "fig05_raw_numeric_correlation.png": (
        "Figure 5 — Correlation between raw numeric variables",
        "Which numeric column is most related to price before cleaning?",
        "Median PSF is strongly related to price; transactions are almost unrelated.",
        "It sets the expectation that PSF will dominate the model.",
        "Correlation measures linear association only."),
    "fig06_raw_psf_vs_price.png": (
        "Figure 6 — Median PSF against Median Price (raw)",
        "What does the PSF–price relationship look like before cleaning?",
        "A positive relationship is visible once density and scale are handled; "
        "extreme values otherwise compress everything into one corner.",
        "It justifies both the log-scale treatment and the outlier assessment.",
        "The zoomed panel changes the visible range only and deletes no records."),
    "fig07_outlier_before_after.png": (
        "Figure 7 — Price before and after outlier deletion",
        "What did log-IQR deletion actually remove?",
        "86 of 2,000 records were removed; the retained range is about "
        "RM90K–RM1.77M.",
        "It defines the market scope the model is valid for.",
        "Removed records are not necessarily errors — this is a scope restriction."),
    "fig08_price_distribution_clean.png": (
        "Figure 8 — Price distribution before and after cleaning",
        "How did the shape of the target change?",
        "Raw-price skewness falls from about 8.8 to about 1.7 while the bulk of "
        "the market is untouched.",
        "A less distorted target is easier for the models to fit.",
        "This is raw-price skewness, not log-price skewness (≈0.11)."),
    "fig09_category_donut.png": (
        "Figure 9 — Landed versus High-Rise share",
        "What is the composition of the cleaned dataset?",
        "About 71% of records are Landed and 29% High-Rise.",
        "Landed property types dominate the dataset the model learns from.",
        "This is a share of records, not of Malaysia's housing stock."),
    "fig10_state_counts_clean.png": (
        "Figure 10 — Records per state after cleaning",
        "Does cleaning change the geographic balance?",
        "Selangor and Johor still account for roughly 47% of records.",
        "Imbalance persists and remains a stated limitation.",
        "Counts are township-level records, not unique township names."),
    "fig11_state_violin.png": (
        "Figure 11 — Price distribution across major states",
        "How do prices differ between the best-represented states?",
        "Kuala Lumpur and Selangor sit clearly higher with long upper tails.",
        "Location is a strong price driver, supporting State as a model feature.",
        "States were chosen by record count, not by price; n is shown per state."),
    "fig12_type_boxplot.png": (
        "Figure 12 — Price by property type",
        "Which property types command higher prices?",
        "A clear ladder runs from Flats at the bottom to Bungalows at the top.",
        "Property type is a strong predictor and is retained as a model feature.",
        "Outlier markers are hidden for readability; the observations remain in "
        "the analysis."),
    "fig13_tenure_violin.png": (
        "Figure 13 — Price distribution by tenure",
        "Is tenure associated with price?",
        "Freehold records show a higher median and a longer upper tail than "
        "Leasehold.",
        "Tenure carries usable signal and is kept as a model feature.",
        "The Mixed group is small (n shown); it cannot support firm conclusions."),
    "fig14_psf_vs_price_clean.png": (
        "Figure 14 — Median PSF against Median Price by category",
        "How strongly is the market rate per square foot related to total price?",
        "A positive relationship holds in both categories, but at similar PSF "
        "levels Landed records reach higher total prices.",
        "PSF is the model's strongest input, yet it cannot explain price alone.",
        "Built-up size is unavailable, so the reason for the gap cannot be "
        "confirmed from this dataset."),
    "fig15_correlation_heatmap.png": (
        "Figure 15 — Correlation matrix of encoded features",
        "How do the encoded features relate to price and to each other?",
        "Median PSF dominates; property-type and tenure indicators carry moderate "
        "signal.",
        "It supports the feature set chosen for modelling.",
        "One-hot indicators of the same variable are negatively correlated by "
        "construction, not by any real effect."),
    "fig16_all_feature_correlations.png": (
        "Figure 16 — Correlation with price for all encoded features",
        "Which individual categories move price up or down?",
        "Selangor and Kuala Lumpur push price up; Perak and Kedah pull it down; "
        "Flats and Apartments are the strongest negative property types.",
        "It quantifies the location and type effects seen in the earlier charts.",
        "These are group differences, not causal effects; categories with very "
        "few records are greyed out."),
    "fig17_top10_transactions.png": (
        "Figure 17 — Most-transacted townships",
        "Do the busiest townships also have the highest prices?",
        "The busiest townships are moderately priced rather than the most "
        "expensive.",
        "Transaction volume reflects market activity more than price level.",
        "The dataset has no supply or income variables, so no cause can be "
        "established."),
    "fig18_model_comparison.png": (
        "Figure 18 — Test-set metric comparison",
        "How do the four models compare on unseen data?",
        "All three ensembles beat the Decision Tree baseline clearly; XGBoost "
        "and LightGBM are close together at the top.",
        "It demonstrates that ensembling improves on a single tree.",
        "The highlighted model was selected on cross-validation, not on these "
        "test bars."),
    "fig19_cv_stability.png": (
        "Figure 19 — Overfitting and cross-validation stability",
        "Which model generalises most consistently?",
        "The baseline memorises its training data; the ensembles show much "
        "smaller train–test gaps.",
        "Stability matters as much as the average score when selecting a model.",
        "Lower point = better average CV RMSE; shorter error bar = greater "
        "fold-to-fold stability."),
    "fig20_pred_actual_residual.png": (
        "Figure 20 — Predicted versus actual, and residuals",
        "Where does the selected model make its errors?",
        "Predictions track the perfect-prediction line in the mainstream range, "
        "but the error spread widens at higher predicted prices.",
        "This heteroscedasticity supports the limitation that the model is less "
        "reliable for premium properties.",
        "Residuals should scatter randomly around zero; a widening funnel does not."),
    "fig21_split_importance.png": (
        "Figure 21 — Split importance of the selected model",
        "Which features does the model split on most often?",
        "Median PSF is used far more than any other feature.",
        "It gives a quick view of what the trees rely on.",
        "Split counts favour continuous features with many possible split points; "
        "read Figure 22 for the fairer measure."),
    "fig22_permutation_importance.png": (
        "Figure 22 — Permutation importance on the test set",
        "How much does each of the five model inputs actually contribute?",
        "Shuffling Median PSF degrades performance far more than shuffling any "
        "other input.",
        "This is measured on unseen data and is the preferred interpretation.",
        "Importance is relative to this feature set; it is not a causal statement."),
}


def render_gallery(filenames: list[str]) -> None:
    """One consistent figure card per figure: number, title, question, chart,
    key finding, business meaning, presentation point, caution.
    Charts are always full-width so long labels stay readable."""
    shown = 0
    for name in filenames:
        path = FIGURES_DIR / name
        if not path.exists():
            continue
        title, question, finding, matters, caution = FIGURE_NOTES[name]
        number, _, heading = title.partition(" — ")
        st.markdown(
            f'<div class="mh-fig"><div class="num">{number}</div>'
            f'<h4>{heading}</h4>'
            f'<p><strong>Question answered:</strong> {question}</p></div>',
            unsafe_allow_html=True)
        with st.container(border=True):
            st.image(str(path), use_container_width=True,
                     caption=f"{number}. {heading}.")
        st.markdown(
            f'<div class="mh-fig">'
            f'<p><strong>Key finding:</strong> {finding}</p>'
            f'<p><strong>Business meaning:</strong> {matters}</p>'
            f'<p><strong>Presentation talking point:</strong> "{finding}"</p>'
            f'<p class="cap"><strong>&#9888; Caution:</strong> {caution}</p></div>',
            unsafe_allow_html=True)
        st.markdown('<div class="mh-divider"></div>', unsafe_allow_html=True)
        shown += 1
    if shown == 0:
        st.info("No figures found for this section. **How to fix:** copy the "
                "`figures/` folder produced by the notebook next to "
                "`streamlit_app.py`, then reload the app.")


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
def render_model_summary(results: pd.DataFrame, recommended: str) -> None:
    """Compact model-summary cards shown under the hero on every visit."""
    row = results.iloc[0]
    stat_grid([
        ("Recommended model", recommended.replace(" (Baseline)", ""), True),
        ("CV RMSE", f"RM {row['CV_RMSE_mean']/1000:,.1f}K", False),
        ("Test RMSE", f"RM {row['RMSE_test']/1000:,.1f}K", False),
        ("Test MAE", f"RM {row['MAE_test']/1000:,.1f}K", False),
        ("Test R²", f"{row['R2_test']:.3f}", False),
    ])
    best_test = results.sort_values("RMSE_test").iloc[0]["Model"]
    line = ("Recommended using training-set cross-validation. "
            "Test metrics are reported for transparency and were not used to "
            "select the model.")
    if best_test != recommended:
        line += (f" <strong>{best_test}</strong> posts the best hold-out test "
                 f"metrics, but <strong>{recommended}</strong> had the lowest "
                 f"cross-validation RMSE — see the Model Report tab for the "
                 f"tie explanation.")
    st.markdown(f'<p class="mh-source">{line}</p>', unsafe_allow_html=True)


def tab_estimator(data: pd.DataFrame, results: pd.DataFrame) -> None:
    recommended = results.iloc[0]["Model"]

    if "reset_counter" not in st.session_state:
        st.session_state.reset_counter = 0
    suffix = st.session_state.reset_counter

    # ---- STEP 1  LOCATION -------------------------------------------------
    step_header(1, "Location",
                "Dependent dropdowns: pick a state, then an area within it.")
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            state = st.selectbox("State", sorted(data["State"].unique()),
                                 key=f"state_{suffix}")
        with right:
            area = st.selectbox("Area", area_options(data, state),
                                key=f"area_{state}_{suffix}",
                                help="Used to look up representative 2025 dataset "
                                     "values. Area is not a model feature.")
        st.caption("Area and township selections help retrieve representative "
                   "2025 market values. They are not direct model inputs — the "
                   "trained pipeline receives state, tenure, property type, "
                   "median PSF and transactions only.")

    # ---- STEP 2  PROPERTY DETAILS ----------------------------------------
    step_header(2, "Property details",
                "Both of these are real model inputs.")
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            ptype = st.selectbox("Property type",
                                 sorted(data["Primary_Type"].unique()),
                                 key=f"ptype_{suffix}",
                                 help="The primary type recorded for the township.")
        with right:
            tenure = st.selectbox("Tenure", sorted(data["Tenure"].unique()),
                                  key=f"tenure_{suffix}",
                                  help="Freehold, Leasehold, or Mixed where the "
                                       "township records list both.")

    ref = derive_reference(data, state, area, ptype, tenure)
    conf_text, conf_level = confidence_label(ref["n"], ref["is_exact"])

    # ---- STEP 3  DATASET REFERENCE ---------------------------------------
    step_header(3, "Dataset reference values",
                "Retrieved automatically from the static 2025 dataset.")
    with st.container(border=True):
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Reference median PSF (RM)", f"{ref['psf']:,.0f}")
        r2.metric("Reference transactions", f"{ref['transactions']:,}")
        r3.metric("Matching records", f"{ref['n']:,}")
        r4.metric("Dataset year", "2025")
        st.markdown(
            f'<div class="mh-note {conf_level}">'
            f'{quality_badge(conf_text, conf_level)} &nbsp;Values are the median of '
            f'<strong>{ref["n"]:,}</strong> record(s) matching the hierarchy level '
            f'<strong>{ref["level"]}</strong>.'
            + ("" if ref["is_exact"] else
               " No exact area + property type + tenure records were available, so "
               "the system fell back to this broader group. Nothing is approximated "
               "silently — the level above is always the group actually used.")
            + "</div>", unsafe_allow_html=True)

    # ---- STEP 4  ADVANCED OPTIONS ----------------------------------------
    step_header(4, "Advanced options (optional)",
                "Model comparison and optional scenario adjustments.")
    with st.expander("Advanced model options and scenario adjustments",
                     expanded=False):
        st.caption("Everything in this panel is optional. The project conclusion "
                   "uses the cross-validation-selected model and the dataset "
                   "reference values.")
        labels = {}
        for _, row_ in results.iterrows():
            tag = " — Recommended by CV" if row_["Model"] == recommended else ""
            if "Baseline" in row_["Model"]:
                tag = " — Baseline"
            labels[f"{row_['Model']}{tag}"] = row_["Model"]
        chosen_label = st.selectbox("Model used for prediction",
                                    list(labels.keys()), index=0,
                                    help="Switching the model changes the "
                                         "prediction and the displayed metrics. It "
                                         "does not change the project's stated "
                                         "recommendation.")
        model_name = labels[chosen_label]

        st.markdown('<div class="mh-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Optional scenario adjustment**")
        st.caption("Override the dataset reference values to explore a different "
                   "market scenario. Only these two values are adjustable, because "
                   "the pipeline accepts five features.")
        a_left, a_right = st.columns(2)
        with a_left:
            psf = st.slider("Custom median price per square foot (RM)",
                            int(data["Median_PSF"].min()),
                            int(data["Median_PSF"].max()), int(ref["psf"]),
                            key=f"psf_{state}_{area}_{ptype}_{tenure}_{suffix}")
        with a_right:
            transactions = st.slider("Custom transactions in the township",
                                     int(data["Transactions"].min()),
                                     int(data["Transactions"].max()),
                                     int(ref["transactions"]),
                                     key=f"txn_{state}_{area}_{ptype}_{tenure}_{suffix}")
        if st.button("Reset to dataset values", use_container_width=False):
            st.session_state.reset_counter += 1
            st.rerun()

    row = results[results["Model"] == model_name].iloc[0]
    try:
        model = load_model(model_name)
    except FileNotFoundError as exc:
        st.error(f"Model file not found: `{exc}`.  \n"
                 "**How to fix:** unzip the `models/` folder next to "
                 "`streamlit_app.py`, then reload the app.")
        st.stop()

    with st.container(border=True):
        stat_grid([
            ("Model in use", model_name.replace(" (Baseline)", ""),
             model_name == recommended),
            ("CV RMSE", f"RM {row['CV_RMSE_mean']/1000:,.1f}K", False),
            ("Test RMSE", f"RM {row['RMSE_test']/1000:,.1f}K", False),
            ("Test MAE", f"RM {row['MAE_test']/1000:,.1f}K", False),
            ("Test R²", f"{row['R2_test']:.3f}", False),
        ])
        if model_name == recommended:
            st.markdown(
                f'<div class="mh-note ok">{quality_badge("Recommended model", "ok")} '
                f'&nbsp;<strong>{model_name}</strong> was selected by '
                f'cross-validation RMSE on the training set — the official choice '
                f'for this project.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="mh-note warn">{quality_badge("Comparison model", "warn")} '
                f'&nbsp;You are using <strong>{model_name}</strong> for academic '
                f'comparison. The project\'s selected model remains '
                f'<strong>{recommended}</strong>, chosen by cross-validation.</div>',
                unsafe_allow_html=True)

        if psf != int(ref["psf"]) or transactions != int(ref["transactions"]):
            st.markdown(
                f'<div class="mh-note warn">{quality_badge("Scenario adjustment", "warn")} '
                f'&nbsp;Using adjusted values (PSF RM{psf:,.0f}, {transactions:,} '
                f'transactions) instead of the dataset reference (PSF '
                f'RM{ref["psf"]:,.0f}, {ref["transactions"]:,} transactions).</div>',
                unsafe_allow_html=True)

        level, message = psf_status(psf, psf_reference(data, state), state)
        note(message, level)

    # ---- STEP 5  ESTIMATE -------------------------------------------------
    step_header(5, "Estimate", "One action produces the township-level benchmark.")
    b_left, b_right = st.columns([3, 1])
    with b_left:
        predict = st.button("Estimate Township Median Price", type="primary",
                            use_container_width=True)
    with b_right:
        if st.button("Reset inputs", use_container_width=True):
            st.session_state.reset_counter += 1
            st.rerun()
    if not predict:
        st.caption("Select your options above, then press "
                   "**Estimate Township Median Price**.")
        return

    features = pd.DataFrame([{
        "State": state, "Tenure": tenure, "Primary_Type": ptype,
        "Median_PSF": psf, "Transactions": transactions}])[MODEL_FEATURES]
    try:
        prediction = float(model.predict(features)[0])
    except Exception:
        st.error("The model could not produce a prediction for these inputs. "
                 "**How to fix:** re-export the models from the notebook so the "
                 "pipelines match this dataset, then reload the app.")
        return

    location = state if area == ALL_AREAS else f"{area}, {state}"
    st.markdown(
        f"""
        <div class="mh-result">
            <div class="label">Estimated township-level median price</div>
            <div class="value">RM {prediction:,.0f}</div>
            <div class="sub">{ptype} · {tenure} · {location}</div>
            <div class="sub">Model used: <strong>{model_name}</strong> ·
                Typical test MAE: <strong>RM {row['MAE_test']:,.0f}</strong> ·
                {quality_badge(conf_text, conf_level)} ·
                {ref['n']:,} matching record(s)</div>
            <div class="note">This estimate represents a township-level median
                benchmark based on the static 2025 dataset. It is not a formal
                valuation of an individual property.<br>
                MAE is the model's average absolute error on the test dataset. It
                is not a confidence interval or a statistically valid prediction
                interval.</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- STEP 6  SIMILAR RECORDS -----------------------------------------
    # Same pool as the reference values - one consistent comparison group
    pool = ref["pool"]
    similar = find_similar_records(pool, psf, transactions)
    group_median = ref["price_median"]

    step_header(6, "Similar 2025 township records",
                "Evidence behind the estimate, from one consistent comparison group.")
    with st.container(border=True):
        st.markdown('<p class="mh-source">Source: cleaned Malaysia housing '
                    'dataset, 2025. These are historical dataset records and are '
                    'not live property-market data.</p>', unsafe_allow_html=True)
        st.markdown(summarise_against_group(prediction, group_median))
        st.caption(f"Comparison group: {ref['n']} record(s) matching "
                   f"{ref['level']}. The same group supplies the reference "
                   f"values, the group median and the records below.")
        render_comparison_chart(prediction, group_median, similar)

        st.markdown(f"**{len(similar)} most similar record(s)** — ranked within "
                    f"the same comparison group")
        table = similar[["Township", "Area", "Median_Price", "Median_PSF",
                         "Transactions"]].reset_index(drop=True)
        st.dataframe(pd.DataFrame({
            "Township": table["Township"].str.title(),
            "Area": table["Area"],
            "Median Price (RM)": table["Median_Price"].map(lambda v: f"{v:,.0f}"),
            "Median PSF (RM)": table["Median_PSF"].map(lambda v: f"{v:,.0f}"),
            "Transactions": table["Transactions"]}),
            use_container_width=True, hide_index=True)
        st.caption("Median Price and Median PSF are township medians in Ringgit "
                   "Malaysia; Transactions is the recorded transaction count.")

        with st.expander("Methodology — how similarity is calculated"):
            st.write(
                f"Records come from the comparison group above. Each is scored as "
                f"{W_PSF:.0%} × the absolute percentage difference in median price "
                f"per square foot plus {W_TXN:.0%} × the absolute difference in "
                f"transactions, normalised by the range within the group. A lower "
                f"score means a closer match. **The weights are a presentation "
                f"choice reflecting the much stronger evidence for Median PSF; "
                f"they have not been statistically validated.**")
            st.dataframe(
                similar[["Township", "Median_PSF", "Transactions",
                         "Similarity_Score"]]
                .rename(columns={"Median_PSF": "Median PSF (RM)",
                                 "Similarity_Score": "Similarity score"})
                .round({"Similarity score": 3}).reset_index(drop=True),
                use_container_width=True, hide_index=True)


def tab_eda() -> None:
    section_header("EDA and market insights",
                   "Figures generated by the project notebook from the same "
                   "dataset the model was trained on.")
    sections = {
        "1. Data quality":
            ["fig02_raw_numeric_boxplots.png", "fig04_raw_tenure_type.png",
             "fig07_outlier_before_after.png"],
        "2. Price distribution":
            ["fig01_raw_target_distribution.png",
             "fig08_price_distribution_clean.png"],
        "3. State comparison":
            ["fig03_raw_state_counts.png", "fig10_state_counts_clean.png",
             "fig11_state_violin.png"],
        "4. Property-type comparison":
            ["fig12_type_boxplot.png", "fig09_category_donut.png"],
        "5. Tenure comparison": ["fig13_tenure_violin.png"],
        "6. PSF relationship":
            ["fig06_raw_psf_vs_price.png", "fig14_psf_vs_price_clean.png"],
        "7. Correlation":
            ["fig05_raw_numeric_correlation.png", "fig15_correlation_heatmap.png",
             "fig16_all_feature_correlations.png"],
        "8. Transaction activity": ["fig17_top10_transactions.png"],
    }
    with st.container(border=True):
        choice = st.selectbox("Section", list(sections.keys()),
                              help="Each section groups the figures that answer "
                                   "one analytical question.")
        st.caption(f"Showing {len(sections[choice])} figure(s) in “{choice}”. "
                   "Every figure is displayed full-width so long state and "
                   "feature labels stay readable.")
    st.markdown('<div class="mh-divider"></div>', unsafe_allow_html=True)
    render_gallery(sections[choice])


def tab_model_report(results: pd.DataFrame) -> None:
    recommended = results.iloc[0]["Model"]

    # ---- 1. Model overview ------------------------------------------------
    section_header("Model report",
                   "Four models, one selection rule, and the evidence behind it.")
    with st.container(border=True):
        st.markdown("##### 1. Model overview")
        st.markdown(
            "- **Decision Tree — baseline**, default configuration, untuned "
            "reference point.\n"
            "- **Random Forest**, bagged trees, tuned by grid search.\n"
            "- **XGBoost**, level-wise regularised gradient boosting, tuned.\n"
            "- **LightGBM**, leaf-wise gradient boosting, tuned.\n\n"
            "All four use the same five inputs and the same preprocessing "
            "pipeline (one-hot encoding + standardisation fitted inside the "
            "pipeline)."
        )

    # ---- 2. Performance comparison ---------------------------------------
    st.markdown("##### 2. Performance comparison")
    table = results.copy()
    table["CV Rank"] = table["CV_RMSE_mean"].rank().astype(int)
    table["Test Rank"] = table["RMSE_test"].rank().astype(int)
    table["Selected by CV"] = ["Yes ✓" if m == recommended else "—"
                               for m in table["Model"]]
    display = pd.DataFrame({
        "Model": table["Model"],
        "CV Rank": table["CV Rank"],
        "CV RMSE Mean": table["CV_RMSE_mean"].map(lambda v: f"RM {v/1000:,.1f}K"),
        "CV RMSE Standard Deviation":
            table["CV_RMSE_std"].map(lambda v: f"RM {v/1000:,.1f}K"),
        "Test Rank": table["Test Rank"],
        "Test RMSE": table["RMSE_test"].map(lambda v: f"RM {v/1000:,.1f}K"),
        "Test MAE": table["MAE_test"].map(lambda v: f"RM {v/1000:,.1f}K"),
        "Test R²": table["R2_test"].map(lambda v: f"{v:.3f}"),
        "Train–test gap": table["Gap_RMSE"].map(lambda v: f"RM {v/1000:,.1f}K"),
        "Selected by CV": table["Selected by CV"]})
    with st.container(border=True):
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption("CV RMSE Mean is the 5-fold cross-validation error on the "
                   "training set — the official selection criterion. Test columns "
                   "describe unseen-data performance and were not used to select "
                   "the model. A larger train–test gap indicates more overfitting.")
        render_metric_chart(results)

    # ---- 3-6. Evidence figures -------------------------------------------
    st.markdown("##### 3. Cross-validation stability, overfitting, diagnostics "
                "and importance")
    render_gallery(["fig18_model_comparison.png", "fig19_cv_stability.png",
                    "fig20_pred_actual_residual.png",
                    "fig22_permutation_importance.png",
                    "fig21_split_importance.png"])

    # ---- 7. Model selection ----------------------------------------------
    st.markdown("##### 7. Model selection")
    top_two = results.head(2)
    with st.container(border=True):
        if len(top_two) == 2:
            gap = abs(top_two.iloc[0]["CV_RMSE_mean"] - top_two.iloc[1]["CV_RMSE_mean"])
            st.markdown(
                f'<div class="mh-note ok">{quality_badge("Effectively tied", "info")} '
                f'&nbsp;The top two models differ by only '
                f'<strong>RM {gap:,.0f}</strong> in cross-validation RMSE, against '
                f'a cross-validation standard deviation of about '
                f'RM {top_two.iloc[0]["CV_RMSE_std"]:,.0f}. '
                f'<strong>{recommended}</strong> was selected because it had the '
                f'lowest CV mean and the lower CV variability, while '
                f'<strong>{top_two.iloc[1]["Model"]}</strong> achieved the stronger '
                f'hold-out test metrics. Selection was made on cross-validation '
                f'only, so the test set stays an untouched estimate of unseen-data '
                f'performance. The selected model is not claimed to be '
                f'dramatically better.</div>', unsafe_allow_html=True)
        st.markdown(
            "**How to read the roles:** the model *selected by training-set CV* is "
            "the project's answer; the model with the *best hold-out test "
            "performance* is reported for transparency; the *baseline* Decision "
            "Tree shows what a single untuned tree achieves and carries the "
            "largest overfitting risk.")

    # ---- 8. Limitations and improvements ---------------------------------
    st.markdown("##### 8. Limitations and recommended improvements")
    with st.container(border=True):
        st.markdown("""
**Key limitations**

- Outlier deletion restricts the model to the mainstream market (about
  RM90K–RM1.77M). It is less suitable for very low-cost or luxury properties.
- Records are township-level: no bedroom count, land size, property age or
  amenity data, which caps achievable accuracy.
- Property type takes the first listed type from a multi-value label. About
  18% of records use a label whose token order could change the assigned type.
- The dataset is imbalanced toward Selangor and Johor and toward terrace houses.
- The prototype needs a realistic median PSF; accuracy degrades without one.
- Errors widen at higher prices (heteroscedasticity), so premium estimates are
  less reliable.

**Recommended improvements**

- Property-level attributes (built-up size, age, bedrooms).
- Geospatial features such as distance to MRT/LRT and schools.
- Multi-year data to capture time trends.
- Multi-label encoding of the raw property-type field.
- Quantile regression to produce a statistically grounded range instead of a
  point estimate.
""")


# ---------------------------------------------------------------------------
# SIDEBAR AND MAIN
# ---------------------------------------------------------------------------
def render_sidebar(recommended: str, record_count: int) -> None:
    with st.sidebar:
        st.markdown("### About this prototype")
        st.write("Estimates the **median house price of a township** from its "
                 "state, tenure, property type, market activity and known median "
                 "price per square foot.")
        st.markdown('<div class="mh-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
**Dataset year** · 2025
**Records after cleaning** · {record_count:,}
**Prediction level** · Township-level median
**Recommended model** · {recommended} (by cross-validation)
        """)
        st.markdown('<div class="mh-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Main model inputs**")
        st.write("State · Tenure · Property type · Median PSF · Transactions. "
                 "Area and township are lookup controls only.")
        st.markdown('<div class="mh-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Main limitation**")
        st.write("The median price per square foot must already be known. The "
                 "model is market-assisted: it refines a known market rate rather "
                 "than discovering prices without market input.")
        st.markdown('<div class="mh-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Navigation**")
        st.write("**Price Estimator** — generate a prediction.  \n"
                 "**EDA & Market Insights** — explore market patterns.  \n"
                 "**Model Report** — examine model evidence.")
        st.markdown('<div class="mh-divider"></div>', unsafe_allow_html=True)
        st.caption("Academic prototype for BMDS2003 coursework. Built on a static "
                   "2025 dataset, not live market data. Estimates are "
                   "township-level medians, not valuations of individual "
                   "properties, and must not be used for real financial decisions.")


def main() -> None:
    inject_css()
    results, data = load_resources()
    recommended = results.iloc[0]["Model"]
    render_sidebar(recommended, len(data))
    render_hero()
    render_model_summary(results, recommended)

    estimator, eda, report = st.tabs(
        ["Price Estimator", "EDA & Market Insights", "Model Report"])
    with estimator:
        tab_estimator(data, results)
    with eda:
        tab_eda()
    with report:
        tab_model_report(results)


main()

st.markdown('<div class="mh-footer">BMDS2003 Data Science Group Assignment | '
            'Academic Prototype | Malaysia Housing Prices 2025<br>'
            'Not a formal property valuation or financial recommendation.</div>',
            unsafe_allow_html=True)
