"""
BMDS2003 Data Science - Deployment Prototype
Malaysia Housing Median Price Estimator

Run locally:   streamlit run streamlit_app.py
Deploy:        Streamlit Community Cloud

--------------------------------------------------------------------------
MODEL CONTRACT - do not change without retraining
--------------------------------------------------------------------------
The saved pipelines accept exactly five features:

    State, Tenure, Primary_Type, Median_PSF, Transactions

`Area` and `Township` are NOT model features. Area is typed freely by the
user and is used only to look up representative market values from the 2025
dataset. It is never added to the prediction DataFrame. Making the model
area-aware would require retraining all four pipelines with Area included.
--------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
import base64
import difflib

import joblib
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ===========================================================================
# CONFIGURATION
# ===========================================================================
st.set_page_config(
    page_title="Malaysia Housing Median Price Estimator",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
RESULTS_PATH = APP_DIR / "model_results.csv"
DATA_PATH = APP_DIR / "malaysia_house_price_cleaned.csv"
MODELS_DIR = APP_DIR / "models"
FIGURES_DIR = APP_DIR / "figures"
BANNER_PATH = APP_DIR / "assets" / "malaysia_housing_banner.png"

# The five columns the trained pipelines expect - never add to this list.
MODEL_FEATURES = ["State", "Tenure", "Primary_Type", "Median_PSF", "Transactions"]


# ===========================================================================
# DESIGN SYSTEM
# Semantic tokens, not raw hex in components. Contrast checked against the
# page surface: text-1 ~17:1, text-2 ~11:1, text-3 ~6.7:1 (all pass 4.5:1).
# ===========================================================================
def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            /* surfaces */
            --surface-0:#0B1220; --surface-1:#131C2E; --surface-2:#1B2740;
            --border:rgba(148,163,184,0.18); --border-strong:rgba(148,163,184,0.34);
            /* text */
            --text-1:#F8FAFC; --text-2:#CBD5E1; --text-3:#94A3B8;
            /* semantic */
            --brand:#3B82F6; --brand-soft:rgba(59,130,246,0.14);
            --success:#22C55E; --success-soft:rgba(34,197,94,0.14);
            --warn:#F59E0B;   --warn-soft:rgba(245,158,11,0.14);
            --danger:#F87171; --danger-soft:rgba(248,113,113,0.14);
            /* rhythm: 4/8px scale */
            --sp-1:4px; --sp-2:8px; --sp-3:16px; --sp-4:24px; --sp-5:32px;
            --radius:14px; --radius-sm:10px;
            --shadow:0 8px 24px rgba(0,0,0,0.32);
        }

        .stApp {
            background:
              radial-gradient(1200px 600px at 15% -10%, #17233C 0%, transparent 60%),
              linear-gradient(168deg, var(--surface-0) 0%, #0E1626 60%, #0B1220 100%);
            color: var(--text-1);
        }
        .block-container { padding-top: var(--sp-4); max-width: 1180px; }

        /* Body text >=16px on mobile so iOS does not auto-zoom */
        html, body, [class*="css"] { font-size: 16px; line-height: 1.6; }

        /* Numbers in columns line up */
        div[data-testid="stMetricValue"], .mh-num, table, .stDataFrame {
            font-variant-numeric: tabular-nums;
        }

        /* ---- Hero ---- */
        .mh-hero {
            border-radius: 18px; padding: var(--sp-5) var(--sp-4);
            margin-bottom: var(--sp-4); border: 1px solid var(--border);
            box-shadow: var(--shadow);
            background-size: cover; background-position: center;
        }
        .mh-kicker {
            display:inline-block; font-size:0.72rem; font-weight:600;
            letter-spacing:0.12em; text-transform:uppercase; color:#EFF6FF;
            background: rgba(59,130,246,0.85); padding:6px 12px;
            border-radius:999px; margin-bottom: var(--sp-3);
        }
        .mh-hero h1 { margin:0 0 var(--sp-2) 0; font-size:2rem; font-weight:700;
                      line-height:1.2; color:#FFFFFF; }
        .mh-hero p  { margin:0; max-width:64ch; color:#E2E8F0; line-height:1.6; }

        /* ---- Section heading ---- */
        .mh-step {
            display:flex; align-items:center; gap:var(--sp-2);
            margin: var(--sp-4) 0 var(--sp-2) 0;
        }
        .mh-step .n {
            width:26px; height:26px; border-radius:50%; flex:0 0 26px;
            background:var(--brand-soft); border:1px solid var(--brand);
            color:#BFDBFE; font-size:0.78rem; font-weight:700;
            display:flex; align-items:center; justify-content:center;
        }
        .mh-step h3 { margin:0; font-size:1rem; font-weight:600; color:var(--text-1); }

        /* ---- Metrics ---- */
        div[data-testid="stMetric"] {
            background: var(--surface-1); border:1px solid var(--border);
            border-radius: var(--radius); padding: 14px 16px; box-shadow: var(--shadow);
        }
        div[data-testid="stMetricLabel"] p {
            color: var(--text-3) !important; font-size:0.74rem !important;
            letter-spacing:0.06em; text-transform:uppercase;
        }
        div[data-testid="stMetricValue"] { color: var(--text-1) !important;
                                           font-size:1.3rem !important; }

        /* ---- Result panel ---- */
        .mh-result {
            background: linear-gradient(135deg, var(--success-soft), var(--brand-soft));
            border:1px solid rgba(34,197,94,0.42); border-radius:18px;
            padding: var(--sp-4) var(--sp-4); box-shadow: var(--shadow);
        }
        .mh-result .cap { font-size:0.76rem; letter-spacing:0.1em;
                          text-transform:uppercase; color:var(--text-3); }
        .mh-result .amount { font-size:2.5rem; font-weight:700; color:var(--success);
                             line-height:1.1; margin:var(--sp-1) 0 var(--sp-3) 0;
                             font-variant-numeric: tabular-nums; }
        .mh-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
                   gap:var(--sp-2) var(--sp-4); margin-top:var(--sp-2); }
        .mh-grid div { font-size:0.88rem; color:var(--text-2); }
        .mh-grid span { display:block; color:var(--text-3); font-size:0.72rem;
                        text-transform:uppercase; letter-spacing:0.05em; }

        /* ---- Inline notices: colour + label, never colour alone ---- */
        .mh-note { border-radius:var(--radius-sm); padding:12px 14px;
                   font-size:0.88rem; line-height:1.55; margin-top:var(--sp-2);
                   border-left:3px solid; }
        .mh-note b { display:block; margin-bottom:2px; font-size:0.78rem;
                     letter-spacing:0.05em; text-transform:uppercase; }
        .mh-note.info   { background:var(--brand-soft);  border-color:var(--brand);
                          color:#DBEAFE; }
        .mh-note.good   { background:var(--success-soft);border-color:var(--success);
                          color:#DCFCE7; }
        .mh-note.warn   { background:var(--warn-soft);   border-color:var(--warn);
                          color:#FEF3C7; }
        .mh-note.danger { background:var(--danger-soft); border-color:var(--danger);
                          color:#FEE2E2; }

        .mh-fine { color:var(--text-3); font-size:0.8rem; line-height:1.55; }

        /* ---- Figure card ---- */
        .mh-fig { background:var(--surface-1); border:1px solid var(--border);
                  border-radius:var(--radius); padding:14px 16px; margin-bottom:6px; }
        .mh-fig h4 { margin:0 0 6px 0; font-size:0.95rem; color:var(--text-1); }
        .mh-fig p  { margin:4px 0; font-size:0.85rem; line-height:1.55;
                     color:var(--text-2); }

        /* ---- Containers, buttons, inputs ---- */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--surface-1); border:1px solid var(--border) !important;
            border-radius: var(--radius); box-shadow: var(--shadow); padding:6px 4px;
        }
        .stButton > button {
            border-radius: var(--radius-sm); font-weight:600;
            min-height:44px;                      /* 44px touch target */
            transition: transform 180ms ease, opacity 180ms ease;
        }
        .stButton > button:hover { transform: translateY(-1px); }
        .stButton > button:active { transform: translateY(0); }

        /* Visible focus rings - never removed */
        button:focus-visible, input:focus-visible, select:focus-visible,
        textarea:focus-visible, [role="tab"]:focus-visible {
            outline:2px solid var(--brand) !important; outline-offset:2px !important;
        }

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] { gap:var(--sp-1); border-bottom:1px solid var(--border); }
        .stTabs [data-baseweb="tab"] {
            height:46px; padding:0 18px; border-radius:var(--radius-sm) var(--radius-sm) 0 0;
            color:var(--text-3);
        }
        .stTabs [aria-selected="true"] {
            background:var(--surface-2); color:var(--text-1) !important;
            border-bottom:2px solid var(--brand);
        }

        section[data-testid="stSidebar"] > div {
            background: var(--surface-1); border-right:1px solid var(--border);
        }

        .mh-footer { margin-top:var(--sp-5); padding-top:var(--sp-3);
                     border-top:1px solid var(--border); text-align:center;
                     color:var(--text-3); font-size:0.8rem; }

        /* ---- Responsive: 375 / 768 / 1024 ---- */
        @media (max-width: 768px) {
            .mh-hero { padding: var(--sp-4) var(--sp-3); }
            .mh-hero h1 { font-size:1.5rem; }
            .mh-result .amount { font-size:1.9rem; }
            .block-container { padding-left:12px; padding-right:12px; }
        }

        /* ---- Respect reduced motion ---- */
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after {
                animation-duration:0.01ms !important; transition-duration:0.01ms !important;
            }
            .stButton > button:hover { transform:none; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def step_heading(number: int, title: str) -> None:
    st.markdown(f'<div class="mh-step"><div class="n">{number}</div>'
                f'<h3>{title}</h3></div>', unsafe_allow_html=True)


def notice(kind: str, label: str, body: str) -> None:
    """Inline notice. A text label carries the meaning, not the colour alone."""
    st.markdown(f'<div class="mh-note {kind}"><b>{label}</b>{body}</div>',
                unsafe_allow_html=True)


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
    encoded = encode_image(str(BANNER_PATH))
    layer = (f"linear-gradient(rgba(11,18,32,0.80), rgba(11,18,32,0.88)), "
             f"url('data:image/png;base64,{encoded}')" if encoded
             else "linear-gradient(120deg,#0B1220 0%,#16243D 45%,#1E3A5F 100%)")
    st.markdown(
        f"""
        <div class="mh-hero" style="background-image:{layer};">
            <span class="mh-kicker">BMDS2003 Data Science Group Assignment</span>
            <h1>Malaysia Housing Median Price Estimator</h1>
            <p>Estimate a township-level median house price from location, property
               type, tenure and market rate, using a regression model trained on
               Malaysian housing data from 2025.</p>
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# LOADING
# ===========================================================================
def model_filename(model_name: str) -> str:
    return model_name.split(" (")[0].lower().replace(" ", "_") + ".pkl"


@st.cache_data(show_spinner=False)
def load_results() -> pd.DataFrame:
    """Model metrics, ordered by the official selection rule (CV RMSE)."""
    results = pd.read_csv(RESULTS_PATH)
    return (results.sort_values(["CV_RMSE_mean", "CV_RMSE_std"],
                                ascending=[True, True]).reset_index(drop=True))


@st.cache_resource(show_spinner="Loading model...")
def load_model(model_name: str):
    path = MODELS_DIR / model_filename(model_name)
    if not path.exists():
        raise FileNotFoundError(path.name)
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def load_resources():
    missing = [p.name for p in (RESULTS_PATH, DATA_PATH) if not p.exists()]
    if missing:
        st.error(
            "**Required file(s) not found:** " + ", ".join(missing) + "  \n\n"
            "Keep `streamlit_app.py`, `model_results.csv`, "
            "`malaysia_house_price_cleaned.csv` and the `models/` folder in the "
            "same directory, then reload the page.")
        st.stop()
    try:
        results, data = load_results(), load_data()
    except Exception:
        st.error("**The metrics file or dataset could not be read.** Check that "
                 "`model_results.csv` and `malaysia_house_price_cleaned.csv` come "
                 "from the same notebook run and are not empty.")
        st.stop()
    if data.empty or results.empty:
        st.error("**The dataset or metrics file is empty.** Re-run the notebook "
                 "and copy the regenerated files next to this app.")
        st.stop()
    return results, data


# ===========================================================================
# AREA RESOLUTION AND REFERENCE LOOKUP  (pure functions)
# ===========================================================================
def resolve_area(user_text: str, data: pd.DataFrame, state: str):
    """Match free-typed text against known areas in the selected state.

    Any text is accepted - this never rejects an unknown location. Matching is
    case-insensitive and whitespace-tolerant. When there is no exact match,
    close spellings are offered as hints so a typo does not silently drop the
    user to a broader reference group.

    Returns (matched_area or None, [suggestions]).
    """
    text = (user_text or "").strip()
    if not text:
        return None, []
    known = [a for a in data.loc[data["State"] == state, "Area"].dropna().unique()]
    lookup = {str(a).strip().casefold(): a for a in known}
    key = text.casefold()
    if key in lookup:
        return lookup[key], []
    partial = [orig for folded, orig in lookup.items() if key in folded or folded in key]
    if partial:
        return None, sorted(partial)[:3]
    close = difflib.get_close_matches(key, list(lookup.keys()), n=3, cutoff=0.7)
    return None, [lookup[c] for c in close]


def derive_reference(data: pd.DataFrame, state: str, area: str | None,
                     ptype: str, tenure: str) -> dict:
    """Suggested Median_PSF and Transactions from the closest available group.

    Fallback hierarchy - the first non-empty group wins:
      1. state + area + type + tenure
      2. state + area + type
      3. state + area                  (any type/tenure - keeps the local market rate)
      4. state + type + tenure
      5. state + type
      6. state
      7. whole dataset                 (safety net only)

    Level 3 is included because when a typed area exists but has no record of
    that property type, the area's own price level is still the most locally
    relevant market rate available.

    Suggested values are MEDIANS, which resist the strong right skew that
    remains in the cleaned data.
    """
    in_state = data["State"] == state
    is_type = data["Primary_Type"] == ptype
    is_tenure = data["Tenure"] == tenure
    candidates = []
    if area:
        in_area = data["Area"] == area
        candidates += [
            (1, f"{area}, {state} · {ptype} · {tenure}", in_state & in_area & is_type & is_tenure),
            (2, f"{area}, {state} · {ptype}", in_state & in_area & is_type),
            (3, f"{area}, {state} (all property types)", in_state & in_area),
        ]
    candidates += [
        (4, f"{state} · {ptype} · {tenure}", in_state & is_type & is_tenure),
        (5, f"{state} · {ptype}", in_state & is_type),
        (6, f"{state} (all property types)", in_state),
        (7, "whole 2025 dataset", pd.Series(True, index=data.index)),
    ]
    for level, label, mask in candidates:
        pool = data[mask]
        if not pool.empty:
            return {
                "psf": int(round(float(pool["Median_PSF"].median()))),
                "transactions": int(round(float(pool["Transactions"].median()))),
                "price_median": float(pool["Median_Price"].median()),
                "n": int(len(pool)),
                "level": level,
                "label": label,
                "pool": pool,
            }
    raise ValueError("no records available")


def fallback_message(reference: dict, area_text: str, area_matched: str | None,
                     state: str) -> tuple[str, str, str]:
    """One concise, honest sentence about where the suggested values came from."""
    level = reference["level"]
    if level <= 2:
        return ("good", "Exact area reference found",
                f"Suggested market values come from {reference['n']} record(s) for "
                f"<strong>{reference['label']}</strong>.")
    if level == 3:
        return ("info", "Area found, property type unavailable",
                f"No record of this property type exists for {area_matched}. "
                f"Suggested values use all {reference['n']} record(s) in "
                f"<strong>{reference['label']}</strong>.")
    if not area_text.strip():
        return ("info", "No specific area entered",
                f"Broader reference values are being used: "
                f"<strong>{reference['label']}</strong> "
                f"({reference['n']} records).")
    return ("warn", "Location not in the 2025 dataset",
            f"No record was found for “{area_text.strip()}”. Suggested market values "
            f"are based on the broader <strong>{reference['label']}</strong> group "
            f"({reference['n']} records).")


def psf_range(data: pd.DataFrame, state: str) -> dict:
    series = data.loc[data["State"] == state, "Median_PSF"]
    return {"median": float(series.median()), "min": float(series.min()),
            "max": float(series.max())}


def find_similar_records(pool: pd.DataFrame, psf: float, transactions: int,
                         top_n: int = 5) -> pd.DataFrame:
    """Rank records in a pool by closeness to the given market values.

    Score = 0.80 x |%diff in Median_PSF| + 0.20 x normalised |diff in Transactions|.
    The weights are a presentation choice reflecting the much stronger evidence
    for Median_PSF; they have not been statistically validated.
    """
    if pool.empty:
        return pool
    ranked = pool.copy()
    psf_gap = (ranked["Median_PSF"] - psf).abs() / max(psf, 1)
    span = ranked["Transactions"].max() - ranked["Transactions"].min()
    txn_gap = (ranked["Transactions"] - transactions).abs() / (span if span > 0 else 1)
    ranked["Similarity_Score"] = 0.80 * psf_gap + 0.20 * txn_gap
    return ranked.sort_values("Similarity_Score").head(top_n)


# ===========================================================================
# FIGURE GALLERY
# ===========================================================================
FIGURE_NOTES = {
    "fig01_raw_target_distribution.png": ("Figure 1 — Raw target distribution", "Is the median price suitable for modelling as-is?", "The raw distribution is extremely right-skewed with a tail reaching RM11.4 million.", "Skew of this size distorts distance-based fences, which is why outliers were assessed on the log scale.", "The log panel is a viewing transformation; no record is deleted."),
    "fig02_raw_numeric_boxplots.png": ("Figure 2 — Raw numeric attributes (log scale)", "How extreme are the values in each numeric column?", "All three columns are strongly right-skewed; a log axis is required to see the middle 50%.", "It shows why raw-scale IQR fences flag many legitimate high-value records.", "Points beyond the whiskers are statistical flags, not proven data errors."),
    "fig03_raw_state_counts.png": ("Figure 3 — Raw record count by state", "How evenly is the dataset spread across Malaysia?", "Selangor and Johor dominate; several territories have fewer than five records.", "The model generalises best to well-represented states.", "Small-sample states cannot support reliable conclusions."),
    "fig04_raw_tenure_type.png": ("Figure 4 — Raw tenure and property-type labels", "What cleaning do the categorical columns need?", "Tenure contains the same pair written in two orders; Type holds 46 multi-value strings.", "Both need standardising before they can be encoded.", "Primary_Type takes the first listed type, which is an operational assumption."),
    "fig05_raw_numeric_correlation.png": ("Figure 5 — Raw numeric correlation", "Which numeric column is most related to price?", "Median PSF is strongly related to price; transactions are almost unrelated.", "It sets the expectation that PSF will dominate the model.", "Correlation measures linear association only."),
    "fig06_raw_psf_vs_price.png": ("Figure 6 — Median PSF against Median Price (raw)", "What does the PSF–price relationship look like before cleaning?", "A positive relationship is visible once density and scale are handled.", "It justifies both the log-scale treatment and the outlier assessment.", "The zoomed panel changes the visible range only and deletes no records."),
    "fig07_outlier_before_after.png": ("Figure 7 — Price before and after outlier deletion", "What did log-IQR deletion actually remove?", "86 of 2,000 records were removed; the retained range is about RM90K–RM1.77M.", "It defines the market scope the model is valid for.", "Removed records are not necessarily errors — this is a scope restriction."),
    "fig08_price_distribution_clean.png": ("Figure 8 — Price distribution before and after cleaning", "How did the shape of the target change?", "Raw-price skewness falls from about 8.8 to about 1.7 while the bulk of the market is untouched.", "A less distorted target is easier for the models to fit.", "This is raw-price skewness, not log-price skewness (≈0.11)."),
    "fig09_category_donut.png": ("Figure 9 — Landed versus High-Rise share", "What is the composition of the cleaned dataset?", "About 71% of records are Landed and 29% High-Rise.", "Landed property types dominate the data the model learns from.", "This is a share of records, not of Malaysia's housing stock."),
    "fig10_state_counts_clean.png": ("Figure 10 — Records per state after cleaning", "Does cleaning change the geographic balance?", "Selangor and Johor still account for roughly 47% of records.", "Imbalance persists and remains a stated limitation.", "Counts are township-level records, not unique township names."),
    "fig11_state_violin.png": ("Figure 11 — Price distribution across major states", "How do prices differ between the best-represented states?", "Kuala Lumpur and Selangor sit clearly higher with long upper tails.", "Location is a strong price driver, supporting State as a model feature.", "States were chosen by record count, not by price; n is shown per state."),
    "fig12_type_boxplot.png": ("Figure 12 — Price by property type", "Which property types command higher prices?", "A clear ladder runs from Flats at the bottom to Bungalows at the top.", "Property type is a strong predictor and is retained as a model feature.", "Outlier markers are hidden for readability; the observations remain in the analysis."),
    "fig13_tenure_violin.png": ("Figure 13 — Price distribution by tenure", "Is tenure associated with price?", "Freehold records show a higher median and a longer upper tail than Leasehold.", "Tenure carries usable signal and is kept as a model feature.", "The Mixed group is small; it cannot support firm conclusions."),
    "fig14_psf_vs_price_clean.png": ("Figure 14 — Median PSF against Median Price by category", "How strongly is the market rate related to total price?", "A positive relationship holds in both categories, but at similar PSF levels Landed records reach higher total prices.", "PSF is the model's strongest input, yet it cannot explain price alone.", "Built-up size is unavailable, so the reason for the gap cannot be confirmed."),
    "fig15_correlation_heatmap.png": ("Figure 15 — Correlation matrix of encoded features", "How do the encoded features relate to price and each other?", "Median PSF dominates; property-type and tenure indicators carry moderate signal.", "It supports the feature set chosen for modelling.", "One-hot indicators of the same variable are negatively correlated by construction."),
    "fig16_all_feature_correlations.png": ("Figure 16 — Correlation with price, all encoded features", "Which individual categories move price up or down?", "Selangor and Kuala Lumpur push price up; Perak and Kedah pull it down.", "It quantifies the location and type effects seen in earlier charts.", "These are group differences, not causal effects; tiny categories are greyed out."),
    "fig17_top10_transactions.png": ("Figure 17 — Most-transacted townships", "Do the busiest townships also have the highest prices?", "The busiest townships are moderately priced rather than the most expensive.", "Transaction volume reflects market activity more than price level.", "No supply or income variables exist, so no cause can be established."),
    "fig18_model_comparison.png": ("Figure 18 — Test-set metric comparison", "How do the four models compare on unseen data?", "All three ensembles beat the Decision Tree baseline clearly; the top two are close together.", "It demonstrates that ensembling improves on a single tree.", "The highlighted model was selected on cross-validation, not on these test bars."),
    "fig19_cv_stability.png": ("Figure 19 — Overfitting and cross-validation stability", "Which model generalises most consistently?", "The baseline memorises its training data; the ensembles show much smaller train–test gaps.", "Stability matters as much as the average score when selecting a model.", "Lower point = better average CV RMSE; shorter error bar = greater stability."),
    "fig20_pred_actual_residual.png": ("Figure 20 — Predicted versus actual, and residuals", "Where does the selected model make its errors?", "Predictions track the perfect-prediction line in the mainstream range, but errors widen at higher prices.", "This heteroscedasticity supports the limitation about premium properties.", "Residuals should scatter randomly around zero; a widening funnel does not."),
    "fig21_split_importance.png": ("Figure 21 — Split importance of the selected model", "Which features does the model split on most often?", "Median PSF is used far more than any other feature.", "It gives a quick view of what the trees rely on.", "Split counts favour continuous features; Figure 22 is the fairer measure."),
    "fig22_permutation_importance.png": ("Figure 22 — Permutation importance on the test set", "How much does each of the five model inputs actually contribute?", "Shuffling Median PSF degrades performance far more than any other input.", "This is measured on unseen data and is the preferred interpretation.", "Importance is relative to this feature set; it is not a causal statement."),
}


def render_gallery(filenames: list[str]) -> None:
    shown = 0
    for name in filenames:
        path = FIGURES_DIR / name
        if not path.exists():
            continue
        title, question, finding, matters, caution = FIGURE_NOTES[name]
        st.markdown(f'<div class="mh-fig"><h4>{title}</h4>'
                    f'<p><strong>Question answered:</strong> {question}</p></div>',
                    unsafe_allow_html=True)
        st.image(str(path), use_container_width=True)
        st.markdown(f'<div class="mh-fig">'
                    f'<p><strong>Key finding:</strong> {finding}</p>'
                    f'<p><strong>Why it matters:</strong> {matters}</p>'
                    f'<p class="mh-fine"><strong>Caution:</strong> {caution}</p></div>',
                    unsafe_allow_html=True)
        st.markdown("")
        shown += 1
    if shown == 0:
        st.info("No figures found. Copy the `figures/` folder produced by the "
                "notebook next to `streamlit_app.py` to enable this gallery.")


# ===========================================================================
# PAGE 1 — PRICE PREDICTION
# ===========================================================================
def page_prediction(data: pd.DataFrame, results: pd.DataFrame) -> None:
    recommended = results.iloc[0]["Model"]

    if "form_version" not in st.session_state:
        st.session_state.form_version = 0
    version = st.session_state.form_version

    # ---------- Step 1: location ----------
    step_heading(1, "Location")
    with st.container(border=True):
        loc_left, loc_right = st.columns(2)
        with loc_left:
            state = st.selectbox("State", sorted(data["State"].unique()),
                                 key=f"state_{version}")
        with loc_right:
            area_text = st.text_input(
                "Area or township",
                placeholder="Example: Skudai, Taman Molek, Area X",
                key=f"area_{state}_{version}",
                help="Enter any area or township. When an exact location is "
                     "unavailable in the 2025 dataset, the application will use a "
                     "broader reference group from the selected state.")

        area_matched, suggestions = resolve_area(area_text, data, state)
        if suggestions:
            notice("info", "Did you mean",
                   "Close matches in the dataset: <strong>"
                   + "</strong>, <strong>".join(suggestions) + "</strong>. "
                   "You can keep your own spelling — the prediction still works.")
        st.markdown('<p class="mh-fine">Area is used to locate suitable reference '
                    'market values. It is not directly passed to the trained '
                    'model.</p>', unsafe_allow_html=True)

    # ---------- Step 2: property ----------
    step_heading(2, "Property details")
    with st.container(border=True):
        prop_left, prop_right = st.columns(2)
        with prop_left:
            ptype = st.selectbox("Property type",
                                 sorted(data["Primary_Type"].unique()),
                                 key=f"ptype_{version}")
        with prop_right:
            tenure = st.selectbox("Tenure", sorted(data["Tenure"].unique()),
                                  key=f"tenure_{version}")

    reference = derive_reference(data, state, area_matched, ptype, tenure)

    # ---------- Step 3: market features ----------
    step_heading(3, "Market features")
    with st.container(border=True):
        st.markdown('<p class="mh-fine">The suggested values come from the nearest '
                    'available 2025 reference group. They may be adjusted when you '
                    'have more appropriate market information. Both are real inputs '
                    'to the trained model.</p>', unsafe_allow_html=True)
        market_left, market_right = st.columns(2)
        widget_key = f"{state}_{area_matched or 'none'}_{ptype}_{tenure}_{version}"
        with market_left:
            psf = st.number_input(
                "Median price per square foot (RM)",
                min_value=1, max_value=100_000, step=10,
                value=int(reference["psf"]), key=f"psf_{widget_key}")
        with market_right:
            transactions = st.number_input(
                "Number of township transactions",
                min_value=1, max_value=100_000, step=1,
                value=int(reference["transactions"]), key=f"txn_{widget_key}")

        kind, label, body = fallback_message(reference, area_text, area_matched, state)
        notice(kind, label, body)

        bounds = psf_range(data, state)
        if psf < bounds["min"] or psf > bounds["max"]:
            notice("danger", "Outside the observed range",
                   f"The selected PSF of RM{psf:,} is outside the range observed in "
                   f"{state} (RM{bounds['min']:,.0f} – RM{bounds['max']:,.0f}). "
                   f"This is an extrapolation and should be interpreted cautiously.")

    # ---------- Step 4: model ----------
    step_heading(4, "Model")
    with st.container(border=True):
        with st.expander("Advanced model options"):
            st.markdown('<p class="mh-fine">The project conclusion uses the '
                        'cross-validation-selected model. The others are offered for '
                        'academic comparison; switching them does not change the '
                        "project's stated conclusion.</p>", unsafe_allow_html=True)
            option_labels = {}
            for _, row in results.iterrows():
                suffix = ""
                if "Baseline" in row["Model"]:
                    suffix = " — Baseline"
                elif row["Model"] == recommended:
                    suffix = " — Recommended by cross-validation"
                option_labels[f"{row['Model']}{suffix}"] = row["Model"]
            picked = st.selectbox("Model used for prediction", list(option_labels),
                                  index=0, key=f"model_{version}")
            model_name = option_labels[picked]

        metrics = results[results["Model"] == model_name].iloc[0]
        try:
            model = load_model(model_name)
        except FileNotFoundError as exc:
            st.error(f"**Model file not found:** `models/{exc}`  \n"
                     "Unzip the `models/` folder next to `streamlit_app.py` so the "
                     "`.pkl` files sit inside it.")
            st.stop()

        info_1, info_2, info_3 = st.columns(3)
        info_1.metric("Model used", model_name)
        info_2.metric("Test MAE", f"RM {metrics['MAE_test']/1000:,.1f}K")
        info_3.metric("Test R²", f"{metrics['R2_test']:.3f}")
        if model_name != recommended:
            notice("warn", "Not the recommended model",
                   f"You are using <strong>{model_name}</strong> for comparison. The "
                   f"project's selected model is <strong>{recommended}</strong>, "
                   f"chosen by cross-validation.")

    # ---------- Step 5: predict ----------
    st.markdown("")
    action_left, action_right = st.columns([3, 1])
    with action_left:
        predict = st.button("Predict township median price", type="primary",
                            use_container_width=True)
    with action_right:
        if st.button("Reset form", use_container_width=True):
            st.session_state.form_version += 1
            st.rerun()

    if not predict:
        return

    # Exactly the five trained features - Area is never included
    features = pd.DataFrame([{
        "State": state, "Tenure": tenure, "Primary_Type": ptype,
        "Median_PSF": psf, "Transactions": transactions}])[MODEL_FEATURES]
    try:
        prediction = float(model.predict(features)[0])
    except Exception:
        st.error("**The model could not produce a prediction for these inputs.** "
                 "The saved pipeline may expect different columns — re-export the "
                 "models from the notebook and try again.")
        return

    location = f"{area_text.strip()}, {state}" if area_text.strip() else state
    st.markdown(
        f"""
        <div class="mh-result">
            <div class="cap">Estimated township-level median price</div>
            <div class="amount">RM {prediction:,.0f}</div>
            <div class="mh-grid">
                <div><span>Location</span>{location}</div>
                <div><span>Property type</span>{ptype}</div>
                <div><span>Tenure</span>{tenure}</div>
                <div><span>Median PSF used</span>RM {psf:,}</div>
                <div><span>Transactions used</span>{transactions:,}</div>
                <div><span>Model</span>{model_name}</div>
                <div><span>Test MAE</span>RM {metrics['MAE_test']:,.0f}</div>
                <div><span>Test R²</span>{metrics['R2_test']:.3f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f'<p class="mh-fine" style="margin-top:12px">'
                f'Reference group used for the suggested market values: '
                f'<strong>{reference["label"]}</strong> '
                f'({reference["n"]} record(s), level {reference["level"]} of 7).'
                f'</p>', unsafe_allow_html=True)

    notice("info", "How to read this estimate",
           "This estimate is a township-level median benchmark based on the "
           "selected features and available 2025 market references. It is not a "
           "formal valuation of an individual property.<br><br>"
           "MAE is the model's average absolute error on the test dataset. It is "
           "not a confidence interval or a prediction range.")

    if reference["level"] >= 4 and area_text.strip():
        st.markdown('<p class="mh-fine">The entered location was not available in '
                    'the dataset, so the prediction uses broader market reference '
                    'values. The model has not learned this specific area — Area is '
                    'not one of its five features.</p>', unsafe_allow_html=True)

    st.markdown('<p class="mh-fine">Looking for comparable historical records? '
                'They are in the <strong>EDA &amp; Market Insights</strong> tab, '
                'under Market Data Explorer.</p>', unsafe_allow_html=True)


# ===========================================================================
# PAGE 2 — EDA & MARKET INSIGHTS
# ===========================================================================
def page_insights(data: pd.DataFrame) -> None:
    explorer, figures = st.tabs(["Market Data Explorer", "EDA figures"])

    with explorer:
        st.markdown("#### Historical 2025 dataset exploration")
        notice("info", "What this section is",
               "This section explores records contained in the project dataset. It "
               "is separate from the machine-learning prediction tool.")

        with st.container(border=True):
            f1, f2 = st.columns(2)
            with f1:
                state = st.selectbox("State", sorted(data["State"].unique()),
                                     key="explore_state")
                areas = ["All areas"] + sorted(
                    data.loc[data["State"] == state, "Area"].dropna().unique())
                area = st.selectbox("Area", areas, key=f"explore_area_{state}")
            with f2:
                types = ["All types"] + sorted(data["Primary_Type"].unique())
                ptype = st.selectbox("Property type", types, key="explore_type")
                tenures = ["All tenures"] + sorted(data["Tenure"].unique())
                tenure = st.selectbox("Tenure", tenures, key="explore_tenure")

        subset = data[data["State"] == state]
        if area != "All areas":
            subset = subset[subset["Area"] == area]
        if ptype != "All types":
            subset = subset[subset["Primary_Type"] == ptype]
        if tenure != "All tenures":
            subset = subset[subset["Tenure"] == tenure]

        if subset.empty:
            notice("warn", "No matching records",
                   "No record in the 2025 dataset matches this combination. "
                   "Try widening one of the filters.")
            return

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Matching records", f"{len(subset):,}")
        m2.metric("Median price", f"RM {subset['Median_Price'].median()/1000:,.0f}K")
        m3.metric("Median PSF", f"RM {subset['Median_PSF'].median():,.0f}")
        m4.metric("Median transactions", f"{subset['Transactions'].median():,.0f}")

        st.markdown("##### Matching records")
        table = subset[["Township", "Area", "Primary_Type", "Tenure",
                        "Median_Price", "Median_PSF", "Transactions"]].copy()
        st.dataframe(pd.DataFrame({
            "Township": table["Township"].str.title(),
            "Area": table["Area"],
            "Property type": table["Primary_Type"],
            "Tenure": table["Tenure"],
            "Median price (RM)": table["Median_Price"].map(lambda v: f"{v:,.0f}"),
            "Median PSF (RM)": table["Median_PSF"].map(lambda v: f"{v:,.0f}"),
            "Transactions": table["Transactions"],
        }), use_container_width=True, hide_index=True, height=320)

        st.markdown("##### Closest records to a chosen market rate")
        c1, c2 = st.columns(2)
        with c1:
            ref_psf = st.number_input("Median PSF to compare against (RM)",
                                      min_value=1, max_value=100_000, step=10,
                                      value=int(subset["Median_PSF"].median()),
                                      key="explore_psf")
        with c2:
            ref_txn = st.number_input("Transactions to compare against",
                                      min_value=1, max_value=100_000, step=1,
                                      value=int(subset["Transactions"].median()),
                                      key="explore_txn")

        similar = find_similar_records(subset, ref_psf, ref_txn)
        st.dataframe(pd.DataFrame({
            "Township": similar["Township"].str.title(),
            "Area": similar["Area"],
            "Median price (RM)": similar["Median_Price"].map(lambda v: f"{v:,.0f}"),
            "Median PSF (RM)": similar["Median_PSF"].map(lambda v: f"{v:,.0f}"),
            "Transactions": similar["Transactions"],
            "Similarity score": similar["Similarity_Score"].round(3),
        }), use_container_width=True, hide_index=True)
        st.markdown('<p class="mh-fine">Score = 0.80 × absolute percentage '
                    'difference in median PSF + 0.20 × normalised difference in '
                    'transactions. Lower is closer. The weights are a presentation '
                    'choice and have not been statistically validated.</p>',
                    unsafe_allow_html=True)

        if PLOTLY_AVAILABLE and len(similar):
            group_median = float(subset["Median_Price"].median())
            labels = ["Group median"] + [str(t).title() for t in similar["Township"]]
            values = [group_median] + [float(v) for v in similar["Median_Price"]]
            colours = ["#3B82F6"] + ["#64748B"] * len(similar)
            figure = go.Figure(go.Bar(
                x=labels, y=values, marker_color=colours,
                text=[f"RM {v/1000:,.0f}K" for v in values], textposition="outside",
                hovertemplate="%{x}<br>RM %{y:,.0f}<extra></extra>"))
            figure.update_layout(
                height=380, margin=dict(l=10, r=10, t=30, b=90),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC", size=12), showlegend=False,
                yaxis=dict(title="Median price (RM)",
                           gridcolor="rgba(148,163,184,0.18)", tickformat=",.0f"),
                xaxis=dict(tickangle=-30))
            st.plotly_chart(figure, use_container_width=True)

    with figures:
        sections = {
            "Data quality and outlier treatment":
                ["fig02_raw_numeric_boxplots.png", "fig04_raw_tenure_type.png",
                 "fig07_outlier_before_after.png"],
            "Housing-price distribution":
                ["fig01_raw_target_distribution.png",
                 "fig08_price_distribution_clean.png"],
            "State and property-type comparisons":
                ["fig03_raw_state_counts.png", "fig10_state_counts_clean.png",
                 "fig11_state_violin.png", "fig12_type_boxplot.png",
                 "fig09_category_donut.png"],
            "Tenure comparison": ["fig13_tenure_violin.png"],
            "PSF–price relationship":
                ["fig06_raw_psf_vs_price.png", "fig14_psf_vs_price_clean.png"],
            "Correlation analysis":
                ["fig05_raw_numeric_correlation.png",
                 "fig15_correlation_heatmap.png",
                 "fig16_all_feature_correlations.png"],
            "Transaction activity": ["fig17_top10_transactions.png"],
        }
        choice = st.selectbox("Section", list(sections), key="eda_section")
        render_gallery(sections[choice])


# ===========================================================================
# PAGE 3 — MODEL REPORT
# ===========================================================================
def page_model_report(results: pd.DataFrame) -> None:
    recommended = results.iloc[0]["Model"]
    table = results.copy()
    table["CV Rank"] = table["CV_RMSE_mean"].rank().astype(int)
    table["Test Rank"] = table["RMSE_test"].rank().astype(int)

    st.markdown("#### Four-model comparison")
    st.dataframe(pd.DataFrame({
        "Model": table["Model"],
        "CV Rank": table["CV Rank"],
        "CV RMSE": table["CV_RMSE_mean"].map(lambda v: f"RM {v/1000:,.1f}K"),
        "CV std": table["CV_RMSE_std"].map(lambda v: f"RM {v/1000:,.1f}K"),
        "Test Rank": table["Test Rank"],
        "Test RMSE": table["RMSE_test"].map(lambda v: f"RM {v/1000:,.1f}K"),
        "Test MAE": table["MAE_test"].map(lambda v: f"RM {v/1000:,.1f}K"),
        "Test R²": table["R2_test"].map(lambda v: f"{v:.3f}"),
        "Train–test gap": table["Gap_RMSE"].map(lambda v: f"RM {v/1000:,.1f}K"),
        "Selected by CV?": ["Yes" if m == recommended else ""
                            for m in table["Model"]],
    }), use_container_width=True, hide_index=True)

    st.markdown("#### Model-selection explanation")
    if len(results) >= 2:
        gap = abs(results.iloc[0]["CV_RMSE_mean"] - results.iloc[1]["CV_RMSE_mean"])
        notice("info", "The top two models are effectively tied",
               f"Their cross-validation RMSE differs by only "
               f"<strong>RM {gap:,.0f}</strong>, against a CV standard deviation of "
               f"about RM {results.iloc[0]['CV_RMSE_std']:,.0f}. "
               f"<strong>{recommended}</strong> was selected because it had the "
               f"lowest CV mean and the lower CV variability, while "
               f"<strong>{results.iloc[1]['Model']}</strong> achieved the stronger "
               f"hold-out metrics. Selection was made on cross-validation only, so "
               f"the test set remains an untouched estimate of unseen-data "
               f"performance.")

    render_gallery(["fig18_model_comparison.png", "fig19_cv_stability.png",
                    "fig20_pred_actual_residual.png",
                    "fig22_permutation_importance.png",
                    "fig21_split_importance.png"])

    st.markdown("#### Limitations")
    st.markdown("""
- **Scope restriction.** Deleting 86 log-IQR-flagged records narrows the model to
  roughly RM90K–RM1.77M. Removed records are not proven errors, so the model is
  simply not validated for very low-cost or luxury properties.
- **Heteroscedasticity.** Residual spread widens at higher predicted prices, so
  premium estimates are less reliable.
- **Area is not a model feature.** It supplies reference market values only. A
  genuinely area-aware model would require retraining all four pipelines with
  Area included and a suitable high-cardinality encoding.
- **`Primary_Type` assumption.** Assigned from the first listed type; about 18% of
  records carry a label whose token order alone determines the assigned type.
- **Missing attributes.** No bedroom count, land size, property age or amenity
  proximity — these absent features cap achievable accuracy.
- **Imbalance.** Records skew toward Selangor, Johor and terrace houses;
  small-sample states are unreliable.
""")


# ===========================================================================
# SIDEBAR AND MAIN
# ===========================================================================
def render_sidebar(recommended: str, record_count: int) -> None:
    with st.sidebar:
        st.markdown("### About this prototype")
        st.write("Estimates the **median house price of a township** from its "
                 "state, property type, tenure and market rate per square foot.")
        st.markdown("---")
        st.markdown(f"""
        **Dataset year** · 2025
        **Records after cleaning** · {record_count:,}
        **Prediction level** · Township-level median
        **Selected model** · {recommended} (by cross-validation)
        """)
        st.markdown("---")
        st.markdown("**How location is used**")
        st.write("You may type any area or township, including one that is not in "
                 "the dataset. The text is used to find suitable reference market "
                 "values. It is **not** passed to the model — the model receives "
                 "state, tenure, property type, median PSF and transactions.")
        st.markdown("---")
        st.markdown("**Main limitation**")
        st.write("The median price per square foot must be reasonable for the "
                 "location. The model refines a known market rate rather than "
                 "discovering prices without market input.")
        st.markdown("---")
        st.caption("Academic prototype for BMDS2003 coursework. Built on a static "
                   "2025 dataset, not live market data. Estimates are township-level "
                   "medians, not valuations of individual properties, and must not "
                   "be used for real financial decisions.")


def main() -> None:
    inject_css()
    results, data = load_resources()
    recommended = results.iloc[0]["Model"]
    render_sidebar(recommended, len(data))
    render_hero()

    prediction_tab, insights_tab, report_tab = st.tabs(
        ["Price Prediction", "EDA & Market Insights", "Model Report"])
    with prediction_tab:
        page_prediction(data, results)
    with insights_tab:
        page_insights(data)
    with report_tab:
        page_model_report(results)


main()

st.markdown('<div class="mh-footer">BMDS2003 Data Science Group Assignment | '
            'Academic Prototype | Data source: Malaysia Housing Prices 2025</div>',
            unsafe_allow_html=True)
