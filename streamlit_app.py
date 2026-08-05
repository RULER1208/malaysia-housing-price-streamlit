"""
BMDS2003 Data Science - Deployment Prototype
Malaysia Housing Median Price Estimator

Run locally:  streamlit run streamlit_app.py

MODEL CONTRACT - six trained features:
    State, Area_Key, Tenure, Primary_Type, Median_PSF, Transactions

Area IS a trained feature, supplied as a state-qualified key ("JOHOR | SKUDAI").
The same area_preprocessing module is used by the notebook at training time and
by this app at prediction time, so cleaning can never drift. Township is
excluded because it is near-unique per row.

LAYOUT
------
Top navigation: the top-level st.tabs bar is styled as a fixed dark navy
banner pinned to row 0, so it shares the strip with Streamlit's own toolbar
icons. There is no second banner or hero section below it.

Prediction page: two columns. Left is the input form, right is the result -
the dark price card, a compact "Prediction inputs used" card, a collapsed
"Technical details" expander, and a short disclaimer.

REFERENCE VALUES
----------------
Median PSF and Transactions are prediction inputs the user sets deliberately.
They use a FIXED, dataset-wide default and a STABLE widget key, so changing
State / Area / Type / Tenure never silently rewrites a value already entered.
The "typical" figure for the current selection is shown beside the label, for
information only.
"""
from __future__ import annotations
from pathlib import Path
import inspect
import joblib
import pandas as pd
import streamlit as st

from area_preprocessing import clean_area_name, clean_state_name, create_area_key, display_name

st.set_page_config(
    page_title="Malaysia Housing Price Estimator",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "malaysia_house_price_cleaned_with_area.csv"
RESULTS_PATH = APP_DIR / "model_results.csv"
MODELS_DIR = APP_DIR / "models"
FIGURES_DIR = APP_DIR / "figures"
MODEL_FEATURES = ["State", "Area_Key", "Tenure", "Primary_Type", "Median_PSF", "Transactions"]

# Streamlit >= 1.45 lets a selectbox accept a typed value outside its option
# list. requirements.txt pins streamlit>=1.45; this check keeps the app from
# crashing outright on an older local install.
SELECTBOX_ACCEPTS_NEW = "accept_new_options" in inspect.signature(st.selectbox).parameters
NO_AREA = "— No specific area —"

st.markdown(r"""
<style>
:root {
    --bg:#F6F8FB;
    --card:#FFFFFF;
    --navy:#15243A;
    --blue:#2F6FED;
    --text:#172033;
    --muted:#667085;
    --border:#E2E7EF;
    --green:#18875D;
    --amber:#C47A10;
    --soft:#EEF4FF;
    --radius:14px;
    --shadow:0 6px 22px rgba(24,49,83,.06);
    --mono:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace;
}
.stApp { background:var(--bg); color:var(--text); }
/* padding-top clears the fixed navigation banner */
.block-container { max-width:1180px; padding-top:80px; padding-bottom:2rem; }
html,body,[class*="css"] { font-family:Inter,"Segoe UI",Roboto,Arial,sans-serif }
h1,h2,h3,h4 { color:var(--navy) }

/* ---------- Top navigation banner ----------
   The top-level tab list IS the banner: fixed at top:0 and 58px tall so it
   occupies the same strip as Streamlit's Share / star / GitHub icons, whose
   own header is made transparent so this bar shows through behind them.
   padding-right reserves space so the nav can never slide under the toolbar.
   The Insights page uses a radio, not a second st.tabs, so these rules can
   target the navigation unambiguously without fragile scoping. */
.stTabs [data-baseweb="tab-list"] {
    position:fixed; top:0; left:0; right:0;
    height:58px; z-index:999989;
    display:flex; align-items:center; gap:6px;
    padding:0 300px 0 22px;
    background:var(--navy);
    border:none; border-radius:0;
    box-shadow:0 1px 0 rgba(255,255,255,.06);
}
.stTabs [data-baseweb="tab-list"]::before {
    content:"\2302\00a0\00a0Housing Price Estimator";
    font-size:1.02rem; font-weight:700; color:#FFFFFF;
    white-space:nowrap; margin-right:22px;
}
.stTabs [data-baseweb="tab"] {
    height:34px; padding:0 15px; border-radius:999px;
    color:#B9C8DF; font-weight:600; font-size:.9rem;
    background:transparent; border:none;
}
.stTabs [data-baseweb="tab"]:hover { color:#FFFFFF; background:rgba(255,255,255,.09); }
.stTabs [aria-selected="true"] {
    color:var(--navy)!important; background:#FFFFFF!important;
    border-bottom:none!important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display:none!important; }

/* Let the banner show through Streamlit's header; keep its icons visible. */
header[data-testid="stHeader"] { background:transparent!important; height:58px; }
[data-testid="stToolbar"] { color:#E8EEF8!important; }
[data-testid="stToolbar"] svg { fill:#E8EEF8!important; color:#E8EEF8!important; }
[data-testid="stToolbar"] button, [data-testid="stToolbar"] a,
[data-testid="stToolbar"] span, [data-testid="stToolbar"] p { color:#E8EEF8!important; }

/* Insights page segmented control (replaces a second st.tabs) */
div[role="radiogroup"] { gap:8px; }

/* ---------- Left input card ---------- */
.mh-panel-title {
    display:flex; align-items:center; gap:9px;
    font-size:1rem; font-weight:700; color:var(--navy); margin:2px 0 14px 0;
}
.mh-label {
    display:flex; align-items:baseline; justify-content:space-between;
    font-family:var(--mono); font-size:.72rem; letter-spacing:.1em;
    color:var(--muted); margin:2px 0 5px 0; text-transform:uppercase;
}
.mh-label .hint {
    font-family:Inter,sans-serif; letter-spacing:0;
    text-transform:none; font-size:.78rem;
}
.mh-rule { border:none; border-top:1px solid var(--border); margin:16px 0 14px 0; }

/* ---------- Right result column ---------- */
.mh-result {
    background:var(--navy); border-radius:var(--radius);
    padding:26px 26px 24px; box-shadow:var(--shadow);
}
.mh-result .cap {
    font-family:var(--mono); font-size:.72rem; letter-spacing:.14em;
    color:#9FB4D4; text-transform:uppercase;
}
.mh-result .price {
    font-size:2.7rem; font-weight:750; color:#FFFFFF;
    margin:8px 0 6px; line-height:1.05; letter-spacing:-.01em;
}
.mh-result .sub { color:#C9D7EC; font-size:.95rem; }
.mh-result .rule { border-top:1px solid rgba(255,255,255,.16); margin:18px 0 14px; }
.mh-stats { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }
.mh-stats .k {
    font-family:var(--mono); font-size:.68rem; letter-spacing:.12em;
    color:#9FB4D4; text-transform:uppercase; margin-bottom:3px;
}
.mh-stats .v { font-family:var(--mono); font-size:.94rem; font-weight:700; color:#FFFFFF; }

/* Compact "Prediction inputs used" card */
.mh-used {
    background:var(--card); border:1px solid var(--border);
    border-radius:var(--radius); padding:16px 18px;
    box-shadow:var(--shadow); margin-top:12px;
}
.mh-used h4 { margin:0 0 10px 0; font-size:.94rem; font-weight:700; color:var(--navy); }
.mh-used .row {
    display:flex; justify-content:space-between; align-items:baseline;
    padding:6px 0; font-size:.9rem;
}
.mh-used .row + .row { border-top:1px dashed var(--border); }
.mh-used .k { color:var(--muted); }
.mh-used .v { font-weight:650; color:var(--text); font-family:var(--mono); }

.mh-note {
    display:flex; gap:9px; align-items:flex-start; margin-top:12px;
    background:var(--soft); border:1px solid var(--border); border-radius:var(--radius);
    padding:12px 15px; font-size:.82rem; color:var(--muted); line-height:1.5;
}
.mh-empty {
    background:var(--card); border:1px dashed #CFD8E6; border-radius:var(--radius);
    padding:52px 24px; text-align:center; color:var(--muted); box-shadow:var(--shadow);
}
.mh-empty .icon { font-size:2rem; margin-bottom:10px; opacity:.7; }
.mh-empty b { color:var(--navy); display:block; margin-bottom:4px; font-size:1rem; }

.stButton>button { min-height:44px; border-radius:11px; font-weight:650; }
.stButton>button[kind="primary"] { background:var(--blue); border-color:var(--blue); }
.stButton>button[kind="secondary"] {
    background:#FFFFFF; color:var(--text); border:1px solid var(--border);
}
div[data-testid="stMetric"] {
    background:white; border:1px solid var(--border);
    border-radius:13px; padding:13px 15px;
}
.mh-fig-caption { font-size:.86rem; color:var(--muted); margin:2px 0 18px 0; line-height:1.4; }

@media(max-width:820px){
    .block-container{padding-left:12px;padding-right:12px;padding-top:74px;}
    .stTabs [data-baseweb="tab-list"]{height:54px;padding:0 130px 0 12px;}
    .stTabs [data-baseweb="tab-list"]::before{font-size:.9rem;margin-right:10px;}
    .stTabs [data-baseweb="tab"]{padding:0 10px;font-size:.84rem;}
    header[data-testid="stHeader"]{height:54px;}
    .mh-result .price{font-size:2.05rem;}
    .mh-stats{grid-template-columns:1fr 1fr;}
}
@media(max-width:560px){
    /* Drop the brand first so the three nav pills always stay reachable */
    .stTabs [data-baseweb="tab-list"]::before{display:none;}
}
@media(prefers-reduced-motion:reduce){
    *,*::before,*::after{animation-duration:.01ms!important;transition-duration:.01ms!important;}
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data(show_spinner=False)
def load_results():
    results = pd.read_csv(RESULTS_PATH)
    return results.sort_values(["Group_CV_RMSE_mean", "Group_CV_RMSE_std"]).reset_index(drop=True)


@st.cache_resource(show_spinner="Loading prediction model...")
def load_model(name):
    filename = name.split(" (")[0].lower().replace(" ", "_") + ".pkl"
    return joblib.load(MODELS_DIR / filename)


def field_label(text: str, hint: str = "") -> None:
    """Small uppercase field label, with an optional right-aligned hint."""
    right = f'<span class="hint">{hint}</span>' if hint else ""
    st.markdown(f'<div class="mh-label"><span>{text}</span>{right}</div>',
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# AREA RESOLUTION AND REFERENCE LOOKUP
# ---------------------------------------------------------------------------
def reset_prediction_form():
    """Clear every prediction widget and remove the previous result."""
    for key in list(st.session_state.keys()):
        if key.startswith("pred_"):
            del st.session_state[key]


def known_areas_for_state(data: pd.DataFrame, state: str) -> list[str]:
    subset = data.loc[data["State"] == state, "Area_Clean"].dropna().unique()
    return sorted({display_name(a) for a in subset})


def resolve_known_area(data, state, area_text):
    clean = clean_area_name(area_text)
    subset = data[data["State"] == state]
    match = subset[subset["Area_Clean"] == clean]
    return (clean if len(match) else None), match


def derive_reference(data, state, area_clean, ptype, tenure):
    """Median PSF / Transactions for the closest matching group - INFORMATION
    ONLY. Never used as a widget's live default (see module docstring)."""
    s = data["State"].eq(state); t = data["Primary_Type"].eq(ptype); n = data["Tenure"].eq(tenure)
    candidates = []
    if area_clean:
        a = data["Area_Clean"].eq(area_clean)
        candidates += [
            ("Local area", s & a & t & n),
            ("Local area + property type", s & a & t),
            ("Local area market", s & a),
        ]
    candidates += [
        (f"{state} property reference", s & t & n),
        (f"{state} property market", s & t),
        (f"{state} market", s),
        ("National dataset", pd.Series(True, index=data.index)),
    ]
    for label, mask in candidates:
        pool = data[mask]
        if len(pool):
            return {"label": label, "psf": int(round(pool["Median_PSF"].median())),
                    "transactions": int(round(pool["Transactions"].median())),
                    "n": len(pool), "pool": pool}
    raise ValueError("No reference data available")


# ---------------------------------------------------------------------------
# PAGE 1 - PRICE PREDICTION
# ---------------------------------------------------------------------------
def prediction_page(data, results):
    recommended = results.iloc[0]["Model"]
    # Fixed, selection-independent defaults so no dropdown ever rewrites them.
    default_psf = int(round(data["Median_PSF"].median()))
    default_txn = int(round(data["Transactions"].median()))
    psf_min, psf_max = int(data["Median_PSF"].min()), int(data["Median_PSF"].max())

    left, right = st.columns([1, 1], gap="large")

    # ---------------- LEFT: inputs ----------------
    with left:
        with st.container(border=True):
            st.markdown('<div class="mh-panel-title">⚙ Inputs</div>', unsafe_allow_html=True)

            field_label("State")
            state = st.selectbox("State", sorted(data["State"].unique()),
                                 key="pred_state", label_visibility="collapsed")

            field_label("Area")
            area_options = [NO_AREA] + known_areas_for_state(data, state)
            if SELECTBOX_ACCEPTS_NEW:
                area_choice = st.selectbox(
                    "Area", area_options, key="pred_area", label_visibility="collapsed",
                    accept_new_options=True,
                    help="Pick a known area, or type one that is not listed. The "
                         "model is trained on Area, not Township; an unrecognised "
                         "area falls back safely to broader market values.")
            else:
                OTHER = "Other (type below)"
                picked = st.selectbox("Area", area_options + [OTHER],
                                      key="pred_area_select", label_visibility="collapsed")
                area_choice = (st.text_input("Type the area", key="pred_area_text")
                               if picked == OTHER else picked)
            area_text = "" if area_choice == NO_AREA else str(area_choice or "")

            c1, c2 = st.columns(2)
            with c1:
                field_label("Type")
                ptype = st.selectbox("Property type", sorted(data["Primary_Type"].unique()),
                                     key="pred_type", label_visibility="collapsed")
            with c2:
                field_label("Tenure")
                tenure = st.selectbox("Tenure", sorted(data["Tenure"].unique()),
                                      key="pred_tenure", label_visibility="collapsed")

            known_area, _ = resolve_known_area(data, state, area_text)
            reference = derive_reference(data, state, known_area, ptype, tenure)

            st.markdown('<hr class="mh-rule">', unsafe_allow_html=True)

            field_label("Median PSF (RM)", f"typical {reference['psf']:,}")
            psf = st.number_input("Median price per square foot (RM)", min_value=1, step=10,
                                  value=default_psf, key="pred_psf",
                                  label_visibility="collapsed")

            field_label("Transactions", f"typical {reference['transactions']:,}")
            transactions = st.number_input("Transactions", min_value=0, step=1,
                                           value=default_txn, key="pred_txn",
                                           label_visibility="collapsed")

            field_label("Model")
            labels, mapping = [], {}
            for _, row in results.iterrows():
                suffix = " — Recommended" if row["Model"] == recommended else ""
                labels.append(row["Model"] + suffix)
                mapping[row["Model"] + suffix] = row["Model"]
            picked_model = st.selectbox("Model", labels, index=0, key="pred_model",
                                        label_visibility="collapsed")
            model_name = mapping[picked_model]

            if psf < psf_min or psf > psf_max:
                st.warning(
                    f"RM {psf:,} is outside the observed range (RM {psf_min:,}–"
                    f"RM {psf_max:,}). Tree-based models do not extrapolate beyond "
                    f"values seen in training, so the estimate will stop responding "
                    f"past a point — treat an extreme input as unreliable.")

            b1, b2 = st.columns([2, 1])
            predict = b1.button("Predict Price", type="primary", use_container_width=True)
            b2.button("Reset", type="secondary", use_container_width=True,
                      on_click=reset_prediction_form, key="reset_prediction")

    # ---------------- RIGHT: result ----------------
    with right:
        if not predict:
            st.markdown(
                '<div class="mh-empty"><div class="icon">⌂</div>'
                '<b>Your estimate will appear here.</b>'
                'Complete the inputs on the left, then select Predict Price.</div>',
                unsafe_allow_html=True)
            return

        model = load_model(model_name)
        area_key = create_area_key(state, area_text)
        features = pd.DataFrame([{"State": state, "Area_Key": area_key, "Tenure": tenure,
                                  "Primary_Type": ptype, "Median_PSF": psf,
                                  "Transactions": transactions}])[MODEL_FEATURES]
        with st.spinner("Calculating estimate..."):
            prediction = float(model.predict(features)[0])
        metrics = results[results["Model"] == model_name].iloc[0]
        try:
            encoder = model.named_steps["preprocess"].named_transformers_["cat"].named_steps["encoder"]
            area_position = ["State", "Area_Key", "Tenure", "Primary_Type"].index("Area_Key")
            area_seen = area_key in set(encoder.categories_[area_position])
        except Exception:
            area_seen = bool(known_area)

        location = f"{area_text.strip()}, {state}" if area_text.strip() else state
        st.markdown(f'''
        <div class="mh-result">
            <div class="cap">Estimated median price</div>
            <div class="price">RM {prediction:,.0f}</div>
            <div class="sub">{location} · {ptype} · {tenure}</div>
            <div class="rule"></div>
            <div class="mh-stats">
                <div><div class="k">Model</div><div class="v">{model_name}</div></div>
                <div><div class="k">Test MAE</div><div class="v">RM {metrics['MAE_test']/1000:,.1f}K</div></div>
                <div><div class="k">Test R²</div><div class="v">{metrics['R2_test']:.3f}</div></div>
            </div>
        </div>''', unsafe_allow_html=True)

        st.markdown(f'''
        <div class="mh-used">
            <h4>Prediction inputs used</h4>
            <div class="row"><span class="k">Median PSF</span><span class="v">RM {psf:,}</span></div>
            <div class="row"><span class="k">Transactions</span><span class="v">{transactions:,}</span></div>
        </div>''', unsafe_allow_html=True)

        st.markdown(
            '<div class="mh-note">ⓘ&nbsp;<span>Indicative township-level estimate '
            'based on the static 2025 dataset. Not a formal property valuation.'
            '</span></div>',
            unsafe_allow_html=True)

        # Reference-lookup provenance is diagnostic, not part of the prediction
        # story, so it is collapsed by default.
        with st.expander("Technical details"):
            st.markdown(
                f"- **Reference source:** {reference['label']}\n"
                f"- **Reference records:** {reference['n']:,}\n"
                f"- **Area recognised:** {'Yes' if known_area else 'No'}\n"
                f"- **Area seen during model training:** {'Yes' if area_seen else 'No'}\n"
                f"- **Broader state reference values used:** "
                f"{'No' if known_area else 'Yes'}")
            st.caption("These describe where the suggested typical values came "
                       "from. The prediction itself used exactly the Median PSF "
                       "and Transactions shown above.")


FIGURE_GROUPS = {
    "Data quality": [
        ("fig01_raw_target_distribution.png", "House prices are heavily skewed — most townships are affordable, a few are very expensive."),
        ("fig02_raw_numeric_boxplots.png", "Boxplots of price, PSF and transactions before cleaning — dots beyond the whiskers are candidate outliers."),
        ("fig10_outlier_before_after.png", "Extreme values removed by outlier cleaning, before vs after."),
        ("fig11_price_distribution_before_after.png", "Price distribution becomes more balanced after cleaning."),
    ],
    "Area quality and coverage": [
        ("fig09_area_labels_before_after.png", "Area name spelling and formatting before vs after standardisation."),
        ("fig12_area_repeated_singleton.png", "Many areas appear only once in the data — a real limitation for those locations."),
        ("fig13_area_frequency_distribution.png", "How many records each area has — most have very few."),
        ("fig14_top20_areas.png", "The 20 areas with the most records in the dataset."),
        ("fig15_area_cleaning_findings.png", "Summary of issues found and fixed while cleaning area names."),
        ("fig16_area_price_distribution.png", "How median price varies across different areas."),
    ],
    "Location and property": [
        ("fig18_state_counts_clean.png", "Number of records per state after cleaning."),
        ("fig19_state_price_distribution.png", "Median price differs a lot from state to state."),
        ("fig20_property_type_price.png", "Bungalows and semi-detached homes cost more than flats and apartments, on average."),
        ("fig21_tenure_price.png", "Freehold properties tend to have a different price profile than leasehold."),
    ],
    "Relationships": [
        ("fig22_psf_price_by_category.png", "Price per square foot is one of the strongest single predictors of price."),
        ("fig23_feature_correlation.png", "How strongly each feature relates to price and to the other features."),
        ("fig24_top_transactions.png", "The most actively traded townships in 2025."),
    ],
}


# ---------------------------------------------------------------------------
# PAGE 2 - MARKET INSIGHTS
# ---------------------------------------------------------------------------
def insights_page(data):
    # A radio is used rather than a second st.tabs so the fixed top-bar CSS
    # can target the navigation tabs without also restyling this control.
    view = st.radio("View", ["Market Explorer", "Visual Insights"],
                    horizontal=True, label_visibility="collapsed", key="insights_view")
    if view == "Market Explorer":
        st.markdown("#### Historical 2025 dataset exploration")
        a, b, c, d = st.columns(4)
        state = a.selectbox("State", ["All"] + sorted(data["State"].unique()), key="ex_state")
        area = b.selectbox("Area", ["All"] + sorted(data["Area_Clean"].unique()), key="ex_area")
        ptype = c.selectbox("Property type", ["All"] + sorted(data["Primary_Type"].unique()), key="ex_type")
        tenure = d.selectbox("Tenure", ["All"] + sorted(data["Tenure"].unique()), key="ex_tenure")
        subset = data.copy()
        if state != "All": subset = subset[subset["State"] == state]
        if area != "All": subset = subset[subset["Area_Clean"] == area]
        if ptype != "All": subset = subset[subset["Primary_Type"] == ptype]
        if tenure != "All": subset = subset[subset["Tenure"] == tenure]
        if len(subset) == 0:
            st.warning("No historical records match these filters.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Records", f"{len(subset):,}")
            m2.metric("Median price", f"RM {subset['Median_Price'].median()/1000:,.0f}K")
            m3.metric("Median PSF", f"RM {subset['Median_PSF'].median():,.0f}")
            m4.metric("Median transactions", f"{subset['Transactions'].median():,.0f}")
            with st.expander("View matching historical records"):
                show = subset[["Township", "Area_Clean", "State", "Primary_Type",
                               "Tenure", "Median_Price", "Median_PSF", "Transactions"]].copy()
                st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.caption("Short explanations are included under each chart for presentation use.")
        group = st.selectbox("Insight category", list(FIGURE_GROUPS))
        for filename, caption in FIGURE_GROUPS[group]:
            path = FIGURES_DIR / filename
            if path.exists():
                st.image(str(path), use_container_width=True)
                st.markdown(f'<p class="mh-fig-caption">{caption}</p>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE 3 - MODEL REPORT
# ---------------------------------------------------------------------------
def model_report_page(results):
    recommended = results.iloc[0]
    st.subheader("Model Report")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected model", recommended["Model"])
    m2.metric("Group CV RMSE", f"RM {recommended['Group_CV_RMSE_mean']/1000:,.1f}K")
    m3.metric("Test MAE", f"RM {recommended['MAE_test']/1000:,.1f}K")
    m4.metric("Test R²", f"{recommended['R2_test']:.3f}")
    st.dataframe(results, use_container_width=True, hide_index=True)
    sections = {
        "Area ablation": [("fig25_area_ablation.png", "Does including Area actually improve predictions? Compared with and without it.")],
        "Performance": [("fig26_model_test_comparison.png", "How the four models compare on data they have not seen."),
                        ("fig27_cv_stability_overfitting.png", "Checking that each model performs consistently, not just well on one lucky split.")],
        "Diagnostics": [("fig28_prediction_diagnostics.png", "Where the selected model's predictions are most and least accurate.")],
        "Importance": [("fig29_permutation_importance.png", "Which inputs the model actually relies on, tested on unseen data."),
                       ("fig30_aggregated_split_importance.png", "Which inputs the model used most often while learning.")],
    }
    section = st.selectbox("Report section", list(sections))
    for filename, caption in sections[section]:
        path = FIGURES_DIR / filename
        if path.exists():
            st.image(str(path), use_container_width=True)
            st.markdown(f'<p class="mh-fig-caption">{caption}</p>', unsafe_allow_html=True)
    with st.expander("Key limitations"):
        st.markdown("- Some Areas contain very few records.\n"
                    "- Completely unseen Areas are harder than previously observed Areas.\n"
                    "- Median PSF remains required market information.\n"
                    "- The dataset is a static 2025 snapshot.\n"
                    "- Results are township-level medians, not individual-property valuations.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    missing = [p.name for p in [DATA_PATH, RESULTS_PATH] if not p.exists()]
    if missing:
        st.error("Missing required files: " + ", ".join(missing)); st.stop()
    data = load_data(); results = load_results()
    pred, insights, report = st.tabs(["Prediction", "Insights", "Model Report"])
    with pred: prediction_page(data, results)
    with insights: insights_page(data)
    with report: model_report_page(results)


main()
