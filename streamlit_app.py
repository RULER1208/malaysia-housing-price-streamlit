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

DESIGN NOTE ON REFERENCE VALUES (v2)
-------------------------------------
Median PSF and Transactions are prediction inputs the user sets deliberately.
They therefore use a FIXED, state-independent default (the dataset-wide
median) and a STABLE widget key, so choosing a different State / Area /
Property type / Tenure never silently rewrites a value the user already set.
A typical-value caption is shown separately, for information only.
"""
from __future__ import annotations
from pathlib import Path
import base64
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
ASSETS_DIR = APP_DIR / "assets"
BANNER_PATH = ASSETS_DIR / "malaysia_housing_banner.png"
MODEL_FEATURES = ["State", "Area_Key", "Tenure", "Primary_Type", "Median_PSF", "Transactions"]

# Streamlit >= 1.45 lets a selectbox accept a typed value outside its option
# list. requirements.txt pins streamlit>=1.45, but this keeps the app from
# crashing outright on an older local install.
SELECTBOX_ACCEPTS_NEW = "accept_new_options" in inspect.signature(st.selectbox).parameters
NO_AREA = "— No specific area —"

st.markdown(r"""
<style>
:root {
    --bg:#F6F8FB;
    --card:#FFFFFF;
    --text:#172033;
    --muted:#667085;
    --blue:#2F6FED;
    --navy:#183153;
    --green:#18875D;
    --amber:#C47A10;
    --red:#C63C4A;
    --border:#E2E7EF;
    --soft:#EEF4FF;
    --radius:16px;
    --shadow:0 8px 28px rgba(24,49,83,.07);
}
.stApp { background:var(--bg); color:var(--text); }
.block-container { max-width:1160px; padding-top:1.4rem; padding-bottom:2rem; }
html,body,[class*="css"] { font-family:Inter,"Segoe UI",Roboto,Arial,sans-serif }
h1,h2,h3,h4 { color:var(--navy) }

.mh-header {
    display:flex; justify-content:space-between; align-items:center;
    padding:18px 22px; background:white; border:1px solid var(--border);
    border-radius:var(--radius); box-shadow:var(--shadow); margin-bottom:18px;
}
.mh-brand { font-size:1.22rem; font-weight:750; color:var(--navy); }
.mh-brand small { display:block; font-size:.82rem; color:var(--muted); font-weight:500; margin-top:2px; }
.mh-chip { display:inline-flex; align-items:center; padding:5px 9px; border-radius:999px; font-size:.82rem; font-weight:650; }
.mh-chip.local { background:#EAF7F1; color:#116B48; }
.mh-chip.fallback { background:#FFF6E6; color:#8A5300; }

/* Hero banner - Price Prediction page only */
.mh-hero {
    position:relative; border-radius:var(--radius); overflow:hidden;
    margin-bottom:18px; box-shadow:var(--shadow); border:1px solid var(--border);
    background-size:cover; background-position:center;
    padding:34px 26px; min-height:118px; display:flex; flex-direction:column;
    justify-content:center;
}
.mh-hero::before {
    content:""; position:absolute; inset:0;
    background:linear-gradient(100deg, rgba(23,32,51,.82) 10%, rgba(23,32,51,.45) 75%);
}
.mh-hero .mh-hero-content { position:relative; z-index:1; }
.mh-hero h2 { color:#FFFFFF; margin:0 0 4px 0; font-size:1.5rem; }
.mh-hero p { color:#E7ECF6; margin:0; font-size:.92rem; }

.mh-card { background:white; border:1px solid var(--border); border-radius:var(--radius); padding:22px; box-shadow:var(--shadow); }
.mh-result {
    background:linear-gradient(135deg,#EAF7F1,#EEF4FF); border:1px solid #BFE5D2;
    border-radius:18px; padding:28px 26px; animation:fadeUp .24s ease both;
}
.mh-result .label { font-size:.82rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; }
.mh-result .price { font-size:2.6rem; font-weight:760; color:var(--navy); margin:4px 0 6px; }
.mh-result .subline { font-size:.98rem; color:var(--text); margin-bottom:14px; }
.mh-result .basis {
    background:rgba(255,255,255,.65); border:1px solid rgba(191,229,210,.9);
    border-radius:12px; padding:12px 14px; font-size:.9rem; color:var(--text); margin-top:6px;
}
.mh-result .disclaimer {
    display:flex; gap:8px; align-items:flex-start; margin-top:14px;
    font-size:.82rem; color:var(--muted); line-height:1.45;
}
.mh-empty { text-align:center; padding:36px 18px; color:var(--muted); }
.mh-empty .icon { font-size:2rem; margin-bottom:8px; }

.stButton>button { min-height:44px; border-radius:11px; font-weight:650; transition:transform .18s ease,box-shadow .18s ease; }
.stButton>button:hover { transform:translateY(-1px) }
.stButton>button[kind="primary"] { background:var(--blue); border-color:var(--blue); }
.stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid var(--border); }
.stTabs [data-baseweb="tab"] { height:46px; color:var(--muted); font-weight:650; }
.stTabs [aria-selected="true"] { color:var(--blue)!important; border-bottom:2px solid var(--blue); }
div[data-testid="stMetric"] { background:white; border:1px solid var(--border); border-radius:13px; padding:13px 15px; }

.mh-fig-caption { font-size:.86rem; color:var(--muted); margin:2px 0 18px 0; line-height:1.4; }

@keyframes fadeUp { from{opacity:0;transform:translateY(8px);} to{opacity:1;transform:translateY(0);} }
@media(max-width:700px){
    .block-container{padding-left:12px;padding-right:12px;}
    .mh-header{padding:15px;}
    .mh-result .price{font-size:2rem;}
    .mh-hero{padding:22px 18px;min-height:96px;}
    .mh-hero h2{font-size:1.2rem;}
}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;transition-duration:.01ms!important;}}
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


@st.cache_data(show_spinner=False)
def encode_image(path_str: str) -> str | None:
    """Base64-encode a local image, or return None if it is missing."""
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return None
    try:
        return base64.b64encode(path.read_bytes()).decode()
    except OSError:
        return None


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
            ("Local area reference", s & a & t & n),
            ("Local area + property type", s & a & t),
            ("Local area market", s & a),
        ]
    candidates += [
        (f"{state} property reference", s & t & n),
        (f"{state} property market", s & t),
        (f"{state} market reference", s),
        ("National dataset reference", pd.Series(True, index=data.index)),
    ]
    for label, mask in candidates:
        pool = data[mask]
        if len(pool):
            return {"label": label, "psf": int(round(pool["Median_PSF"].median())),
                    "transactions": int(round(pool["Transactions"].median())),
                    "n": len(pool), "pool": pool}
    raise ValueError("No reference data available")


def header():
    st.markdown('<div class="mh-header"><div class="mh-brand">🏠 Malaysia Housing '
                'Price Estimator<small>2025 market-assisted prediction</small></div>'
                '<span class="mh-chip local">Academic prototype</span></div>',
                unsafe_allow_html=True)


def render_hero():
    """Banner background for the Price Prediction page. Falls back to a plain
    gradient if assets/malaysia_housing_banner.png is not present."""
    encoded = encode_image(str(BANNER_PATH))
    layer = (f"linear-gradient(100deg, rgba(23,32,51,.82) 10%, rgba(23,32,51,.45) 75%), "
             f"url('data:image/png;base64,{encoded}')" if encoded
             else "linear-gradient(120deg,#183153 0%,#2F5C8A 100%)")
    st.markdown(
        f'<div class="mh-hero" style="background-image:{layer};">'
        f'<div class="mh-hero-content"><h2>Estimate a township median price</h2>'
        f'<p>Enter the property and market information below to get an instant estimate.</p>'
        f'</div></div>',
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE 1 - PRICE PREDICTION
# ---------------------------------------------------------------------------
def prediction_page(data, results):
    render_hero()
    st.caption("Values you enter stay exactly as you set them until you press "
              "Predict Price or Reset — changing a dropdown never silently "
              "changes another field.")
    recommended = results.iloc[0]["Model"]

    # Fixed, state-independent defaults (computed once from the whole dataset).
    default_psf = int(round(data["Median_PSF"].median()))
    default_txn = int(round(data["Transactions"].median()))
    psf_min, psf_max = int(data["Median_PSF"].min()), int(data["Median_PSF"].max())

    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            state = st.selectbox("State", sorted(data["State"].unique()), key="pred_state")

            area_options = [NO_AREA] + known_areas_for_state(data, state)
            if SELECTBOX_ACCEPTS_NEW:
                area_choice = st.selectbox(
                    "Area", area_options, key="pred_area", accept_new_options=True,
                    help="Pick a known area, or type one that is not in the "
                         "dropdown. The model is trained on Area, not Township; "
                         "an unrecognised area falls back safely to broader "
                         "market values.")
            else:
                OTHER = "Other (type below)"
                picked = st.selectbox("Area", area_options + [OTHER], key="pred_area_select")
                area_choice = (st.text_input("Type the area", key="pred_area_text")
                              if picked == OTHER else picked)
            area_text = "" if area_choice == NO_AREA else str(area_choice or "")

            ptype = st.selectbox("Property type", sorted(data["Primary_Type"].unique()), key="pred_type")
            tenure = st.selectbox("Tenure", sorted(data["Tenure"].unique()), key="pred_tenure")

        known_area, _ = resolve_known_area(data, state, area_text)
        reference = derive_reference(data, state, known_area, ptype, tenure)

        with right:
            psf = st.number_input(
                "Median price per square foot (RM)", min_value=1, step=10,
                value=default_psf, key="pred_psf",
                help=f"Typical value for this selection: RM {reference['psf']:,} "
                     f"({reference['label']}, n={reference['n']}). This is a "
                     "suggestion only — the field keeps whatever you type.")
            transactions = st.number_input(
                "Transactions", min_value=0, step=1,
                value=default_txn, key="pred_txn",
                help=f"Typical value for this selection: {reference['transactions']:,} "
                     f"({reference['label']}, n={reference['n']}).")

            labels, mapping = [], {}
            for _, row in results.iterrows():
                suffix = " — Recommended" if row["Model"] == recommended else ""
                labels.append(row["Model"] + suffix)
                mapping[row["Model"] + suffix] = row["Model"]
            picked_model = st.selectbox("Model", labels, index=0, key="pred_model")
            model_name = mapping[picked_model]

            st.caption(f"Typical for this selection: RM {reference['psf']:,}/sq ft, "
                      f"{reference['transactions']:,} transactions "
                      f"({reference['label']}, {reference['n']} record(s)).")

        if psf < psf_min or psf > psf_max:
            st.warning(
                f"RM {psf:,} is far outside the observed range for this dataset "
                f"(RM {psf_min:,}–RM {psf_max:,}). Tree-based models such as this "
                f"one do not extrapolate beyond values seen in training — the "
                f"prediction will not keep increasing past a certain point, and "
                f"an extreme input like this should be treated as unreliable.")

        c1, c2 = st.columns([4, 1])
        predict = c1.button("Predict Price", type="primary", use_container_width=True)
        c2.button("Reset", use_container_width=True, on_click=reset_prediction_form, key="reset_prediction")

    if not predict:
        st.markdown('<div class="mh-card mh-empty"><div class="icon">⌂</div>'
                    '<b>Your estimated price will appear here.</b><br>'
                    'Complete the form and select Predict Price.</div>',
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
    st.toast("Prediction completed", icon="✅")

    location = f"{area_text.strip()}, {state}" if area_text.strip() else state
    st.markdown(f'''
    <div class="mh-result">
        <div class="label">Estimated value</div>
        <div class="price">RM {prediction:,.0f}</div>
        <div class="subline"><b>{location}</b> · {ptype} · {tenure}</div>
        <div class="basis">Based on a median of <b>RM {psf:,}/sq ft</b>
            ({reference['label']}, {reference['n']} record(s)),
            {transactions:,} transactions, using the <b>{model_name}</b> model.</div>
        <div class="disclaimer">ⓘ&nbsp;An indicative estimate from 2025 township-level
            data — a market guide, not a formal valuation. Actual price depends on
            property-specific factors not captured here (size, condition, floor,
            view). For a precise figure, consult a registered valuer or real estate
            agent.</div>
    </div>''', unsafe_allow_html=True)

    with st.expander("Technical details"):
        d1, d2, d3 = st.columns(3)
        d1.metric("Model", model_name)
        d2.metric("Typical test MAE", f"RM {metrics['MAE_test']/1000:,.1f}K")
        d3.metric("Test R²", f"{metrics['R2_test']:.3f}")
        st.caption(f"Area status: {'seen during model training' if area_seen else 'not seen during training — the model falls back on State, Property type and Tenure'}.")

    if not known_area and area_text.strip():
        st.info(f"This area was not represented in the dataset. Broader {state} "
               f"market values supplied the typical-PSF suggestion shown above; "
               f"your Predict Price inputs were still used exactly as entered.")


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
    explorer, visual = st.tabs(["Market Explorer", "Visual Insights"])
    with explorer:
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
    with visual:
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
    data = load_data(); results = load_results(); header()
    pred, insights, report = st.tabs(["Price Prediction", "Market Insights", "Model Report"])
    with pred: prediction_page(data, results)
    with insights: insights_page(data)
    with report: model_report_page(results)


main()
