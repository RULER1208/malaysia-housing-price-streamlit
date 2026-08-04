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
"""
from __future__ import annotations
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

from area_preprocessing import clean_area_name, clean_state_name, create_area_key, display_name

st.set_page_config(
    page_title = "Malaysia Housing Price Estimator",
    page_icon = "🏠",
    layout = "wide",
    initial_sidebar_state = "collapsed",
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
.stApp {
    background:var(--bg);
    color:var(--text);
}
.block-container {
    max-width:1160px;
    padding-top:1.4rem;
    padding-bottom:2rem;
}
html,body,[class*="css"] { font-family:Inter,"Segoe UI",Roboto,Arial,sans-serif }
h1,h2,h3,h4 { color:var(--navy) }
.mh-header {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:18px 22px;
    background:white;
    border:1px solid var(--border);
    border-radius:var(--radius);
    box-shadow:var(--shadow);
    margin-bottom:18px;
}
.mh-brand {
    font-size:1.22rem;
    font-weight:750;
    color:var(--navy);
}
.mh-brand small {
    display:block;
    font-size:.82rem;
    color:var(--muted);
    font-weight:500;
    margin-top:2px;
}
.mh-card {
    background:white;
    border:1px solid var(--border);
    border-radius:var(--radius);
    padding:22px;
    box-shadow:var(--shadow);
}
.mh-result {
    background:linear-gradient(135deg,#EAF7F1,#EEF4FF);
    border:1px solid #BFE5D2;
    border-radius:18px;
    padding:24px;
    animation:fadeUp .24s ease both;
}
.mh-result .label {
    font-size:.82rem;
    color:var(--muted);
    text-transform:uppercase;
    letter-spacing:.08em;
}
.mh-result .price {
    font-size:2.5rem;
    font-weight:760;
    color:var(--navy);
    margin:4px 0 8px;
}
.mh-grid {
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:12px;
    margin-top:16px;
}
.mh-item {
    font-size:.92rem;
    color:var(--text);
}
.mh-item span {
    display:block;
    font-size:.74rem;
    color:var(--muted);
    text-transform:uppercase;
}
.mh-chip {
    display:inline-flex;
    align-items:center;
    padding:5px 9px;
    border-radius:999px;
    font-size:.82rem;
    font-weight:650;
}
.mh-chip.local {
    background:#EAF7F1;
    color:#116B48;
}
.mh-chip.fallback {
    background:#FFF6E6;
    color:#8A5300;
}
.mh-empty {
    text-align:center;
    padding:36px 18px;
    color:var(--muted);
}
.mh-empty .icon {
    font-size:2rem;
    margin-bottom:8px;
}
.stButton>button {
    min-height:44px;
    border-radius:11px;
    font-weight:650;
    transition:transform .18s ease,box-shadow .18s ease;
}
.stButton>button:hover { transform:translateY(-1px) }
.stButton>button[kind="primary"] {
    background:var(--blue);
    border-color:var(--blue);
}
.stTabs [data-baseweb="tab-list"] {
    gap:8px;
    border-bottom:1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    height:46px;
    color:var(--muted);
    font-weight:650;
}
.stTabs [aria-selected="true"] {
    color:var(--blue)!important;
    border-bottom:2px solid var(--blue);
}
div[data-testid="stMetric"] {
    background:white;
    border:1px solid var(--border);
    border-radius:13px;
    padding:13px 15px;
}
@keyframes fadeUp{from {
    opacity:0;
    transform:translateY(8px);
}
to {
    opacity:1;
    transform:translateY(0);
}
}
@media(max-width:700px){.block-container {
    padding-left:12px;
    padding-right:12px;
}
.mh-header { padding:15px }
.mh-result .price { font-size:2rem }
}
@media(prefers-reduced-motion:reduce){*,*::before,*::after {
    animation-duration:.01ms!important;
    transition-duration:.01ms!important;
}
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data(show_spinner=False)
def load_results():
    results = pd.read_csv(RESULTS_PATH)
    return results.sort_values(
        ["Group_CV_RMSE_mean", "Group_CV_RMSE_std"]
    ).reset_index(drop=True)

@st.cache_resource(show_spinner="Loading prediction model...")
def load_model(name):
    filename = name.split(" (")[0].lower().replace(" ","_")+".pkl"
    return joblib.load(MODELS_DIR/filename)

def model_filename(name):
    return name.split(" (")[0].lower().replace(" ","_")+".pkl"

# ---------------------------------------------------------------------------
# AREA RESOLUTION AND REFERENCE LOOKUP
# ---------------------------------------------------------------------------
def reset_prediction_form():
    """Clear every prediction widget and remove the previous result."""
    for key in list(st.session_state.keys()):
        if key.startswith("pred_") or key.startswith("psf_") or key.startswith("txn_"):
            del st.session_state[key]

def resolve_known_area(data, state, area_text):
    clean = clean_area_name(area_text)
    subset = data[data["State"]==state]
    match = subset[subset["Area_Clean"]==clean]
    return (clean if len(match) else None), match

def derive_reference(data, state, area_clean, ptype, tenure):
    s = data["State"].eq(state); t = data["Primary_Type"].eq(ptype); n = data["Tenure"].eq(tenure)
    candidates = []
    if area_clean:
        a = data["Area_Clean"].eq(area_clean)
        candidates += [
            ("Local Area reference", s & a & t & n),
            ("Local Area + property type", s & a & t),
            ("Local Area market", s & a),
        ]
    candidates += [(f"{state} property reference",s&t&n),(f"{state} property market",s&t),(f"{state} market reference",s),("National dataset reference",pd.Series(True,index=data.index))]
    for label, mask in candidates:
        pool = data[mask]
        if len(pool):
            return {"label":label,"psf":int(round(pool["Median_PSF"].median())),"transactions":int(round(pool["Transactions"].median())),"n":len(pool),"pool":pool}
    raise ValueError("No reference data available")

def header():
    st.markdown('<div class="mh-header"><div class="mh-brand">🏠 Malaysia Housing Price Estimator<small>2025 market-assisted prediction</small></div><span class="mh-chip local">Academic prototype</span></div>',unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PAGE 1 - PRICE PREDICTION
# ---------------------------------------------------------------------------
def prediction_page(data, results):
    st.subheader("Housing Price Prediction")
    st.caption("Enter the property and market information to estimate a township-level median price.")
    recommended = results.iloc[0]["Model"]
    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            state = st.selectbox("State",sorted(data["State"].unique()),key = "pred_state")
            area_text = st.text_input("Area",placeholder = "Example: Skudai or Taman Molek",help = "The model is trained on Area, not Township. Any text you enter is cleaned into a state-qualified Area_Key; unrecognised Areas fall back safely to broader market values.",key = "pred_area")
            ptype = st.selectbox("Property type",sorted(data["Primary_Type"].unique()),key = "pred_type")
            tenure = st.selectbox("Tenure",sorted(data["Tenure"].unique()),key = "pred_tenure")
        known_area, _ = resolve_known_area(data, state, area_text)
        reference = derive_reference(data, state, known_area, ptype, tenure)
        with right:
            ref_key = f"{state}|{clean_area_name(area_text)}|{ptype}|{tenure}"
            psf = st.number_input("Median price per square foot (RM)",min_value = 1,value = int(reference["psf"]),step = 10,key = f"psf_{ref_key}")
            transactions = st.number_input("Transactions",min_value = 0,value = int(reference["transactions"]),step = 1,key = f"txn_{ref_key}")
            labels = []; mapping = {}
            for _, row in results.iterrows():
                suffix = " — Recommended" if row["Model"]==recommended else ""
                labels.append(row["Model"]+suffix);mapping[row["Model"]+suffix]=row["Model"]
            picked = st.selectbox("Model",labels,index = 0,key = "pred_model")
            model_name = mapping[picked]
            chip_class = "local" if known_area else "fallback"
            st.markdown(f'<span class="mh-chip {chip_class}">{reference["label"]}</span>',unsafe_allow_html=True)
        c1, c2 = st.columns([4, 1])
        predict = c1.button("Predict Price",type = "primary",use_container_width = True)
        c2.button("Reset",use_container_width = True,on_click = reset_prediction_form,key = "reset_prediction")
    if not predict:
        st.markdown('<div class="mh-card mh-empty"><div class="icon">⌂</div><b>Your estimated price will appear here.</b><br>Complete the form and select Predict Price.</div>',unsafe_allow_html=True)
        return
    model = load_model(model_name)
    area_key = create_area_key(state, area_text)
    features = pd.DataFrame([{"State":state,"Area_Key":area_key,"Tenure":tenure,"Primary_Type":ptype,"Median_PSF":psf,"Transactions":transactions}])[MODEL_FEATURES]
    with st.spinner("Calculating estimate..."):
        prediction = float(model.predict(features)[0])
    metrics = results[results["Model"]==model_name].iloc[0]
    try:
        encoder = model.named_steps["preprocess"].named_transformers_["cat"].named_steps["encoder"]
        area_position = ["State","Area_Key","Tenure","Primary_Type"].index("Area_Key")
        area_seen = area_key in set(encoder.categories_[area_position])
    except Exception:
        area_seen = bool(known_area)
    st.toast("Prediction completed", icon="✅")
    location = f"{area_text.strip()}, {state}" if area_text.strip() else state
    st.markdown(f'''<div class="mh-result"><div class="label">Estimated township median price</div><div class="price">RM {prediction:,.0f}</div><b>{location}</b> · {ptype} · {tenure}<div class="mh-grid"><div class="mh-item"><span>Median PSF used</span>RM {psf:,}</div><div class="mh-item"><span>Transactions</span>{transactions:,}</div><div class="mh-item"><span>Model</span>{model_name}</div><div class="mh-item"><span>Typical test MAE</span>RM {metrics['MAE_test']/1000:,.1f}K</div><div class="mh-item"><span>Test R²</span>{metrics['R2_test']:.3f}</div><div class="mh-item"><span>Area status</span>{'Seen during model training' if area_seen else 'Unseen Area'}</div></div><p style="color:#667085;margin:16px 0 0">Township-level market estimate based on the 2025 dataset. Not a formal property valuation.</p></div>''',unsafe_allow_html=True)
    if not known_area and area_text.strip():
        st.info(f"This Area was not represented in the dataset reference. Broader {state} market values supplied the suggested PSF and transactions.")

FIGURE_GROUPS = {
"Data quality":["fig01_raw_target_distribution.png","fig02_raw_numeric_boxplots.png","fig10_outlier_before_after.png","fig11_price_distribution_before_after.png"],
"Area quality and coverage":["fig09_area_labels_before_after.png","fig12_area_repeated_singleton.png","fig13_area_frequency_distribution.png","fig14_top20_areas.png","fig15_area_cleaning_findings.png","fig16_area_price_distribution.png"],
"Location and property":["fig18_state_counts_clean.png","fig19_state_price_distribution.png","fig20_property_type_price.png","fig21_tenure_price.png"],
"Relationships":["fig22_psf_price_by_category.png","fig23_feature_correlation.png","fig24_top_transactions.png"],
}

# ---------------------------------------------------------------------------
# PAGE 2 - MARKET INSIGHTS
# ---------------------------------------------------------------------------
def insights_page(data):
    explorer,visual = st.tabs(["Market Explorer","Visual Insights"])
    with explorer:
        st.markdown("#### Historical 2025 dataset exploration")
        a, b, c, d = st.columns(4)
        state = a.selectbox("State",["All"]+sorted(data["State"].unique()),key = "ex_state")
        area = b.selectbox("Area",["All"]+sorted(data["Area_Clean"].unique()),key = "ex_area")
        ptype = c.selectbox("Property type",["All"]+sorted(data["Primary_Type"].unique()),key = "ex_type")
        tenure = d.selectbox("Tenure",["All"]+sorted(data["Tenure"].unique()),key = "ex_tenure")
        subset = data.copy()
        if state!="All":subset=subset[subset["State"]==state]
        if area!="All":subset=subset[subset["Area_Clean"]==area]
        if ptype!="All":subset=subset[subset["Primary_Type"]==ptype]
        if tenure!="All":subset=subset[subset["Tenure"]==tenure]
        if len(subset)==0:
            st.warning("No historical records match these filters.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Records",f"{len(subset):,}")
            m2.metric("Median price",f"RM {subset['Median_Price'].median()/1000:,.0f}K")
            m3.metric("Median PSF",f"RM {subset['Median_PSF'].median():,.0f}")
            m4.metric("Median transactions",f"{subset['Transactions'].median():,.0f}")
            with st.expander("View matching historical records"):
                show = subset[["Township","Area_Clean","State","Primary_Type","Tenure","Median_Price","Median_PSF","Transactions"]].copy()
                st.dataframe(show, use_container_width=True, hide_index=True)
    with visual:
        group = st.selectbox("Insight category",list(FIGURE_GROUPS))
        for filename in FIGURE_GROUPS[group]:
            path = FIGURES_DIR/filename
            if path.exists():
                st.image(str(path), use_container_width=True)

# ---------------------------------------------------------------------------
# PAGE 3 - MODEL REPORT
# ---------------------------------------------------------------------------
def model_report_page(results):
    recommended = results.iloc[0]
    st.subheader("Model Report")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected model",recommended["Model"])
    m2.metric("Group CV RMSE",f"RM {recommended['Group_CV_RMSE_mean']/1000:,.1f}K")
    m3.metric("Test MAE",f"RM {recommended['MAE_test']/1000:,.1f}K")
    m4.metric("Test R²",f"{recommended['R2_test']:.3f}")
    st.dataframe(results, use_container_width=True, hide_index=True)
    sections = {"Area ablation":["fig25_area_ablation.png"],"Performance":["fig26_model_test_comparison.png","fig27_cv_stability_overfitting.png"],"Diagnostics":["fig28_prediction_diagnostics.png"],"Importance":["fig29_permutation_importance.png","fig30_aggregated_split_importance.png"]}
    section = st.selectbox("Report section",list(sections))
    for filename in sections[section]:
        path = FIGURES_DIR/filename
        if path.exists():st.image(str(path), use_container_width=True)
    with st.expander("Key limitations"):
        st.markdown("- Some Areas contain very few records.\n- Completely unseen Areas are harder than previously observed Areas.\n- Median PSF remains required market information.\n- The dataset is a static 2025 snapshot.\n- Results are township-level medians, not individual-property valuations.")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    missing = [p.name for p in [DATA_PATH, RESULTS_PATH] if not p.exists()]
    if missing:
        st.error("Missing required files: "+", ".join(missing));st.stop()
    data = load_data();results = load_results();header()
    pred,insights,report = st.tabs(["Price Prediction","Market Insights","Model Report"])
    with pred: prediction_page(data, results)
    with insights: insights_page(data)
    with report: model_report_page(results)

main()
