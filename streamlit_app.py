"""
BMDS2003 Data Science - Deployment Prototype
Malaysia Housing Median Price Estimator

Run locally:  streamlit run streamlit_app.py
"""
from __future__ import annotations
from pathlib import Path
import inspect
import joblib
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# Used to analyze the address and zoom map to specific coordinates
try:
    from geopy.geocoders import Nominatim
    HAS_GEOPY = True
except ImportError:
    HAS_GEOPY = False

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

SELECTBOX_ACCEPTS_NEW = "accept_new_options" in inspect.signature(st.selectbox).parameters
NO_AREA = "— No specific area —"

# Accurate regional coordinates for all Malaysian States & Federal Territories
STATE_COORDS = {
    "Johor": [1.9344, 103.3587], "Kedah": [6.1184, 100.3685],
    "Kelantan": [5.3500, 102.0000], "Melaka": [2.2500, 102.2500],
    "Negeri Sembilan": [2.7258, 101.9424], "Pahang": [3.8126, 102.8000],
    "Penang": [5.4141, 100.3288], "Perak": [4.5921, 101.0901],
    "Perlis": [6.4449, 100.2048], "Sabah": [5.4204, 116.7968],
    "Sarawak": [2.5574, 113.0012], "Selangor": [3.0738, 101.5183],
    "Terengganu": [4.7500, 103.0000], "Kuala Lumpur": [3.1390, 101.6869],
    "Putrajaya": [2.9264, 101.6964], "Labuan": [5.2831, 115.2308]
}

st.markdown(r"""
<style>
:root {
    --bg:#F6F8FB; --card:#FFFFFF; --navy:#15243A; --blue:#2F6FED;
    --text:#172033; --muted:#667085; --border:#E2E7EF; --green:#18875D;
    --soft:#EEF4FF; --radius:14px; --shadow:0 6px 22px rgba(24,49,83,.06);
    --mono:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace;
}
.stApp { background:var(--bg); color:var(--text); }
html,body,[class*="css"] { font-family:Inter,"Segoe UI",Roboto,Arial,sans-serif }
h1,h2,h3,h4 { color:var(--navy) }

.block-container { max-width:1180px; padding-top:12px!important; padding-bottom:2rem; }
header[data-testid="stHeader"] { background:transparent!important; }
[data-testid="stToolbar"] { color:var(--muted)!important; }

.stTabs [role="tablist"] {
    display:flex!important; align-items:center!important; gap:8px!important;
    min-height:62px; padding:0 210px 0 22px!important;
    background:var(--navy)!important; border-radius:var(--radius)!important;
    border-bottom:none!important; margin-bottom:22px!important; overflow:visible!important;
}
.stTabs [role="tablist"]::before {
    content:"\2302\00a0\00a0Malaysia Housing Price Estimator";
    font-size:1.04rem; font-weight:700; color:#FFFFFF; margin-right:24px;
}
.stTabs [role="tablist"]::after {
    margin-left:auto; font-family:var(--mono); font-size:.72rem; letter-spacing:.1em; color:#8FA6C6;
}
.stTabs [role="tab"] {
    height:38px!important; padding:0 18px!important; border-radius:999px!important;
    color:#B9C8DF!important; font-weight:600; font-size:.92rem;
    background:transparent!important; border:none!important;
}
.stTabs [role="tab"]:hover { color:#FFFFFF!important; background:rgba(255,255,255,.10)!important; }
.stTabs [role="tab"][aria-selected="true"] { color:var(--navy)!important; background:#FFFFFF!important; border-bottom:none!important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none!important; background:transparent!important; }

.mh-label {
    display:flex; align-items:baseline; justify-content:space-between;
    font-family:var(--mono); font-size:.72rem; letter-spacing:.1em;
    color:var(--muted); margin:2px 0 5px 0; text-transform:uppercase;
}
.mh-rule { border:none; border-top:1px solid var(--border); margin:16px 0 14px 0; }
.mh-result { background:var(--navy); border-radius:var(--radius); padding:26px 26px 24px; box-shadow:var(--shadow); }
.mh-result .cap { font-family:var(--mono); font-size:.72rem; letter-spacing:.14em; color:#9FB4D4; text-transform:uppercase; }
.mh-result .price { font-size:2.7rem; font-weight:750; color:#FFFFFF; margin:8px 0 6px; line-height:1.05; letter-spacing:-.01em; }
.mh-result .sub { color:#C9D7EC; font-size:.95rem; }
.mh-result .rule { border-top:1px solid rgba(255,255,255,.16); margin:18px 0 14px; }
.mh-stats { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }
.mh-stats .k { font-family:var(--mono); font-size:.68rem; letter-spacing:.12em; color:#9FB4D4; text-transform:uppercase; margin-bottom:3px; }
.mh-stats .v { font-family:var(--mono); font-size:.94rem; font-weight:700; color:#FFFFFF; }

.mh-used { background:var(--card); border:1px solid var(--border); border-radius:var(--radius); padding:16px 18px; box-shadow:var(--shadow); margin-top:12px; }
.mh-used h4 { margin:0 0 10px 0; font-size:.94rem; font-weight:700; color:var(--navy); }
.mh-used .row { display:flex; justify-content:space-between; align-items:baseline; padding:6px 0; font-size:.9rem; }
.mh-used .row + .row { border-top:1px dashed var(--border); }
.mh-used .k { color:var(--muted); }
.mh-used .v { font-weight:650; color:var(--text); font-family:var(--mono); }

.mh-empty { background:var(--card); border:1px dashed #CFD8E6; border-radius:var(--radius); padding:52px 24px; text-align:center; color:var(--muted); box-shadow:var(--shadow); }
.mh-empty .icon { font-size:2rem; margin-bottom:10px; opacity:.7; }
.mh-empty b { color:var(--navy); display:block; margin-bottom:4px; font-size:1rem; }

.stButton>button { min-height:44px; border-radius:11px; font-weight:650; }
.stButton>button[kind="primary"] { background:var(--blue); border-color:var(--blue); }
.stButton>button[kind="secondary"] { background:#FFFFFF; color:var(--text); border:1px solid var(--border); }
div[data-testid="stMetric"] { background:white; border:1px solid var(--border); border-radius:13px; padding:13px 15px; }

@media(max-width:820px){
    .stTabs [role="tablist"] { min-height:56px; padding:0 12px!important; }
    .stTabs [role="tablist"]::before { font-size:.9rem; margin-right:12px; }
    .mh-result .price{font-size:2.05rem;}
    .mh-stats{grid-template-columns:1fr 1fr;}
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
    right = f'<span class="hint">{hint}</span>' if hint else ""
    st.markdown(f'<div class="mh-label"><span>{text}</span>{right}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ADDRESS PARSER & MAP SYNC
# ---------------------------------------------------------------------------
def reset_prediction_form():
    for key in list(st.session_state.keys()):
        if key.startswith("pred_") or key in ["selected_state", "selected_area", "map_center", "map_zoom", "address_input"]:
            del st.session_state[key]

def known_areas_for_state(data: pd.DataFrame, state: str) -> list[str]:
    subset = data.loc[data["State"] == state, "Area_Clean"].dropna().unique()
    return sorted({display_name(a) for a in subset})

def resolve_known_area(data, state, area_text):
    clean = clean_area_name(area_text)
    subset = data[data["State"] == state]
    match = subset[subset["Area_Clean"] == clean]
    return (clean if len(match) else None), match

def analyze_address(data, available_states):
    """Fired when the user types an address. Analyzes text for State & Area, then geocodes for map zooming."""
    addr = st.session_state.get("address_input", "").lower()
    if not addr: 
        return

    # 1. String Match State
    matched_state = None
    for st_name in sorted(available_states, key=len, reverse=True): # Sort by length to catch full names
        if st_name.lower() in addr:
            matched_state = st_name
            st.session_state["selected_state"] = st_name
            break
    
    # 2. String Match Area
    if matched_state:
        areas = known_areas_for_state(data, matched_state)
        for a in sorted(areas, key=len, reverse=True):
            if a.lower() in addr:
                st.session_state["selected_area"] = a
                break

    # 3. Use Geopy to fetch exact Map Coordinates for the specific area
    if HAS_GEOPY:
        try:
            geolocator = Nominatim(user_agent="my_housing_app", timeout=3)
            location = geolocator.geocode(f"{addr}, Malaysia")
            if location:
                st.session_state["map_center"] = [location.latitude, location.longitude]
                st.session_state["map_zoom"] = 13 # Zoom tightly into specific area
                return
        except Exception:
            pass # Failsafe: Continue to fallback if API fails
            
    # 4. Fallback if Geopy is offline or address unresolvable: Center on State
    if matched_state and matched_state in STATE_COORDS:
        st.session_state["map_center"] = STATE_COORDS[matched_state]
        st.session_state["map_zoom"] = 8

def derive_reference(data, state, area_clean, ptype, tenure):
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
# PAGE 1 - PRICE PREDICTION (REDESIGNED LAYOUT)
# ---------------------------------------------------------------------------
def prediction_page(data, results):
    recommended = results.iloc[0]["Model"]
    default_psf = int(round(data["Median_PSF"].median()))
    default_txn = int(round(data["Transactions"].median()))
    psf_min, psf_max = int(data["Median_PSF"].min()), int(data["Median_PSF"].max())
    available_states = sorted(data["State"].unique())

    # Session State Initialization
    if "selected_state" not in st.session_state:
        st.session_state["selected_state"] = "Selangor"
    if "selected_area" not in st.session_state:
        st.session_state["selected_area"] = NO_AREA
        
    current_state = st.session_state["selected_state"]

    with st.container(border=True):
        st.markdown("<h4 style='margin-top:0;'>📍 Location Selection</h4>", unsafe_allow_html=True)
        
        # 1. Address Input Analyzer
        st.text_input("Enter address to auto-detect State & Area (e.g., Jalan Mewah, Kulai, Johor) or click the map:", 
                      key="address_input", on_change=analyze_address, args=(data, available_states))

        # 2. Interactive Map (Reacts to both Address Input & Mouse Clicks)
        map_center = st.session_state.get("map_center", STATE_COORDS.get(current_state, [4.2105, 108.9758]))
        map_zoom = st.session_state.get("map_zoom", 7)
        
        m = folium.Map(location=map_center, zoom_start=map_zoom, tiles="OpenStreetMap")
        for st_name in available_states:
            if st_name in STATE_COORDS:
                is_selected = (st_name == current_state)
                folium.CircleMarker(
                    location=STATE_COORDS[st_name],
                    radius=11 if is_selected else 8,
                    color="#18875D" if is_selected else "#15243A",
                    weight=3 if is_selected else 2,
                    fill=True,
                    fill_color="#10B981" if is_selected else "#2F6FED",
                    fill_opacity=0.9 if is_selected else 0.75,
                    tooltip=folium.Tooltip(f"<b>{st_name}</b> (Click to select)", sticky=True),
                    popup=st_name
                ).add_to(m)

        # Draw map 
        map_data = st_folium(m, height=350, use_container_width=True, key="malaysia_map")
        
        # Override if user manually clicks a state marker on the map
        if map_data and map_data.get("last_object_clicked_popup"):
            clicked_st = map_data["last_object_clicked_popup"]
            if clicked_st in available_states and clicked_st != st.session_state["selected_state"]:
                st.session_state["selected_state"] = clicked_st
                st.session_state["selected_area"] = NO_AREA
                st.session_state["map_center"] = STATE_COORDS[clicked_st]
                st.session_state["map_zoom"] = 8
                st.rerun()

        st.markdown('<hr class="mh-rule">', unsafe_allow_html=True)
        st.markdown("<h4 style='margin-top:0;'>🏡 Property Details</h4>", unsafe_allow_html=True)
        
        # 3. Inputs Section (3 Columns to compress vertical height)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"<div style='font-size:.9rem; font-weight:600; color:var(--navy); margin-bottom:5px;'>State</div><div style='font-size:1.05rem; margin-bottom:15px;'>{current_state}</div>", unsafe_allow_html=True)
            
            field_label("Area")
            area_options = [NO_AREA] + known_areas_for_state(data, current_state)
            
            # Sync Dropdown with parsed address memory
            area_idx = 0
            if st.session_state["selected_area"] in area_options:
                area_idx = area_options.index(st.session_state["selected_area"])
                
            if SELECTBOX_ACCEPTS_NEW:
                picked_area = st.selectbox("Area", area_options, index=area_idx, key="pred_area", label_visibility="collapsed", accept_new_options=True)
                area_text = "" if picked_area == NO_AREA else str(picked_area or "")
            else:
                OTHER = "Other (type below)"
                picked_area = st.selectbox("Area", area_options + [OTHER], index=area_idx, key="pred_area_select", label_visibility="collapsed")
                area_text = st.text_input("Type the area", key="pred_area_text") if picked_area == OTHER else ( "" if picked_area == NO_AREA else str(picked_area))

        with col2:
            field_label("Property type")
            ptype = st.selectbox("Property type", sorted(data["Primary_Type"].unique()), key="pred_type", label_visibility="collapsed")
            
            field_label("Tenure")
            tenure = st.selectbox("Tenure", sorted(data["Tenure"].unique()), key="pred_tenure", label_visibility="collapsed")

        with col3:
            field_label("Median price per sq ft (RM)")
            psf = st.number_input("Median price per square foot (RM)", min_value=1, step=10, value=default_psf, key="pred_psf", label_visibility="collapsed")

            field_label("Transactions")
            transactions = st.number_input("Transactions", min_value=0, step=1, value=default_txn, key="pred_txn", label_visibility="collapsed")

        field_label("Model Override")
        labels, mapping = [], {}
        for _, row in results.iterrows():
            suffix = " — Recommended" if row["Model"] == recommended else ""
            labels.append(row["Model"] + suffix)
            mapping[row["Model"] + suffix] = row["Model"]
        picked_model = st.selectbox("Model", labels, index=0, key="pred_model", label_visibility="collapsed")
        model_name = mapping[picked_model]

        if psf < psf_min or psf > psf_max:
            st.warning(f"RM {psf:,} is outside the observed range (RM {psf_min:,}–RM {psf_max:,}). Estimate reliability drops past this point.")

        # Action Buttons
        b1, b2 = st.columns([4, 1])
        predict = b1.button("Predict Price  →", type="primary", use_container_width=True)
        b2.button("Reset Form", type="secondary", use_container_width=True, on_click=reset_prediction_form, key="reset_prediction")

    # ---------------- BOTTOM: Result Box ----------------
    if predict:
        model = load_model(model_name)
        area_key = create_area_key(current_state, area_text)
        features = pd.DataFrame([{
            "State": current_state, "Area_Key": area_key, "Tenure": tenure,
            "Primary_Type": ptype, "Median_PSF": psf, "Transactions": transactions
        }])[MODEL_FEATURES]
        
        with st.spinner("Calculating estimate..."):
            prediction = float(model.predict(features)[0])
        
        metrics = results[results["Model"] == model_name].iloc[0]
        known_area, _ = resolve_known_area(data, current_state, area_text)
        reference = derive_reference(data, current_state, known_area, ptype, tenure)
        
        try:
            encoder = model.named_steps["preprocess"].named_transformers_["cat"].named_steps["encoder"]
            area_position = ["State", "Area_Key", "Tenure", "Primary_Type"].index("Area_Key")
            area_seen = area_key in set(encoder.categories_[area_position])
        except Exception:
            area_seen = bool(known_area)

        location = f"{area_text.strip()}, {current_state}" if area_text.strip() else current_state
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

        with st.expander("Technical details"):
            st.markdown(
                f"- **Reference source:** {reference['label']}\n"
                f"- **Reference records:** {reference['n']:,}\n"
                f"- **Area recognised:** {'Yes' if known_area else 'No'}\n"
                f"- **Area seen during model training:** {'Yes' if area_seen else 'No'}\n"
                f"- **Broader state reference values used:** {'No' if known_area else 'Yes'}"
            )
    else:
        st.markdown(
            '<div class="mh-empty"><div class="icon">⌂</div>'
            '<b>Your estimate will appear here.</b>'
            'Complete the inputs above, then select Predict Price.</div>',
            unsafe_allow_html=True
        )

FIGURE_GROUPS = {
    "Data quality": [
        ("fig01_raw_target_distribution.png", "House prices are heavily skewed."),
        ("fig02_raw_numeric_boxplots.png", "Boxplots of price, PSF and transactions before cleaning."),
        ("fig10_outlier_before_after.png", "Extreme values removed by outlier cleaning."),
        ("fig11_price_distribution_before_after.png", "Price distribution becomes more balanced."),
    ],
    "Area quality and coverage": [
        ("fig09_area_labels_before_after.png", "Area name spelling and formatting before vs after standardisation."),
        ("fig12_area_repeated_singleton.png", "Many areas appear only once in the data."),
        ("fig13_area_frequency_distribution.png", "How many records each area has."),
        ("fig14_top20_areas.png", "The 20 areas with the most records in the dataset."),
        ("fig15_area_cleaning_findings.png", "Summary of issues found and fixed."),
        ("fig16_area_price_distribution.png", "How median price varies across different areas."),
    ],
    "Location and property": [
        ("fig18_state_counts_clean.png", "Number of records per state after cleaning."),
        ("fig19_state_price_distribution.png", "Median price differs a lot from state to state."),
        ("fig20_property_type_price.png", "Bungalows and semi-detached homes cost more."),
        ("fig21_tenure_price.png", "Freehold properties tend to have a different price profile."),
    ],
    "Relationships": [
        ("fig22_psf_price_by_category.png", "Price per square foot is one of the strongest single predictors of price."),
        ("fig23_feature_correlation.png", "How strongly each feature relates to price."),
        ("fig24_top_transactions.png", "The most actively traded townships in 2025."),
    ],
}

# ---------------------------------------------------------------------------
# PAGE 2 & 3
# ---------------------------------------------------------------------------
def insights_page(data):
    view = st.radio("View", ["Market Explorer", "Visual Insights"], horizontal=True, label_visibility="collapsed", key="insights_view")
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
                show = subset[["Township", "Area_Clean", "State", "Primary_Type", "Tenure", "Median_Price", "Median_PSF", "Transactions"]].copy()
                st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        group = st.selectbox("Insight category", list(FIGURE_GROUPS))
        for filename, caption in FIGURE_GROUPS[group]:
            path = FIGURES_DIR / filename
            if path.exists():
                st.image(str(path), use_container_width=True)
                st.markdown(f'<p class="mh-fig-caption">{caption}</p>', unsafe_allow_html=True)

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
        "Area ablation": [("fig25_area_ablation.png", "Does including Area actually improve predictions?")],
        "Performance": [("fig26_model_test_comparison.png", "How the models compare on unseen data."),
                        ("fig27_cv_stability_overfitting.png", "Checking that each model performs consistently.")],
        "Diagnostics": [("fig28_prediction_diagnostics.png", "Where the model's predictions are most and least accurate.")],
        "Importance": [("fig29_permutation_importance.png", "Which inputs the model actually relies on."),
                       ("fig30_aggregated_split_importance.png", "Which inputs the model used most often while learning.")],
    }
    section = st.selectbox("Report section", list(sections))
    for filename, caption in sections[section]:
        path = FIGURES_DIR / filename
        if path.exists():
            st.image(str(path), use_container_width=True)
            st.markdown(f'<p class="mh-fig-caption">{caption}</p>', unsafe_allow_html=True)
    with st.expander("Key limitations"):
        st.markdown(
            "- Some Areas contain very few records.\n"
            "- Completely unseen Areas are harder than previously observed Areas.\n"
            "- Median PSF remains required market information.\n"
            "- The dataset is a static 2025 snapshot."
        )

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    missing = [p.name for p in [DATA_PATH, RESULTS_PATH] if not p.exists()]
    if missing:
        st.error("Missing required files: " + ", ".join(missing)); st.stop()
    data = load_data(); results = load_results()
    pred, insights, report = st.tabs(["Price Prediction", "Market Insights", "Model Report"])
    with pred: prediction_page(data, results)
    with insights: insights_page(data)
    with report: model_report_page(results)

main()
