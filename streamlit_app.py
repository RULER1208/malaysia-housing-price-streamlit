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
# CONFIGURATION & COORDINATES
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "malaysia_house_price_cleaned_with_area.csv"
RESULTS_PATH = APP_DIR / "model_results.csv"
MODELS_DIR = APP_DIR / "models"
FIGURES_DIR = APP_DIR / "figures"
MODEL_FEATURES = ["State", "Area_Key", "Tenure", "Primary_Type", "Median_PSF", "Transactions"]

NO_AREA = "— No Area Selected —"

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
.stTabs [role="tab"] {
    height:38px!important; padding:0 18px!important; border-radius:999px!important;
    color:#B9C8DF!important; font-weight:600; font-size:.92rem;
    background:transparent!important; border:none!important;
}
.stTabs [role="tab"]:hover { color:#FFFFFF!important; background:rgba(255,255,255,.10)!important; }
.stTabs [role="tab"][aria-selected="true"] { color:var(--navy)!important; background:#FFFFFF!important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none!important; }

.mh-label { display:flex; font-family:var(--mono); font-size:.72rem; letter-spacing:.1em; color:var(--muted); margin:2px 0 5px 0; text-transform:uppercase; }
.mh-rule { border:none; border-top:1px solid var(--border); margin:16px 0 14px 0; }
.mh-result { background:var(--navy); border-radius:var(--radius); padding:26px 26px 24px; box-shadow:var(--shadow); margin-top:20px; }
.mh-result .cap { font-family:var(--mono); font-size:.72rem; letter-spacing:.14em; color:#9FB4D4; text-transform:uppercase; }
.mh-result .price { font-size:2.7rem; font-weight:750; color:#FFFFFF; margin:8px 0 6px; }
.mh-result .sub { color:#C9D7EC; font-size:.95rem; }
.mh-result .rule { border-top:1px solid rgba(255,255,255,.16); margin:18px 0 14px; }
.mh-stats { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }
.mh-stats .k { font-family:var(--mono); font-size:.68rem; letter-spacing:.12em; color:#9FB4D4; text-transform:uppercase; margin-bottom:3px; }
.mh-stats .v { font-family:var(--mono); font-size:.94rem; font-weight:700; color:#FFFFFF; }

.mh-empty { background:var(--card); border:1px dashed #CFD8E6; border-radius:var(--radius); padding:52px 24px; text-align:center; color:var(--muted); box-shadow:var(--shadow); margin-top:20px;}
.stButton>button[kind="primary"] { background:var(--blue); border-color:var(--blue); border-radius:11px; min-height:44px;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DATA & HELPERS
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data(show_spinner=False)
def load_results():
    results = pd.read_csv(RESULTS_PATH)
    return results.sort_values(["Group_CV_RMSE_mean", "Group_CV_RMSE_std"]).reset_index(drop=True)

@st.cache_resource(show_spinner=False)
def load_model(name):
    filename = name.split(" (")[0].lower().replace(" ", "_") + ".pkl"
    return joblib.load(MODELS_DIR / filename)

@st.cache_data(show_spinner=False)
def get_area_coords(area_name: str, state_name: str):
    """Geocode an area specifically within a state to place map markers."""
    if not HAS_GEOPY: return None
    try:
        geolocator = Nominatim(user_agent="mh_estimator", timeout=3)
        loc = geolocator.geocode(f"{area_name}, {state_name}, Malaysia")
        if loc: return [loc.latitude, loc.longitude]
    except:
        pass
    return None

def field_label(text: str) -> None:
    st.markdown(f'<div class="mh-label">{text}</div>', unsafe_allow_html=True)

def ptype_svg_card(ptype: str, is_selected: bool) -> str:
    """Returns an HTML string containing an SVG icon for the property type."""
    color = "#2F6FED" if is_selected else "#667085"
    bg = "#EEF4FF" if is_selected else "#F6F8FB"
    border = "2px solid #2F6FED" if is_selected else "1px solid #E2E7EF"
    
    # Highrise SVG
    if ptype in ["Condominium", "Apartment", "Flat", "Service Residence"]:
        svg = f'''<svg width="34" height="34" viewBox="0 0 24 24" fill="{color}"><path d="M7 19h10V4H7v15zm2-13h2v2H9V6zm4 0h2v2h-2V6zm-4 3h2v2H9V9zm4 0h2v2h-2V9zm-4 3h2v2H9v-2zm4 0h2v2h-2v-2zm-4 3h2v2H9v-2zm4 0h2v2h-2v-2z"/></svg>'''
    # Landed SVG
    else:
        svg = f'''<svg width="34" height="34" viewBox="0 0 24 24" fill="{color}"><path d="M12 3L2 12h3v8h6v-6h2v6h6v-8h3L12 3zm-1 9H9V9h2v3zm4 0h-2V9h2v3z"/></svg>'''
        
    return f"""
    <div style="border:{border}; background:{bg}; border-radius:12px; padding:12px 6px; text-align:center; height:85px; display:flex; flex-direction:column; justify-content:center; align-items:center;">
        {svg}
        <div style="font-size:0.75rem; font-weight:650; color:{color}; margin-top:6px; line-height:1.1;">{ptype}</div>
    </div>
    """

# ---------------------------------------------------------------------------
# LOGIC CONTROLLERS
# ---------------------------------------------------------------------------
def reset_prediction_form():
    for key in list(st.session_state.keys()):
        if key.startswith("pred_") or key in ["selected_state", "selected_area", "map_center", "map_zoom", "address_input", "selected_ptype"]:
            del st.session_state[key]

def known_areas_for_state(data: pd.DataFrame, state: str) -> list[str]:
    subset = data.loc[data["State"] == state, "Area_Clean"].dropna().unique()
    return sorted({display_name(a) for a in subset})

def analyze_address(data, available_states):
    """Fired when the user types an address. Detects State & Area, then geocodes."""
    addr = st.session_state.get("address_input", "").lower()
    if not addr: return

    matched_state = None
    for st_name in sorted(available_states, key=len, reverse=True):
        if st_name.lower() in addr:
            matched_state = st_name
            st.session_state["selected_state"] = st_name
            break
            
    if matched_state:
        areas = known_areas_for_state(data, matched_state)
        for a in sorted(areas, key=len, reverse=True):
            if a.lower() in addr:
                st.session_state["selected_area"] = a
                break

    if HAS_GEOPY:
        try:
            geolocator = Nominatim(user_agent="mh_estimator", timeout=3)
            location = geolocator.geocode(f"{addr}, Malaysia")
            if location:
                st.session_state["map_center"] = [location.latitude, location.longitude]
                st.session_state["map_zoom"] = 13 
                return
        except Exception:
            pass 
            
    if matched_state and matched_state in STATE_COORDS:
        st.session_state["map_center"] = STATE_COORDS[matched_state]
        st.session_state["map_zoom"] = 8

# ---------------------------------------------------------------------------
# PAGE 1 - PREDICTION INTERFACE
# ---------------------------------------------------------------------------
def prediction_page(data, results):
    recommended = results.iloc[0]["Model"]
    available_states = sorted(data["State"].unique())
    ptypes = sorted(data["Primary_Type"].unique())

    # Session State Initialization
    if "selected_state" not in st.session_state:
        st.session_state["selected_state"] = None
    if "selected_area" not in st.session_state:
        st.session_state["selected_area"] = NO_AREA
    if "selected_ptype" not in st.session_state:
        st.session_state["selected_ptype"] = ptypes[0]

    current_state = st.session_state["selected_state"]
    current_area = st.session_state["selected_area"]
    
    st.markdown("<h3 style='margin-top:0;'>📍 1. Location Selection</h3>", unsafe_allow_html=True)
    
    # Address Input Parser
    st.text_input("Enter your address to auto-detect location, or click the map below:", 
                  placeholder="e.g. Jalan Mewah, Kulai, Johor",
                  key="address_input", on_change=analyze_address, args=(data, available_states))

    # Dynamic Map Control
    map_center = st.session_state.get("map_center", [4.2105, 108.9758] if not current_state else STATE_COORDS.get(current_state, [4.2105, 108.9758]))
    map_zoom = st.session_state.get("map_zoom", 6 if not current_state else 9)
    
    m = folium.Map(location=map_center, zoom_start=map_zoom, tiles="OpenStreetMap")

    # Mode 1: No state selected -> Show State Markers
    if not current_state:
        for st_name in available_states:
            if st_name in STATE_COORDS:
                folium.CircleMarker(
                    location=STATE_COORDS[st_name], radius=10, color="#15243A", weight=2,
                    fill=True, fill_color="#2F6FED", fill_opacity=0.8,
                    tooltip=folium.Tooltip(f"<b>{st_name}</b> (Click to explore)", sticky=True),
                    popup=f"STATE:{st_name}"
                ).add_to(m)
    # Mode 2: State selected -> Show Top Area Markers for that state
    else:
        top_areas = data[data["State"] == current_state]["Area_Clean"].value_counts().head(20).index
        for area_clean in top_areas:
            disp_area = display_name(area_clean)
            coords = get_area_coords(disp_area, current_state)
            if coords:
                is_sel = (disp_area == current_area)
                folium.CircleMarker(
                    location=coords, radius=10 if is_sel else 6,
                    color="#18875D" if is_sel else "#C47A10", weight=3 if is_sel else 2,
                    fill=True, fill_color="#10B981" if is_sel else "#F59E0B", fill_opacity=0.9 if is_sel else 0.7,
                    tooltip=folium.Tooltip(f"<b>{disp_area}</b> (Click to select)", sticky=True),
                    popup=f"AREA:{disp_area}"
                ).add_to(m)

    map_data = st_folium(m, height=350, use_container_width=True, key="malaysia_map")
    
    # Process Map Clicks
    if map_data and map_data.get("last_object_clicked_popup"):
        popup_txt = map_data["last_object_clicked_popup"]
        if popup_txt.startswith("STATE:"):
            clicked_st = popup_txt.split(":")[1]
            st.session_state["selected_state"] = clicked_st
            st.session_state["selected_area"] = NO_AREA
            st.session_state["map_center"] = STATE_COORDS.get(clicked_st, [4.2105, 108.9758])
            st.session_state["map_zoom"] = 9
            st.rerun()
        elif popup_txt.startswith("AREA:"):
            clicked_area = popup_txt.split(":")[1]
            st.session_state["selected_area"] = clicked_area
            st.rerun()

    # Location Readout & Reset
    col_loc1, col_loc2 = st.columns([4, 1])
    with col_loc1:
        st.markdown(f"**State:** {current_state or 'Not Selected'} &nbsp; | &nbsp; **Area:** {current_area.replace(NO_AREA, 'Not Selected')}")
    with col_loc2:
        if st.button("Reset Location", use_container_width=True):
            st.session_state["selected_state"] = None
            st.session_state["selected_area"] = NO_AREA
            st.session_state["map_center"] = [4.2105, 108.9758]
            st.session_state["map_zoom"] = 6
            st.rerun()

    st.markdown('<hr class="mh-rule">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>🏡 2. Property Details</h3>", unsafe_allow_html=True)

    # Visual Property Type Grid
    field_label("Select Property Type")
    svg_cols = st.columns(len(ptypes))
    for i, pt in enumerate(ptypes):
        with svg_cols[i]:
            is_sel = (pt == st.session_state["selected_ptype"])
            st.markdown(ptype_svg_card(pt, is_sel), unsafe_allow_html=True)
            if st.button(f"Select", key=f"btn_{pt}", use_container_width=True):
                st.session_state["selected_ptype"] = pt
                st.rerun()

    # Numerical & Categorical Inputs
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        field_label("Tenure")
        tenure = st.selectbox("Tenure", sorted(data["Tenure"].unique()), key="pred_tenure", label_visibility="collapsed")
    with col_in2:
        field_label("Median price per sq ft (RM)")
        psf = st.number_input("PSF", min_value=1, step=10, value=int(round(data["Median_PSF"].median())), key="pred_psf", label_visibility="collapsed")
    with col_in3:
        field_label("Transactions")
        transactions = st.number_input("Transactions", min_value=0, step=1, value=int(round(data["Transactions"].median())), key="pred_txn", label_visibility="collapsed")

    # Prediction Action
    st.markdown('<br>', unsafe_allow_html=True)
    predict_clicked = st.button("Generate Price Estimate  →", type="primary", use_container_width=True)

    # ---------------- RESULT RENDER ----------------
    if predict_clicked:
        if not current_state:
            st.error("Please select a State from the map or address input before predicting.")
            return

        model = load_model(recommended)
        area_key = create_area_key(current_state, current_area if current_area != NO_AREA else "")
        ptype = st.session_state["selected_ptype"]
        
        features = pd.DataFrame([{
            "State": current_state, "Area_Key": area_key, "Tenure": tenure,
            "Primary_Type": ptype, "Median_PSF": psf, "Transactions": transactions
        }])[MODEL_FEATURES]
        
        with st.spinner("Calculating estimate..."):
            prediction = float(model.predict(features)[0])
            
        metrics = results[results["Model"] == recommended].iloc[0]
        location_str = f"{current_area}, {current_state}" if current_area != NO_AREA else current_state

        st.markdown(f'''
        <div class="mh-result" id="estimate-result">
            <div class="cap">Estimated median price</div>
            <div class="price">RM {prediction:,.0f}</div>
            <div class="sub">{location_str} · {ptype} · {tenure}</div>
            <div class="rule"></div>
            <div class="mh-stats">
                <div><div class="k">Model</div><div class="v">{recommended}</div></div>
                <div><div class="k">Test MAE</div><div class="v">RM {metrics['MAE_test']/1000:,.1f}K</div></div>
                <div><div class="k">Test R²</div><div class="v">{metrics['R2_test']:.3f}</div></div>
            </div>
        </div>''', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# PAGE 2 & 3
# ---------------------------------------------------------------------------
FIGURE_GROUPS = {
    "Data quality": [("fig01_raw_target_distribution.png", "House prices are heavily skewed."), ("fig10_outlier_before_after.png", "Extreme values removed by outlier cleaning.")],
    "Area quality and coverage": [("fig14_top20_areas.png", "The 20 areas with the most records in the dataset."), ("fig16_area_price_distribution.png", "How median price varies across different areas.")],
    "Location and property": [("fig19_state_price_distribution.png", "Median price differs a lot from state to state."), ("fig20_property_type_price.png", "Bungalows and semi-detached homes cost more.")],
}

def insights_page(data):
    view = st.radio("View", ["Market Explorer", "Visual Insights"], horizontal=True, label_visibility="collapsed")
    if view == "Market Explorer":
        st.markdown("#### Historical 2025 dataset exploration")
        a, b, c, d = st.columns(4)
        state = a.selectbox("State", ["All"] + sorted(data["State"].unique()))
        area = b.selectbox("Area", ["All"] + sorted(data["Area_Clean"].unique()))
        ptype = c.selectbox("Property type", ["All"] + sorted(data["Primary_Type"].unique()))
        tenure = d.selectbox("Tenure", ["All"] + sorted(data["Tenure"].unique()))
        
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
            st.dataframe(subset, use_container_width=True, hide_index=True)
    else:
        group = st.selectbox("Insight category", list(FIGURE_GROUPS))
        for filename, caption in FIGURE_GROUPS[group]:
            path = FIGURES_DIR / filename
            if path.exists():
                st.image(str(path), use_container_width=True)

def model_report_page(results):
    recommended = results.iloc[0]
    st.subheader("Model Report")
    st.dataframe(results, use_container_width=True, hide_index=True)

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
