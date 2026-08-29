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
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import hashlib
import time
import re

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
# CONFIGURATION & REAL-WORLD COORDINATES
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "malaysia_house_price_cleaned_with_area.csv"
RESULTS_PATH = APP_DIR / "model_results.csv"
MODELS_DIR = APP_DIR / "models"
FIGURES_DIR = APP_DIR / "figures"
MODEL_FEATURES = ["State", "Area_Key", "Tenure", "Primary_Type", "Median_PSF", "Transactions"]

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

HARDCODED_AREAS = {
    "Skudai": [1.5333, 103.6667], "Tebrau": [1.5833, 103.7500], "Pasir Gudang": [1.4703, 103.8966], 
    "Kulai": [1.6561, 103.6023], "Johor Bahru": [1.4927, 103.7414], "Batu Pahat": [1.8548, 102.9325],
    "Kluang": [2.0251, 103.3328], "Muar": [2.0442, 102.5689], "Sekinchan": [3.5053, 101.1036], 
    "Tanjong Karang": [3.4267, 101.1773], "Petaling Jaya": [3.1073, 101.6067], "Shah Alam": [3.0738, 101.5183], 
    "Subang Jaya": [3.0471, 101.5832], "Klang": [3.0449, 101.4456], "Puchong": [3.0246, 101.6168], 
    "Kajang": [2.9935, 101.7892], "Cheras": [3.1062, 101.7690], "Tapah": [4.2000, 101.2600], 
    "Ipoh": [4.5975, 101.0901], "Taiping": [4.8500, 100.7333], "Georgetown": [5.4141, 100.3288], 
    "Butterworth": [5.3995, 100.3638], "Bayan Lepas": [5.2952, 100.2588], "Bemban": [2.2667, 102.3667], 
    "Jasin": [2.3130, 102.4312], "Seremban": [2.7297, 101.9381], "Alor Setar": [6.1210, 100.3601], 
    "Kuantan": [3.8077, 103.3260], "Kota Kinabalu": [5.9804, 116.0735], "Kuching": [1.5533, 110.3592]
}

# ---------------------------------------------------------------------------
# STYLES & INVISIBLE OVERLAY HACK FOR SVG CLICKS
# ---------------------------------------------------------------------------
st.markdown(r"""
<style>
:root { --navy:#15243A; --blue:#2F6FED; --muted:#667085; --border:#E2E7EF; --mono:ui-monospace,"SF Mono",monospace; }
.stApp { background:#F6F8FB; color:#172033; }
.block-container { max-width:1180px; padding-top:12px!important; padding-bottom:2rem; }
header[data-testid="stHeader"] { background:transparent!important; }

/* Navigation Banner */
.stTabs [role="tablist"] {
    display:flex!important; align-items:center!important; gap:8px!important;
    min-height:62px; padding:0 210px 0 22px!important;
    background:var(--navy)!important; border-radius:14px!important; margin-bottom:22px!important;
}
.stTabs [role="tablist"]::before { content:"\2302\00a0\00a0Malaysia Housing Estimator"; font-size:1.04rem; font-weight:700; color:#FFFFFF; margin-right:24px; }
.stTabs [role="tab"] { height:38px!important; padding:0 18px!important; border-radius:999px!important; color:#B9C8DF!important; font-weight:600; background:transparent!important; border:none!important; }
.stTabs [role="tab"][aria-selected="true"] { color:var(--navy)!important; background:#FFFFFF!important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none!important; }

/* Labels & Results */
.mh-label { font-family:var(--mono); font-size:.72rem; letter-spacing:.1em; color:var(--muted); margin:2px 0 5px 0; text-transform:uppercase; }
.mh-rule { border:none; border-top:1px solid var(--border); margin:16px 0 14px 0; }
.mh-result { background:var(--navy); border-radius:14px; padding:26px; box-shadow:0 6px 22px rgba(24,49,83,.06); margin-top:20px; }
.mh-result .cap { font-family:var(--mono); font-size:.72rem; letter-spacing:.14em; color:#9FB4D4; text-transform:uppercase; }
.mh-result .price { font-size:2.7rem; font-weight:750; color:#FFFFFF; margin:8px 0 6px; }
.mh-result .sub { color:#C9D7EC; font-size:.95rem; }
.mh-result .rule { border-top:1px solid rgba(255,255,255,.16); margin:18px 0 14px; }
.mh-stats { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }
.mh-stats .k { font-family:var(--mono); font-size:.68rem; letter-spacing:.12em; color:#9FB4D4; text-transform:uppercase; margin-bottom:3px; }
.mh-stats .v { font-family:var(--mono); font-size:.94rem; font-weight:700; color:#FFFFFF; }
.stButton>button[kind="primary"] { background:var(--blue); border-color:var(--blue); border-radius:11px; min-height:44px;}
.mh-empty { background:#FFFFFF; border:1px dashed #CFD8E6; border-radius:14px; padding:52px 24px; text-align:center; color:var(--muted); margin-top:20px;}

/* CSS HACK: Stretches the Streamlit button over the SVG box to make it directly clickable */
div[data-testid="column"]:has(svg) {
    position: relative !important;
}
div[data-testid="column"]:has(svg) div[data-testid="stButton"] {
    position: absolute !important;
    top: 0 !important; left: 0 !important;
    width: 100% !important; height: 100% !important;
    z-index: 999 !important;
}
div[data-testid="column"]:has(svg) button {
    width: 100% !important; height: 100% !important;
    opacity: 0 !important; cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DATA & MAP HELPERS
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data(show_spinner=False)
def load_results():
    return pd.read_csv(RESULTS_PATH).sort_values(["Group_CV_RMSE_mean"]).reset_index(drop=True)

@st.cache_resource(show_spinner=False)
def load_model(name):
    filename = name.split(" (")[0].lower().replace(" ", "_") + ".pkl"
    return joblib.load(MODELS_DIR / filename)

@st.cache_data(show_spinner=False)
def get_area_coords(area_name: str, state_name: str):
    if area_name in HARDCODED_AREAS:
        return HARDCODED_AREAS[area_name]
    
    if HAS_GEOPY:
        try:
            geolocator = Nominatim(user_agent="mh_estimator", timeout=2)
            loc = geolocator.geocode(f"{area_name}, {state_name}, Malaysia")
            if loc: 
                time.sleep(0.5) 
                return [loc.latitude, loc.longitude]
        except: pass
            
    base_coords = STATE_COORDS.get(state_name, [4.2105, 108.9758])
    hash_val = int(hashlib.md5(area_name.encode('utf-8')).hexdigest(), 16)
    lat_offset = ((hash_val % 1000) / 1000.0 - 0.5) * 0.8
    lon_offset = (((hash_val // 1000) % 1000) / 1000.0 - 0.5) * 0.8
    return [base_coords[0] + lat_offset, base_coords[1] + lon_offset]

def field_label(text: str) -> None:
    st.markdown(f'<div class="mh-label">{text}</div>', unsafe_allow_html=True)

def get_colored_svg(ptype: str, is_selected: bool) -> str:
    c_roof = "#E63946"  # Red
    c_wall = "#F1FAEE"  # Off-white
    c_door = "#1D3557"  # Navy
    c_glass = "#A8DADC" # Light Blue
    c_frame = "#457B9D" # Muted Blue
    c_accent = "#F4A261" # Orange
    
    filter_style = "filter: grayscale(0%) opacity(100%); transform: scale(1.05);" if is_selected else "filter: grayscale(100%) opacity(40%);"
    
    if ptype == "Bungalow":
        svg = f'<path fill="{c_roof}" d="M12 2L2 12h3v10h14V12h3L12 2z"/><rect fill="{c_wall}" x="5" y="12" width="14" height="10"/><rect fill="{c_door}" x="10" y="14" width="4" height="8"/><rect fill="{c_glass}" x="6" y="14" width="3" height="4"/><rect fill="{c_glass}" x="15" y="14" width="3" height="4"/>'
    elif ptype == "Semi D":
        svg = f'<path fill="{c_roof}" d="M12 2L2 12h10V2z"/><path fill="{c_accent}" d="M12 2l10 10h-10V2z"/><rect fill="{c_wall}" x="5" y="12" width="14" height="10"/><rect fill="{c_door}" x="6" y="14" width="4" height="8"/><rect fill="{c_door}" x="14" y="14" width="4" height="8"/>'
    elif ptype == "Terrace House":
        svg = f'<path fill="{c_roof}" d="M1 10l3-3v-3h2v1l2-2 4 4V10H1z"/><path fill="{c_accent}" d="M9 10l3-3v-3h2v1l2-2 4 4V10H9z"/><path fill="{c_frame}" d="M17 10l3-3v-3h2v1l2-2 4 4V10h-11z"/><rect fill="{c_wall}" x="2" y="10" width="20" height="12"/><rect fill="{c_door}" x="3" y="14" width="3" height="8"/><rect fill="{c_door}" x="11" y="14" width="3" height="8"/><rect fill="{c_door}" x="19" y="14" width="3" height="8"/>'
    elif ptype == "Condominium":
        svg = f'<rect fill="{c_frame}" x="5" y="2" width="14" height="20"/><rect fill="{c_glass}" x="7" y="4" width="4" height="4"/><rect fill="{c_glass}" x="13" y="4" width="4" height="4"/><rect fill="{c_glass}" x="7" y="10" width="4" height="4"/><rect fill="{c_glass}" x="13" y="10" width="4" height="4"/><rect fill="{c_glass}" x="7" y="16" width="4" height="4"/><rect fill="{c_glass}" x="13" y="16" width="4" height="4"/>'
    elif ptype in ["Apartment", "Flat", "Service Residence"]:
        svg = f'<rect fill="{c_accent}" x="4" y="4" width="16" height="18"/><rect fill="{c_wall}" x="6" y="6" width="3" height="3"/><rect fill="{c_wall}" x="10" y="6" width="3" height="3"/><rect fill="{c_wall}" x="14" y="6" width="3" height="3"/><rect fill="{c_wall}" x="6" y="11" width="3" height="3"/><rect fill="{c_wall}" x="10" y="11" width="3" height="3"/><rect fill="{c_wall}" x="14" y="11" width="3" height="3"/><rect fill="{c_door}" x="10" y="16" width="4" height="6"/>'
    else:
        svg = f'<path fill="{c_frame}" d="M12 3L2 12h3v10h14V12h3L12 3z"/><rect fill="{c_wall}" x="5" y="12" width="14" height="10"/><rect fill="{c_accent}" x="10" y="15" width="4" height="7"/>'
        
    return f'<svg style="{filter_style} transition: all 0.2s ease-in-out;" width="42" height="42" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">{svg}</svg>'

# ---------------------------------------------------------------------------
# LOGIC CONTROLLERS
# ---------------------------------------------------------------------------
def reset_location_state():
    st.session_state["selected_state"] = None
    st.session_state["selected_area"] = None
    st.session_state["map_center"] = [4.2105, 108.9758]
    st.session_state["map_zoom"] = 6
    st.session_state["address_input"] = ""

def analyze_address(data, available_states):
    """Detects Postcodes & Text directly and extracts State & Area via Geopy."""
    addr = st.session_state.get("address_input", "").lower()
    if not addr: return

    matched_state = None
    matched_area = None
    
    # 1. Postcode Geocoding Check
    postcode_match = re.search(r'\b\d{5}\b', addr)
    if postcode_match and HAS_GEOPY:
        postcode = postcode_match.group()
        try:
            geolocator = Nominatim(user_agent="mh_estimator", timeout=3)
            loc = geolocator.geocode(f"{postcode}, Malaysia", addressdetails=True)
            if loc and 'address' in loc.raw:
                addr_details = loc.raw['address']
                raw_state = addr_details.get('state', '').lower()
                raw_area = addr_details.get('town', addr_details.get('city', addr_details.get('county', addr_details.get('suburb', ''))))
                
                for st_name in available_states:
                    if st_name.lower() in raw_state:
                        matched_state = st_name
                        break
                if raw_area:
                    matched_area = display_name(raw_area)
        except: pass

    # 2. String Match Fallback
    if not matched_state:
        for st_name in sorted(available_states, key=len, reverse=True):
            if st_name.lower() in addr:
                matched_state = st_name
                break
                
    if matched_state and not matched_area:
        valid_areas = data[data["State"] == matched_state]["Area_Clean"].dropna().unique()
        valid_disp_areas = [display_name(a) for a in valid_areas]
        for a in sorted(valid_disp_areas, key=len, reverse=True):
            if a.lower() in addr:
                matched_area = a
                break

    # 3. Apply state and zoom changes
    if matched_state:
        st.session_state["selected_state"] = matched_state
        if matched_area:
            st.session_state["selected_area"] = matched_area
            coords = get_area_coords(matched_area, matched_state)
            if coords:
                st.session_state["map_center"] = coords
                st.session_state["map_zoom"] = 12
        else:
            st.session_state["selected_area"] = None
            st.session_state["map_center"] = STATE_COORDS.get(matched_state, [4.2105, 108.9758])
            st.session_state["map_zoom"] = 8

# ---------------------------------------------------------------------------
# PAGE 1 - PREDICTION INTERFACE
# ---------------------------------------------------------------------------
def prediction_page(data, results):
    recommended = results.iloc[0]["Model"]
    available_states = sorted(data["State"].unique())
    ptypes = sorted(data["Primary_Type"].unique())

    if "selected_state" not in st.session_state: st.session_state["selected_state"] = None
    if "selected_area" not in st.session_state: st.session_state["selected_area"] = None
    if "selected_ptype" not in st.session_state: st.session_state["selected_ptype"] = ptypes[0]
    if "address_input" not in st.session_state: st.session_state["address_input"] = ""

    current_state = st.session_state["selected_state"]
    current_area = st.session_state["selected_area"]
    
    st.markdown("<h3 style='margin-top:0;'>📍 1. Location Selection</h3>", unsafe_allow_html=True)
    
    st.text_input("Enter your address/postcode to auto-detect location, or click the map below:", 
                  placeholder="e.g. 45400 or Sekinchan, Selangor",
                  key="address_input", on_change=analyze_address, args=(data, available_states))

    map_center = st.session_state.get("map_center", [4.2105, 108.9758] if not current_state else STATE_COORDS.get(current_state, [4.2105, 108.9758]))
    map_zoom = st.session_state.get("map_zoom", 6 if not current_state else 9)
    
    m = folium.Map(location=map_center, zoom_start=map_zoom, tiles="OpenStreetMap")

    # Map Mode 1: Malaysia State Overview
    if not current_state:
        for st_name in available_states:
            if st_name in STATE_COORDS:
                folium.CircleMarker(
                    location=STATE_COORDS[st_name], radius=11, color="#15243A", weight=2,
                    fill=True, fill_color="#2F6FED", fill_opacity=0.8,
                    tooltip=folium.Tooltip(f"<b>{st_name}</b> (Click to select state)", sticky=True),
                    popup=f"STATE:{st_name}"
                ).add_to(m)
                
    # Map Mode 2: Dynamic Area Drill-Down
    else:
        # Load all CSV areas for the current state, and include the user's custom area if Geopy found one
        all_state_areas = data[data["State"] == current_state]["Area_Clean"].dropna().unique()
        areas_to_plot = set([display_name(a) for a in all_state_areas])
        if current_area:
            areas_to_plot.add(current_area)
            
        marker_cluster = MarkerCluster(name="Areas").add_to(m)
        
        for disp_area in areas_to_plot:
            coords = get_area_coords(disp_area, current_state)
            if coords:
                is_sel = (disp_area == current_area)
                folium.CircleMarker(
                    location=coords, radius=12 if is_sel else 8,
                    color="#18875D" if is_sel else "#C47A10", weight=3 if is_sel else 2,
                    fill=True, fill_color="#10B981" if is_sel else "#F59E0B", fill_opacity=0.9 if is_sel else 0.7,
                    tooltip=folium.Tooltip(f"<b>{disp_area}</b> (Click to select area)", sticky=True),
                    popup=f"AREA:{disp_area}"
                ).add_to(marker_cluster)

    map_data = st_folium(m, height=350, use_container_width=True, key="malaysia_map")
    
    # Process Map Clicks
    if map_data and map_data.get("last_object_clicked_popup"):
        popup_txt = map_data["last_object_clicked_popup"]
        if popup_txt.startswith("STATE:"):
            clicked_st = popup_txt.split(":")[1]
            st.session_state["selected_state"] = clicked_st
            st.session_state["selected_area"] = None
            st.session_state["map_center"] = STATE_COORDS.get(clicked_st, [4.2105, 108.9758])
            st.session_state["map_zoom"] = 9
            st.rerun()
        elif popup_txt.startswith("AREA:"):
            clicked_area = popup_txt.split(":")[1]
            st.session_state["selected_area"] = clicked_area
            st.session_state["map_center"] = get_area_coords(clicked_area, current_state)
            st.session_state["map_zoom"] = 12
            st.rerun()

    col_loc1, col_loc2 = st.columns([4, 1])
    with col_loc1:
        st.markdown(f"**State:** {current_state or 'Not Selected'} &nbsp; | &nbsp; **Area:** {current_area or 'Not Selected'}")
    with col_loc2:
        st.button("Reset Location", use_container_width=True, on_click=reset_location_state)

    st.markdown('<hr class="mh-rule">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top:0;'>🏡 2. Property Details</h3>", unsafe_allow_html=True)

    # Interactive SVG Grid via CSS Overlay
    field_label("Select Property Type")
    svg_cols = st.columns(len(ptypes))
    for i, pt in enumerate(ptypes):
        with svg_cols[i]:
            is_sel = (pt == st.session_state["selected_ptype"])
            bg = "#EEF4FF" if is_sel else "#FFFFFF"
            border = "2px solid #2F6FED" if is_sel else "1px solid #E2E7EF"
            txt_color = "#2F6FED" if is_sel else "#667085"
            
            # 1. Render the visual card
            st.markdown(f'''
            <div style="border:{border}; background:{bg}; border-radius:12px; padding:15px 4px; text-align:center; display:flex; flex-direction:column; justify-content:center; align-items:center;">
                {get_colored_svg(pt, is_sel)}
                <div style="font-size:0.75rem; font-weight:650; color:{txt_color}; margin-top:8px; line-height:1.1;">{pt}</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # 2. The invisible Streamlit button captures clicks perfectly over the SVG!
            if st.button(" ", key=f"btn_{pt}", use_container_width=True):
                st.session_state["selected_ptype"] = pt
                st.rerun()

    # Numerical Inputs (Transactions Removed)
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        field_label("Tenure")
        tenure = st.selectbox("Tenure", sorted(data["Tenure"].unique()), key="pred_tenure", label_visibility="collapsed")
    with col_in2:
        field_label("Median price per sq ft (RM)")
        psf = st.number_input("PSF", min_value=1, step=10, value=int(round(data["Median_PSF"].median())), key="pred_psf", label_visibility="collapsed")

    st.markdown('<br>', unsafe_allow_html=True)
    predict_clicked = st.button("Generate Price Estimate  →", type="primary", use_container_width=True)

    # ---------------- RESULT RENDER ----------------
    if predict_clicked:
        if not current_state or not current_area:
            st.error("Please select both a State and an Area from the map or address input before predicting.")
            return

        model = load_model(recommended)
        area_key = create_area_key(current_state, current_area)
        ptype = st.session_state["selected_ptype"]
        
        # We silently inject the median historical transactions into the features dataframe to keep the model happy
        transactions = int(round(data["Transactions"].median()))
        
        features = pd.DataFrame([{
            "State": current_state, "Area_Key": area_key, "Tenure": tenure,
            "Primary_Type": ptype, "Median_PSF": psf, "Transactions": transactions
        }])[MODEL_FEATURES]
        
        with st.spinner("Calculating estimate..."):
            prediction = float(model.predict(features)[0])
            
        metrics = results[results["Model"] == recommended].iloc[0]

        st.markdown(f'''
        <div class="mh-result" id="estimate-result">
            <div class="cap">Estimated median price</div>
            <div class="price">RM {prediction:,.0f}</div>
            <div class="sub">{current_area}, {current_state} · {ptype} · {tenure}</div>
            <div class="rule"></div>
            <div class="mh-stats">
                <div><div class="k">Model</div><div class="v">{recommended}</div></div>
                <div><div class="k">Test MAE</div><div class="v">RM {metrics['MAE_test']/1000:,.1f}K</div></div>
                <div><div class="k">Test R²</div><div class="v">{metrics['R2_test']:.3f}</div></div>
            </div>
        </div>''', unsafe_allow_html=True)

    else:
        st.markdown(
            '<div class="mh-empty"><div class="icon">⌂</div>'
            '<b>Your estimate will appear here.</b>'
            'Select location and property details above, then predict.</div>',
            unsafe_allow_html=True
        )

# ---------------------------------------------------------------------------
# PAGE 2 & 3
# ---------------------------------------------------------------------------
FIGURE_GROUPS = {
    "Data quality": [("fig01_raw_target_distribution.png", "House prices are heavily skewed."), ("fig10_outlier_before_after.png", "Extreme values removed by outlier cleaning.")],
    "Area quality and coverage": [("fig14_top20_areas.png", "The 20 areas with the most records in the dataset."), ("fig16_area_price_distribution.png", "How median price varies across different areas.")],
    "Location and property": [("fig18_state_counts_clean.png", "Number of records per state after cleaning."),("fig19_state_price_distribution.png", "Median price differs a lot from state to state."), ("fig20_property_type_price.png", "Bungalows and semi-detached homes cost more.")],
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
