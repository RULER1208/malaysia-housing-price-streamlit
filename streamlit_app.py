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

# Massively expanded pre-calculated real-world coordinates for exact map pins
HARDCODED_AREAS = {
    "Selangor": {
        "Sekinchan": [3.5053, 101.1036], "Tanjong Karang": [3.4267, 101.1773],
        "Pandamaran": [3.0132, 101.4172], "Kuala Selangor": [3.3364, 101.2504],
        "Sabak Bernam": [3.7667, 100.9833], "Sungai Besar": [3.6833, 100.9833],
        "Banting": [2.8155, 101.4975], "Petaling Jaya": [3.1073, 101.6067], 
        "Shah Alam": [3.0738, 101.5183], "Subang Jaya": [3.0471, 101.5832],
        "Klang": [3.0449, 101.4456], "Puchong": [3.0246, 101.6168], 
        "Kajang": [2.9935, 101.7892], "Cheras": [3.1062, 101.7690],
        "Rawang": [3.3213, 101.5822], "Cyberjaya": [2.9228, 101.6572],
        "Setia Alam": [3.1110, 101.4450], "Bukit Beruntung": [3.3100, 101.5540],
        "Bandar Saujana Putra": [2.9490, 101.5790], "Semenyih": [2.9480, 101.8440],
        "Bangi": [2.9200, 101.7800], "Serdang": [3.0220, 101.7100],
        "Batu Caves": [3.2380, 101.6810], "Ampang": [3.1490, 101.7610],
        "Sungai Buloh": [3.2080, 101.5790], "Gombak": [3.2200, 101.7000],
        "Sepang": [2.6865, 101.7483], "Selayang": [3.2505, 101.6448],
    },
    "Kuala Lumpur": {
        "Bukit Bintang": [3.1460, 101.7110], "Setapak": [3.1895, 101.7058], 
        "Kepong": [3.2120, 101.6358], "Mont Kiara": [3.1672, 101.6508], 
        "Bukit Jalil": [3.0578, 101.6885], "Wangsa Maju": [3.2045, 101.7348], 
        "Bangsar": [3.1253, 101.6749], "Old Klang Road": [3.0830, 101.6740],
    },
    "Johor": {
        "Skudai": [1.5333, 103.6667], "Tebrau": [1.5833, 103.7500], 
        "Pasir Gudang": [1.4703, 103.8966], "Kulai": [1.6561, 103.6023], 
        "Johor Bahru": [1.4927, 103.7414], "Batu Pahat": [1.8548, 102.9325],
        "Kluang": [2.0251, 103.3328], "Muar": [2.0442, 102.5689], 
        "Pontian": [1.4883, 103.3888], "Kota Tinggi": [1.7381, 103.8999],
        "Segamat": [2.5144, 102.8159], "Mersing": [2.4312, 103.8361],
    },
    "Perak": {
        "Tapah": [4.2000, 101.2600], "Ipoh": [4.5975, 101.0901],
        "Taiping": [4.8500, 100.7333], "Teluk Intan": [4.0259, 101.0213],
        "Sitiawan": [4.2144, 100.6974], "Seri Manjung": [4.1950, 100.6650], 
        "Kampar": [4.3000, 101.1500], "Lumut": [4.2333, 100.6333],
        "Chenderiang": [4.2667, 101.2333],
    },
    "Penang": {
        "Georgetown": [5.4141, 100.3288], "Butterworth": [5.3995, 100.3638], 
        "Bayan Lepas": [5.2952, 100.2588], "Tasek Gelugor": [5.4833, 100.4833],
        "Bukit Mertajam": [5.3629, 100.4666], "Perai": [5.3833, 100.3833],
        "Batu Kawan": [5.2652, 100.4283], "Nibong Tebal": [5.1667, 100.4667],
        "Kepala Batas": [5.5167, 100.4333],
    },
    "Melaka": {
        "Bemban": [2.2667, 102.3667], "Jasin": [2.3130, 102.4312],
        "Ayer Keroh": [2.2642, 102.2858], "Alor Gajah": [2.3833, 102.2000],
    },
    "Negeri Sembilan": {
        "Seremban": [2.7297, 101.9381], "Port Dickson": [2.5228, 101.7959],
        "Nilai": [2.8167, 101.8000],
    },
    "Kedah": {
        "Alor Setar": [6.1210, 100.3601], "Sungai Petani": [5.6436, 100.4897],
        "Kulim": [5.3667, 100.5500],
    },
    "Pahang": {
        "Kuantan": [3.8077, 103.3260], "Temerloh": [3.4506, 102.4168], 
        "Cameron Highlands": [4.4721, 101.3801]
    },
    "Kelantan": { "Kota Bharu": [6.1254, 102.2381] },
    "Terengganu": { "Kuala Terengganu": [5.3302, 103.1408], "Kemaman": [4.2333, 103.3333] },
    "Sabah": { "Kota Kinabalu": [5.9804, 116.0735] },
    "Sarawak": { "Kuching": [1.5533, 110.3592] }
}

# ---------------------------------------------------------------------------
# STYLES & FLAWLESS CSS GRID OVERLAY
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
.mh-empty { background:#FFFFFF; border:1px dashed #CFD8E6; border-radius:14px; padding:52px 24px; text-align:center; color:var(--muted); box-shadow:var(--shadow); margin-top:20px;}

/* Custom Tenure Radio Pills layout */
div[role="radiogroup"] { gap: 15px; }

/* Property type now uses real Streamlit buttons.
   No invisible overlay button is needed, so the blank box is removed. */
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
    if state_name in HARDCODED_AREAS and area_name in HARDCODED_AREAS[state_name]:
        return HARDCODED_AREAS[state_name][area_name]
    
    if HAS_GEOPY:
        try:
            geolocator = Nominatim(user_agent="mh_estimator", timeout=2)
            loc = geolocator.geocode(f"{area_name}, {state_name}, Malaysia")
            if loc: 
                time.sleep(0.5) 
                return [loc.latitude, loc.longitude]
        except: pass
            
    # Do NOT invent a map position when an area cannot be geocoded.
    # Returning None is safer than showing a wrong pin. The area is still
    # selectable from the dataset-driven Area dropdown below.
    return None

def field_label(text: str) -> None:
    st.markdown(f'<div class="mh-label">{text}</div>', unsafe_allow_html=True)

def get_colored_svg(ptype: str, is_selected: bool) -> str:
    """Returns a fully colored SVG card tailored to property types, using a green border for selection."""
    c_roof = "#E63946"  
    c_wall = "#F1FAEE"  
    c_door = "#1D3557"  
    c_glass = "#A8DADC" 
    c_frame = "#457B9D" 
    c_accent = "#F4A261" 
    
    # Selection styling focuses on Green border and subtle scaling, retaining full colors either way
    bg = "#F0FDF4" if is_selected else "#FFFFFF"
    border = "2px solid #10B981" if is_selected else "1px solid #E2E7EF"
    txt_color = "#18875D" if is_selected else "#667085"
    filter_style = "transform: scale(1.05);" if is_selected else "transform: scale(1.0); opacity: 0.85;"
    
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
        
    return f'<div class="ptype-card" style="border:{border}; background:{bg}; border-radius:12px; padding:15px 4px; text-align:center; height:105px; display:flex; flex-direction:column; justify-content:center; align-items:center;"><svg style="{filter_style} transition: all 0.2s ease-in-out;" width="42" height="42" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">{svg}</svg><div style="font-size:0.75rem; font-weight:650; color:{txt_color}; margin-top:8px; line-height:1.1;">{ptype}</div></div>'

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
    addr = st.session_state.get("address_input", "").lower()
    if not addr: return

    matched_state = None
    matched_area = None
    
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

    if not matched_state:
        for st_name in sorted(available_states, key=len, reverse=True):
            if st_name.lower() in addr:
                matched_state = st_name
                break
                
    if matched_state and not matched_area:
        valid_areas = set([display_name(a) for a in data[data["State"] == matched_state]["Area_Clean"].dropna().unique()])
        for a in sorted(valid_areas, key=len, reverse=True):
            if a.lower() in addr:
                matched_area = a
                break

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

    # Malaysia = 13 states + 3 Federal Territories.
    expected_states = {
        "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan",
        "Pahang", "Penang", "Perak", "Perlis", "Sabah", "Sarawak",
        "Selangor", "Terengganu", "Kuala Lumpur", "Putrajaya", "Labuan"
    }
    dataset_states = set(available_states)
    missing_states = sorted(expected_states - dataset_states)
    unexpected_states = sorted(dataset_states - expected_states)

    st.markdown("<h3 style='margin-top:0;'>📍 1. Location Selection</h3>", unsafe_allow_html=True)

    if missing_states:
        st.warning("Dataset is missing state(s): " + ", ".join(missing_states))
    if unexpected_states:
        st.warning("Unrecognised state value(s) in dataset: " + ", ".join(unexpected_states))

    # Dataset-driven State -> Area selectors guarantee that every selectable area
    # is assigned to the same state used during model training.
    loc_a, loc_b = st.columns(2)
    with loc_a:
        state_options = ["— Select State —"] + available_states
        state_index = state_options.index(current_state) if current_state in state_options else 0
        selected_state_list = st.selectbox("State", state_options, index=state_index, key="state_list")

    if selected_state_list != (current_state or "— Select State —"):
        if selected_state_list == "— Select State —":
            reset_location_state()
        else:
            st.session_state["selected_state"] = selected_state_list
            st.session_state["selected_area"] = None
            st.session_state["map_center"] = STATE_COORDS.get(selected_state_list, [4.2105, 108.9758])
            st.session_state["map_zoom"] = 9
        st.rerun()

    current_state = st.session_state["selected_state"]
    current_area = st.session_state["selected_area"]

    with loc_b:
        if current_state:
            state_areas = sorted(
                {display_name(a) for a in data.loc[data["State"] == current_state, "Area_Clean"].dropna().unique()}
            )
            area_options = ["— Select Area —"] + state_areas
            area_index = area_options.index(current_area) if current_area in area_options else 0
            selected_area_list = st.selectbox("Area", area_options, index=area_index, key=f"area_list_{current_state}")
        else:
            state_areas = []
            selected_area_list = st.selectbox("Area", ["— Select Area —"], disabled=True)

    if current_state and selected_area_list != (current_area or "— Select Area —"):
        if selected_area_list == "— Select Area —":
            st.session_state["selected_area"] = None
            st.session_state["map_center"] = STATE_COORDS.get(current_state, [4.2105, 108.9758])
            st.session_state["map_zoom"] = 9
        else:
            st.session_state["selected_area"] = selected_area_list
            coords = get_area_coords(selected_area_list, current_state)
            if coords:
                st.session_state["map_center"] = coords
                st.session_state["map_zoom"] = 12
        st.rerun()

    st.text_input("Enter your address or postcode to auto-detect location, or click the map below:", 
                  placeholder="e.g. 45400 or Sekinchan, Selangor",
                  key="address_input", on_change=analyze_address, args=(data, available_states))

    current_state = st.session_state["selected_state"]
    current_area = st.session_state["selected_area"]
    map_center = st.session_state.get("map_center", [4.2105, 108.9758] if not current_state else STATE_COORDS.get(current_state, [4.2105, 108.9758]))
    map_zoom = st.session_state.get("map_zoom", 6 if not current_state else 9)
    
    m = folium.Map(location=map_center, zoom_start=map_zoom, tiles="OpenStreetMap")

    if not current_state:
        for st_name in available_states:
            if st_name in STATE_COORDS:
                folium.CircleMarker(
                    location=STATE_COORDS[st_name], radius=11, color="#15243A", weight=2,
                    fill=True, fill_color="#2F6FED", fill_opacity=0.8,
                    tooltip=folium.Tooltip(f"<b>{st_name}</b> (Click to select state)", sticky=True),
                    popup=f"STATE:{st_name}"
                ).add_to(m)
                
    else:
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

    # ---------------- PROPERTY TYPE: REAL BUTTONS, NO EMPTY OVERLAY ----------------
    field_label("Select Property Type")
    property_icons = {
        "Bungalow": "🏠",
        "Semi D": "🏘️",
        "Terrace House": "🏡",
        "Condominium": "🏢",
        "Apartment": "🏬",
        "Flat": "🏬",
        "Service Residence": "🏙️",
    }
    ptype_cols = st.columns(len(ptypes))

    for i, pt in enumerate(ptypes):
        with ptype_cols[i]:
            is_sel = (pt == st.session_state["selected_ptype"])
            icon = property_icons.get(pt, "🏠")
            if st.button(
                f"{icon}  {pt}",
                key=f"btn_{pt}",
                type="primary" if is_sel else "secondary",
                use_container_width=True,
            ):
                st.session_state["selected_ptype"] = pt
                st.rerun()

    # Numerical Inputs
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        field_label("Tenure")
        # Direct clickable horizontal radio pills for Tenure
        tenure = st.radio("Tenure", sorted(data["Tenure"].unique()), key="pred_tenure", label_visibility="collapsed", horizontal=True)
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
        from area_preprocessing import create_area_key
        area_key = create_area_key(current_state, current_area)
        ptype = st.session_state["selected_ptype"]
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

if __name__ == "__main__":
    main()
