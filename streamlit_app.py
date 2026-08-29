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

HARDCODED_AREAS = {
    # Johor
    "Skudai": [1.5333, 103.6667], "Tebrau": [1.5833, 103.7500],
    "Pasir Gudang": [1.4703, 103.8966], "Kulai": [1.6561, 103.6023],
    "Johor Bahru": [1.4927, 103.7414], "Batu Pahat": [1.8548, 102.9325],
    "Kluang": [2.0251, 103.3328], "Muar": [2.0442, 102.5689],
    "Pontian": [1.4883, 103.3888], "Kota Tinggi": [1.7381, 103.8999],
    "Segamat": [2.5144, 102.8159], "Mersing": [2.4312, 103.8361],
    # Selangor
    "Sekinchan": [3.5053, 101.1036], "Petaling Jaya": [3.1073, 101.6067],
    "Shah Alam": [3.0738, 101.5183], "Subang Jaya": [3.0471, 101.5832],
    "Klang": [3.0449, 101.4456], "Kapar": [3.1396, 101.3752],
    "Puchong": [3.0246, 101.6168], "Cyberjaya": [2.9228, 101.6572],
    "Kajang": [2.9935, 101.7892], "Sepang": [2.6865, 101.7483],
    "Ampang": [3.1496, 101.7610], "Cheras": [3.1062, 101.7690],
    "Rawang": [3.3213, 101.5822], "Selayang": [3.2505, 101.6448],
    "Gombak": [3.2200, 101.7000], "Sungai Buloh": [3.2086, 101.5794],
    "Kuala Selangor": [3.3364, 101.2504], "Banting": [2.8155, 101.4975],
    # Perak
    "Tapah": [4.2000, 101.2600], "Chenderiang": [4.2667, 101.2333],
    "Teluk Intan": [4.0259, 101.0213], "Ipoh": [4.5975, 101.0901],
    "Taiping": [4.8500, 100.7333], "Sitiawan": [4.2144, 100.6974],
    "Seri Manjung": [4.1950, 100.6650], "Kampar": [4.3000, 101.1500],
    "Lumut": [4.2333, 100.6333],
    # Penang
    "Georgetown": [5.4141, 100.3288], "Tasek Gelugor": [5.4833, 100.4833],
    "Butterworth": [5.3995, 100.3638], "Bayan Lepas": [5.2952, 100.2588],
    "Bukit Mertajam": [5.3629, 100.4666], "Perai": [5.3833, 100.3833],
    "Batu Kawan": [5.2652, 100.4283], "Nibong Tebal": [5.1667, 100.4667],
    "Kepala Batas": [5.5167, 100.4333],
    # Melaka
    "Bemban": [2.2667, 102.3667], "Jasin": [2.3130, 102.4312],
    "Ayer Keroh": [2.2642, 102.2858], "Alor Gajah": [2.3833, 102.2000],
    # Negeri Sembilan
    "Seremban": [2.7297, 101.9381], "Port Dickson": [2.5228, 101.7959],
    "Nilai": [2.8167, 101.8000],
    # Kedah
    "Alor Setar": [6.1210, 100.3601], "Sungai Petani": [5.6436, 100.4897],
    "Kulim": [5.3667, 100.5500],
    # Pahang & East Coast & Borneo & KL
    "Kuantan": [3.8077, 103.3260], "Temerloh": [3.4506, 102.4168],
    "Cameron Highlands": [4.4721, 101.3801], "Kota Bharu": [6.1254, 102.2381],
    "Kuala Terengganu": [5.3302, 103.1408], "Kemaman": [4.2333, 103.3333],
    "Kota Kinabalu": [5.9804, 116.0735], "Kuching": [1.5533, 110.3592],
    "Bangsar": [3.1253, 101.6749], "Setapak": [3.1895, 101.7058],
    "Kepong": [3.2120, 101.6358], "Mont Kiara": [3.1672, 101.6508],
    "Bukit Jalil": [3.0578, 101.6885], "Wangsa Maju": [3.2045, 101.7348],
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
    """Retrieve precise map coordinates for an area."""
    if area_name in HARDCODED_AREAS:
        return HARDCODED_AREAS[area_name]
    
    if HAS_GEOPY:
        try:
            geolocator = Nominatim(user_agent="mh_estimator", timeout=2)
            loc = geolocator.geocode(f"{area_name}, {state_name}, Malaysia")
            if loc: 
                time.sleep(1)
                return [loc.latitude, loc.longitude]
        except:
            pass
            
    base_coords = STATE_COORDS.get(state_name, [4.2105, 108.9758])
    hash_val = int(hashlib.md5(area_name.encode('utf-8')).hexdigest(), 16)
    lat_offset = ((hash_val % 1000) / 1000.0 - 0.5) * 0.7
    lon_offset = (((hash_val // 1000) % 1000) / 1000.0 - 0.5) * 0.7
    return [base_coords[0] + lat_offset, base_coords[1] + lon_offset]

def field_label(text: str) -> None:
    st.markdown(f'<div class="mh-label">{text}</div>', unsafe_allow_html=True)

def ptype_svg_card(ptype: str, is_selected: bool) -> str:
    """Returns a specific inline SVG card customized for each property category."""
    color = "#2F6FED" if is_selected else "#667085"
    bg = "#EEF4FF" if is_selected else "#F6F8FB"
    border = "2px solid #2F6FED" if is_selected else "1px solid #E2E7EF"
    
    svg_icons = {
        "Bungalow": f'''<svg width="34" height="34" viewBox="0 0 24 24" fill="{color}"><path d="M12 3L2 12h3v8h14v-8h3L12 3zm1 15h-2v-4h2v4zm4-5h-2V9h2v4zM9 9h2v4H9V9z"/></svg>''',
        "Semi D": f'''<svg width="34" height="34" viewBox="0 0 24 24" fill="{color}"><path d="M2 12l5-4.5V4h4v3.5L16 3l6 5.5V20H2V12zm9 6h-2v-3h2v3zm0-5h-2v-3h2v3zm9 5h-2v-3h2v3zm0-5h-2v-3h2v3z"/></svg>''',
        "Terrace House": f'''<svg width="34" height="34" viewBox="0 0 24 24" fill="{color}"><path d="M1 11l4-3.5V4h3v2L12 3l4 3.5V4h3v3.5l4 3.5V20H1v-9zm6 7H5v-3h2v3zm7 0h-2v-3h2v3zm7 0h-2v-3h2v3z"/></svg>''',
        "Town House": f'''<svg width="34" height="34" viewBox="0 0 24 24" fill="{color}"><path d="M17 11V3H7v4H3v13h18V11h-4zm-8-6h6v4H9V5zm-4 8h2v2H5v-2zm0 4h2v2H5v-2zm6 0h2v2h-2v-2zm0-4h2v2h-2v-2zm6 4h2v2h-2v-2zm0-4h2v2h-2v-2zm0-4h2v2h-2V9z"/></svg>''',
        "Cluster House": f'''<svg width="34" height="34" viewBox="0 0 24 24" fill="{color}"><path d="M3 3h8v8H3V3zm10 0h8v8h-8V3zM3 13h8v8H3v-8zm10 0h8v8h-8v-8z"/></svg>''',
        "Condominium": f'''<svg width="34" height="34" viewBox="0 0 24 24" fill="{color}"><path d="M19 2H5c-1.1 0-2 .9-2 2v18h18V4c0-1.1-.9-2-2-2zm-8 16H7v-2h4v2zm0-4H7v-2h4v2zm0-4H7V8h4v2zm6 8h-4v-2h4v2zm0-4h-4v-2h4v2zm0-4h-4V8h4v2z"/></svg>''',
        "Apartment": f'''<svg width="34" height="34" viewBox="0 0 24 24" fill="{color}"><path d="M17 19h2V5h-2v14zm-4 0h2V9h-2v10zm-4 0h2V13H9v6zm-4 0h2v-4H5v4zM3 21h18v2H3v-2z"/></svg>''',
        "Flat": f'''<svg width="34" height="34" viewBox="0 0 24 24" fill="{color}"><path d="M4 3h16v18H4V3zm3 3v2h2V6H7zm0 4v2h2v-2H7zm0 4v2h2v-2H7zm6-8v2h4V6h-4zm0 4v2h4v-2h-4zm0 4v2h4v-2h-4z"/></svg>''',
        "Service Residence": f'''<svg width="34" height="34" viewBox="0 0 24 24" fill="{color}"><path d="M12 2l-7 4v16h14V6l-7-4zm5 18H7v-2h10v2zm0-4H7v-2h10v2zm0-4H7v-2h10v2zm-3-5h-4V7h4v2z"/></svg>'''
    }
    
    svg = svg_icons.get(ptype, f'''<svg width="34" height="34" viewBox="0 0 24 24" fill="{color}"><path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/></svg>''')
    
    return f"""
    <div style="border:{border}; background:{bg}; border-radius:12px; padding:10px 4px; text-align:center; height:85px; display:flex; flex-direction:column; justify-content:center; align-items:center;">
        {svg}
        <div style="font-size:0.75rem; font-weight:650; color:{color}; margin-top:5px; line-height:1.1;">{ptype}</div>
    </div>
    """

# ---------------------------------------------------------------------------
# LOGIC CONTROLLERS
# ---------------------------------------------------------------------------
def reset_location_state():
    """Explicitly reset state, area, map view, and the user address input."""
    st.session_state["selected_state"] = None
    st.session_state["selected_area"] = None
    st.session_state["map_center"] = [4.2105, 108.9758]
    st.session_state["map_zoom"] = 6
    st.session_state["address_input"] = ""

def reset_prediction_form():
    """Clear all widgets and address input."""
    for key in list(st.session_state.keys()):
        if key.startswith("pred_") or key in ["selected_state", "selected_area", "map_center", "map_zoom", "address_input", "selected_ptype"]:
            del st.session_state[key]

def known_areas_for_state(data: pd.DataFrame, state: str) -> list[str]:
    subset = data.loc[data["State"] == state, "Area_Clean"].dropna().unique()
    return sorted({display_name(a) for a in subset})

def analyze_address(data, available_states):
    """Fired when user types an address. Detects State & Area, updates map coordinates."""
    addr = st.session_state.get("address_input", "").lower()
    if not addr: return

    matched_state = None
    for st_name in sorted(available_states, key=len, reverse=True):
        if st_name.lower() in addr:
            matched_state = st_name
            st.session_state["selected_state"] = st_name
            break
            
    if matched_state:
        valid_areas = data[data["State"] == matched_state]["Area_Clean"].dropna().unique()
        valid_disp_areas = [display_name(a) for a in valid_areas]
        
        for a in sorted(valid_disp_areas, key=len, reverse=True):
            if a.lower() in addr:
                st.session_state["selected_area"] = a
                coords = get_area_coords(a, matched_state)
                if coords:
                    st.session_state["map_center"] = coords
                    st.session_state["map_zoom"] = 12
                return

    if matched_state and matched_state in STATE_COORDS:
        st.session_state["selected_area"] = None
        st.session_state["map_center"] = STATE_COORDS[matched_state]
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
    
    st.text_input("Enter your address to auto-detect location, or click the map below:", 
                  placeholder="e.g. Sekinchan, Selangor",
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
                
    # Map Mode 2: State Area Drill-Down
    else:
        all_state_areas = data[data["State"] == current_state]["Area_Clean"].dropna().unique()
        marker_cluster = MarkerCluster(name="Areas").add_to(m)
        
        for area_clean in all_state_areas:
            disp_area = display_name(area_clean)
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

    # Location Readout & Controls
    col_loc1, col_loc2 = st.columns([4, 1])
    with col_loc1:
        st.markdown(f"**State:** {current_state or 'Not Selected'} &nbsp; | &nbsp; **Area:** {current_area or 'Not Selected'}")
    with col_loc2:
        if st.button("Reset Location", use_container_width=True):
            reset_location_state()
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
            if st.button("Select", key=f"btn_{pt}", use_container_width=True):
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

    st.markdown('<br>', unsafe_allow_html=True)
    predict_clicked = st.button("Generate Price Estimate  →", type="primary", use_container_width=True)

    # ---------------- RESULT RENDER ----------------
    if predict_clicked:
        if not current_state or not current_area:
            st.error("Please click a State, and then click an Area on the map before predicting.")
            return

        model = load_model(recommended)
        from area_preprocessing import create_area_key
        area_key = create_area_key(current_state, current_area)
        ptype = st.session_state["selected_ptype"]
        
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

FIGURE_GROUPS = {
    "Data quality": [("fig01_raw_target_distribution.png", "House prices are heavily skewed."), ("fig10_outlier_before_after.png", "Extreme values removed by outlier cleaning.")],
    "Area quality and coverage": [("fig14_top20_areas.png", "The 20 areas with the most records in the dataset."), ("fig16_area_price_distribution.png", "How median price varies across different areas.")],
    "Location and property": [("fig18_state_counts_clean.png", "Number of records per state after cleaning."), ("fig19_state_price_distribution.png", "Median price differs a lot from state to state."), ("fig20_property_type_price.png", "Bungalows and semi-detached homes cost more.")],
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
