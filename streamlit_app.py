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
MAP_DATA_PATH = APP_DIR / "malaysia_house_price_data_2025.csv"
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


# ---------------------------------------------------------------------------
# MAP COVERAGE
# ---------------------------------------------------------------------------
# Nationwide map coverage combines three layers:
# 1) every State -> Area pair in the uploaded 2025 housing dataset,
# 2) all 160 DOSM administrative-district units across Malaysia, and
# 3) useful official/local subdivisions for the four DOSM state-level units
#    that are not split into lower administrative districts in that dataset.
#
# The prediction model still receives the selected location directly. Its
# OneHotEncoder was trained with handle_unknown="infrequent_if_exist", so
# official locations absent from the housing data remain valid inputs but are
# identified in the result as out-of-dataset area estimates.

OFFICIAL_DISTRICTS = {'Johor': ['Batu Pahat',
           'Johor Bahru',
           'Kluang',
           'Kota Tinggi',
           'Kulai',
           'Mersing',
           'Muar',
           'Pontian',
           'Segamat',
           'Tangkak'],
 'Kedah': ['Baling',
           'Bandar Baharu',
           'Kota Setar',
           'Kuala Muda',
           'Kubang Pasu',
           'Kulim',
           'Langkawi',
           'Padang Terap',
           'Pendang',
           'Pokok Sena',
           'Sik',
           'Yan'],
 'Kelantan': ['Bachok',
              'Kota Bharu',
              'Machang',
              'Pasir Mas',
              'Pasir Puteh',
              'Tanah Merah',
              'Tumpat',
              'Gua Musang',
              'Kuala Krai',
              'Jeli',
              'Kecil Lojing'],
 'Melaka': ['Alor Gajah', 'Jasin', 'Melaka Tengah'],
 'Negeri Sembilan': ['Jelebu', 'Jempol', 'Kuala Pilah', 'Port Dickson', 'Rembau', 'Seremban', 'Tampin'],
 'Pahang': ['Bentong',
            'Bera',
            'Cameron Highlands',
            'Jerantut',
            'Kuantan',
            'Lipis',
            'Maran',
            'Pekan',
            'Raub',
            'Rompin',
            'Temerloh'],
 'Penang': ['Barat Daya',
            'Seberang Perai Selatan',
            'Seberang Perai Tengah',
            'Seberang Perai Utara',
            'Timur Laut'],
 'Perak': ['Batang Padang',
           'Manjung',
           'Kinta',
           'Kerian',
           'Kuala Kangsar',
           'Larut & Matang',
           'Hilir Perak',
           'Hulu Perak',
           'Perak Tengah',
           'Kampar',
           'Muallim',
           'Bagan Datuk',
           'Selama'],
 'Perlis': ['Perlis'],
 'Sabah': ['Tawau',
           'Lahad Datu',
           'Semporna',
           'Sandakan',
           'Kinabatangan',
           'Beluran',
           'Kota Kinabalu',
           'Ranau',
           'Kota Belud',
           'Tuaran',
           'Penampang',
           'Papar',
           'Kudat',
           'Kota Marudu',
           'Pitas',
           'Beaufort',
           'Kuala Penyu',
           'Sipitang',
           'Tenom',
           'Nabawan',
           'Keningau',
           'Tambunan',
           'Kunak',
           'Tongod',
           'Putatan',
           'Telupid',
           'Kalabakan'],
 'Sarawak': ['Kuching',
             'Bau',
             'Lundu',
             'Samarahan',
             'Serian',
             'Simunjan',
             'Sri Aman',
             'Lubok Antu',
             'Betong',
             'Saratok',
             'Sarikei',
             'Maradong',
             'Daro',
             'Julau',
             'Sibu',
             'Dalat',
             'Mukah',
             'Kanowit',
             'Bintulu',
             'Tatau',
             'Kapit',
             'Song',
             'Belaga',
             'Miri',
             'Marudi',
             'Limbang',
             'Lawas',
             'Matu',
             'Asajaya',
             'Pakan',
             'Selangau',
             'Tebedu',
             'Pusa',
             'Kabong',
             'Tanjung Manis',
             'Sebauh',
             'Bukit Mabong',
             'Subis',
             'Beluru',
             'Telang Usan'],
 'Selangor': ['Gombak',
              'Hulu Langat',
              'Hulu Selangor',
              'Klang',
              'Kuala Langat',
              'Kuala Selangor',
              'Petaling',
              'Sabak Bernam',
              'Sepang'],
 'Terengganu': ['Besut',
                'Dungun',
                'Hulu Terengganu',
                'Kemaman',
                'Kuala Nerus',
                'Kuala Terengganu',
                'Marang',
                'Setiu'],
 'Kuala Lumpur': ['Kuala Lumpur'],
 'Labuan': ['Labuan'],
 'Putrajaya': ['Putrajaya']}

assert len(OFFICIAL_DISTRICTS) == 16
assert sum(len(v) for v in OFFICIAL_DISTRICTS.values()) == 160

SPECIAL_DISPLAY_AREAS = {'Putrajaya': ['Precinct 1',
               'Precinct 2',
               'Precinct 3',
               'Precinct 4',
               'Precinct 5',
               'Precinct 6',
               'Precinct 7',
               'Precinct 8',
               'Precinct 9',
               'Precinct 10',
               'Precinct 11',
               'Precinct 12',
               'Precinct 13',
               'Precinct 14',
               'Precinct 15',
               'Precinct 16',
               'Precinct 17',
               'Precinct 18',
               'Precinct 19',
               'Precinct 20'],
 'Perlis': ['Titi Tinggi',
            'Beseri',
            'Chuping',
            'Paya',
            'Padang Siding',
            'Abi',
            'Padang Pauh',
            'Ngulang',
            'Oran',
            'Kurong Batang',
            'Arau',
            'Kechor',
            'Sena',
            'Sungai Adam',
            'Kurong Anai',
            'Jejawi',
            'Kuala Perlis',
            'Wang Bintong',
            'Seriab',
            'Kayang',
            'Utan Aji',
            'Sanglang'],
 'Kuala Lumpur': ['City Centre',
                  'Wangsa Maju - Maluri',
                  'Bukit Jalil - Seputeh',
                  'Bandar Tun Razak - Sungai Besi',
                  'Sentul - Menjalara',
                  'Damansara - Penchala'],
 'Labuan': ['Batu Arang',
            'Batu Manikar',
            'Bebuloh',
            'Belukut',
            'Bukit Kallam',
            'Bukit Kuda',
            'Durian Tunjong',
            'Ganggarak',
            'Gersik/Saguking',
            'Kerupang/Nagalang',
            'Kilan/Pulau Akar',
            'Lajau',
            'Layang-Layangan',
            'Lubuk Temiang',
            'Pantai',
            'Patau-Patau 1',
            'Patau-Patau 2',
            'Pohon Batu',
            'Rancha-Rancha',
            'Sungai Bangat',
            'Sungai Bedaun',
            'Sungai Buton',
            'Sungai Keling',
            'Sungai Labu',
            'Sungai Lada',
            'Sungai Miri',
            'Tanjung Aru']}

SPECIAL_AREA_LABELS = {'Putrajaya': 'Precinct', 'Perlis': 'Mukim', 'Kuala Lumpur': 'Strategic zone', 'Labuan': 'Village area'}

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
:root {
    --ink:#1E293B;
    --muted:#667085;
    --line:#DDE5E1;
    --surface:#FFFFFF;
    --page:#F5F8F6;
    --forest:#18392F;
    --forest-2:#234C3F;
    --forest-3:#2C5B4B;
    --green:#2F8F68;
    --green-dark:#247653;
    --green-soft:#EAF7F0;
    --green-pale:#F4FBF7;
    --amber:#C68A35;
    --mono:ui-monospace,"SF Mono",monospace;
    --shadow:0 12px 30px rgba(24,57,47,.10);
}

.stApp { background:var(--page); color:var(--ink); }
.block-container { max-width:1240px; padding-top:14px!important; padding-bottom:2.2rem; }
header[data-testid="stHeader"] { background:transparent!important; }
div[data-testid="stToolbar"] { display:none!important; }
#MainMenu { visibility:hidden; }
footer { visibility:hidden; }

/* ---------- HEADER / NAVIGATION ---------- */
.stTabs [role="tablist"] {
    display:flex!important;
    align-items:center!important;
    gap:12px!important;
    min-height:104px;
    padding:18px 24px!important;
    background:linear-gradient(135deg, var(--forest) 0%, var(--forest-2) 72%, #315E4F 100%)!important;
    border:1px solid rgba(255,255,255,.08)!important;
    border-radius:26px!important;
    margin:10px 0 30px!important;
    box-shadow:0 18px 38px rgba(22,52,43,.18);
}
.stTabs [role="tablist"]::before {
    content:"⌂  Malaysia\AHousing Estimator";
    white-space:pre;
    display:block;
    min-width:270px;
    padding:8px 12px;
    line-height:1.18;
    font-size:1.18rem;
    font-weight:800;
    letter-spacing:.01em;
    color:#FFFFFF;
    margin-right:14px;
}
.stTabs [role="tab"] {
    height:50px!important;
    padding:0 28px!important;
    border-radius:15px!important;
    color:#D2E1DB!important;
    font-weight:700!important;
    font-size:.98rem!important;
    background:rgba(255,255,255,.035)!important;
    border:1px solid transparent!important;
    transition:all .18s ease!important;
}
.stTabs [role="tab"]:hover {
    color:#FFFFFF!important;
    background:rgba(255,255,255,.09)!important;
}
.stTabs [role="tab"][aria-selected="true"] {
    color:var(--forest)!important;
    background:#FFFFFF!important;
    border-color:rgba(255,255,255,.7)!important;
    box-shadow:0 9px 20px rgba(12,35,28,.18), inset 0 -4px 0 var(--green)!important;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none!important; }

/* ---------- SECTION HEADERS ---------- */
.mh-section-head {
    display:flex;
    align-items:flex-start;
    gap:13px;
    margin:2px 0 16px;
}
.mh-step {
    width:38px;
    height:38px;
    flex:0 0 38px;
    border-radius:12px;
    background:var(--green-soft);
    color:var(--green-dark);
    display:flex;
    align-items:center;
    justify-content:center;
    font-family:var(--mono);
    font-weight:800;
    border:1px solid #CBEADB;
}
.mh-section-title { margin:0!important; color:#17231F; font-size:1.38rem; line-height:1.2; }
.mh-section-note { color:var(--muted); font-size:.94rem; margin:.3rem 0 0; }
.mh-label { font-family:var(--mono); font-size:.75rem; letter-spacing:.16em; color:#66776F; margin:6px 0 8px; text-transform:uppercase; }
.mh-rule { border:none; border-top:1px solid var(--line); margin:26px 0 22px; }

/* ---------- MAP / LOCATION ---------- */
.mh-map-wrap {
    border:1px solid var(--line);
    border-radius:20px;
    overflow:hidden;
    box-shadow:0 10px 26px rgba(27,66,53,.08);
    background:#FFFFFF;
}
.mh-chiprow { display:flex; flex-wrap:wrap; gap:10px; margin-top:10px; }
.mh-chip {
    display:inline-flex;
    align-items:center;
    gap:8px;
    background:#FFFFFF;
    border:1px solid #D7E3DD;
    padding:10px 15px;
    min-height:48px;
    border-radius:14px;
    color:#405148;
    font-size:.95rem;
    box-shadow:0 3px 10px rgba(27,66,53,.04);
}
.mh-chip strong { color:#173A2F; }

/* ---------- PROPERTY TYPE ---------- */
div[class*="st-key-btn_"] button[kind="secondary"],
div[class*="st-key-btn_"] button[kind="primary"] {
    min-height:82px!important;
    border-radius:17px!important;
    white-space:pre-line!important;
    line-height:1.24!important;
    padding:9px 5px!important;
    font-weight:650!important;
    font-size:.76rem!important;
    text-align:center!important;
    transition:all .16s ease!important;
}
div[class*="st-key-btn_"] button[kind="secondary"] {
    background:#FFFFFF!important;
    color:#27352F!important;
    border:1px solid #D7E3DD!important;
    box-shadow:0 5px 14px rgba(27,66,53,.05)!important;
}
div[class*="st-key-btn_"] button[kind="secondary"]:hover {
    background:var(--green-pale)!important;
    border-color:#9ED3BB!important;
    color:var(--green-dark)!important;
    transform:translateY(-1px);
}
div[class*="st-key-btn_"] button[kind="primary"] {
    background:linear-gradient(180deg, #3B9B75 0%, var(--green) 100%)!important;
    color:#FFFFFF!important;
    border:1px solid var(--green)!important;
    box-shadow:0 9px 18px rgba(47,143,104,.20)!important;
}
div[class*="st-key-btn_"] button p {
    margin:0!important;
    white-space:pre-line!important;
    overflow-wrap:anywhere!important;
}

/* ---------- TENURE PILLS ---------- */
div[data-testid="stVerticalBlock"]:has(.tenure-anchor) div[role="radiogroup"] {
    display:flex!important;
    flex-wrap:wrap!important;
    gap:10px!important;
}
div[data-testid="stVerticalBlock"]:has(.tenure-anchor) div[role="radiogroup"] > label {
    background:#FFFFFF!important;
    border:1px solid #D7E3DD!important;
    border-radius:999px!important;
    min-height:48px!important;
    padding:9px 15px!important;
    display:flex!important;
    align-items:center!important;
    box-shadow:0 4px 12px rgba(27,66,53,.04)!important;
}
div[data-testid="stVerticalBlock"]:has(.tenure-anchor) div[role="radiogroup"] > label:has(input:checked) {
    border-color:var(--green)!important;
    background:var(--green-soft)!important;
    box-shadow:0 6px 16px rgba(47,143,104,.14)!important;
}
div[data-testid="stVerticalBlock"]:has(.tenure-anchor) div[role="radiogroup"] p {
    font-weight:650!important;
    color:#33443C!important;
}
div[data-testid="stVerticalBlock"]:has(.tenure-anchor) div[role="radiogroup"] > label:has(input:checked) p {
    color:var(--green-dark)!important;
}
input[type="radio"] { accent-color:var(--green)!important; }

/* ---------- INPUTS / BUTTONS ---------- */
div[data-testid="stNumberInputContainer"], div[data-baseweb="input"] {
    background:#FFFFFF;
    border:1px solid #D7E3DD;
    border-radius:14px;
}
.stButton>button[kind="primary"] {
    background:linear-gradient(180deg, #399973 0%, var(--green) 100%);
    border-color:var(--green);
    border-radius:14px;
    min-height:50px;
    font-weight:750;
    box-shadow:0 10px 20px rgba(47,143,104,.18);
}
.stButton>button[kind="primary"]:hover {
    background:var(--green-dark);
    border-color:var(--green-dark);
}
button[kind="secondary"] { border-radius:14px!important; }

/* ---------- PREDICTION RESULT ---------- */
.mh-result {
    position:relative;
    overflow:hidden;
    background:linear-gradient(135deg, #15372D 0%, #214A3D 64%, #2B5A49 100%);
    border-radius:26px;
    padding:30px 32px;
    box-shadow:0 18px 42px rgba(22,70,52,.18);
    margin-top:22px;
    border:1px solid rgba(255,255,255,.08);
}
.mh-result::after {
    content:"";
    position:absolute;
    width:260px;
    height:260px;
    right:-110px;
    top:-120px;
    border-radius:50%;
    background:rgba(114,207,166,.10);
    pointer-events:none;
}
.mh-result-top {
    position:relative;
    z-index:1;
    display:flex;
    justify-content:space-between;
    align-items:flex-start;
    gap:20px;
}
.mh-result .cap {
    font-family:var(--mono);
    font-size:.75rem;
    letter-spacing:.18em;
    color:#B8D8CB;
    text-transform:uppercase;
}
.mh-result .price {
    font-size:3.7rem;
    line-height:1.02;
    font-weight:820;
    color:#FFFFFF;
    margin:12px 0 10px;
}
.mh-result .sub { color:#D8EAE2; font-size:1.02rem; }
.mh-result .note { margin-top:9px; color:#BFD8CD; font-size:.86rem; line-height:1.45; }
.mh-result-badge {
    min-width:220px;
    background:rgba(255,255,255,.09);
    border:1px solid rgba(255,255,255,.12);
    border-radius:18px;
    padding:15px 17px;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.05);
}
.mh-result-badge .kicker {
    font-family:var(--mono);
    font-size:.66rem;
    letter-spacing:.13em;
    color:#B8D8CB;
    text-transform:uppercase;
}
.mh-result-badge .model { color:#FFFFFF; font-size:1.16rem; font-weight:800; margin-top:6px; }
.mh-result .rule { border-top:1px solid rgba(255,255,255,.14); margin:22px 0 17px; }
.mh-stats { position:relative; z-index:1; display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }
.mh-stat {
    background:rgba(255,255,255,.065);
    border:1px solid rgba(255,255,255,.09);
    border-radius:16px;
    padding:15px 17px;
}
.mh-stats .k {
    font-family:var(--mono);
    font-size:.66rem;
    letter-spacing:.12em;
    color:#B8D8CB;
    text-transform:uppercase;
    margin-bottom:6px;
}
.mh-stats .v { font-family:var(--mono); font-size:1.04rem; font-weight:800; color:#FFFFFF; }
.mh-empty {
    background:#FFFFFF;
    border:1px dashed #BCD9CC;
    border-radius:20px;
    padding:42px 24px;
    text-align:center;
    color:#687A72;
    box-shadow:0 5px 18px rgba(27,66,53,.05);
    margin-top:20px;
}
.mh-empty .icon { font-size:1.7rem; color:var(--green); margin-bottom:7px; }

/* ---------- FOOTER ---------- */
.mh-footer {
    margin-top:36px;
    padding:18px 22px;
    border-radius:20px;
    background:linear-gradient(135deg, var(--forest) 0%, var(--forest-2) 72%, #315E4F 100%);
    color:#DCEAE4;
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:18px;
    box-shadow:0 13px 30px rgba(22,52,43,.14);
    border:1px solid rgba(255,255,255,.07);
    border-top:3px solid var(--green);
}
.mh-footer .brand { color:#FFFFFF; font-weight:800; letter-spacing:.01em; }
.mh-footer .sub { color:#BFD2CA; font-size:.84rem; }

@media (max-width:900px) {
    .stTabs [role="tablist"] { flex-wrap:wrap!important; }
    .stTabs [role="tablist"]::before { width:100%; min-width:0; }
    .mh-stats { grid-template-columns:1fr; }
    .mh-chip { width:100%; justify-content:center; }
    .mh-result-top { flex-direction:column; }
    .mh-result-badge { width:100%; min-width:auto; }
    .mh-footer { flex-direction:column; align-items:flex-start; }
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
def load_map_data():
    raw = pd.read_csv(MAP_DATA_PATH, usecols=["State", "Area"])
    raw["State"] = raw["State"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    raw["Area"] = raw["Area"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    raw = raw.dropna(subset=["State", "Area"]).drop_duplicates(["State", "Area"])
    return raw.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def load_results():
    return pd.read_csv(RESULTS_PATH).sort_values(["Group_CV_RMSE_mean"]).reset_index(drop=True)

@st.cache_resource(show_spinner=False)
def load_model(name):
    filename = name.split(" (")[0].lower().replace(" ", "_") + ".pkl"
    return joblib.load(MODELS_DIR / filename)

@st.cache_data(show_spinner=False)
def get_area_coords(area_name: str, state_name: str):
    """Return a real known/geocoded coordinate, or None if it cannot be verified."""
    if state_name in HARDCODED_AREAS and area_name in HARDCODED_AREAS[state_name]:
        return HARDCODED_AREAS[state_name][area_name]

    if HAS_GEOPY:
        try:
            geolocator = Nominatim(user_agent="mh_estimator", timeout=4)
            for query in (
                f"{area_name}, {state_name}, Malaysia",
                f"{area_name} District, {state_name}, Malaysia",
            ):
                loc = geolocator.geocode(query, country_codes="my")
                if loc:
                    return [loc.latitude, loc.longitude]
        except Exception:
            pass

    # Perlis and Federal Territories have no lower DOSM administrative subdivision.
    if area_name == state_name:
        return STATE_COORDS.get(state_name)
    return None

@st.cache_data(show_spinner=False)
def get_area_map_coords(area_name: str, state_name: str):
    """Always return a map pin while preserving whether its position is verified."""
    real = get_area_coords(area_name, state_name)
    if real:
        return real, False

    # If live geocoding is unavailable, keep the area selectable but mark the pin
    # as approximate rather than pretending the fallback coordinate is exact.
    base = STATE_COORDS.get(state_name)
    if not base:
        return None, True
    h = int(hashlib.md5(f"{state_name}|{area_name}".encode("utf-8")).hexdigest(), 16)
    lat_offset = (((h % 1000) / 999) - 0.5) * 0.55
    lon_offset = ((((h // 1000) % 1000) / 999) - 0.5) * 0.75
    return [base[0] + lat_offset, base[1] + lon_offset], True

def field_label(text: str) -> None:
    st.markdown(f'<div class="mh-label">{text}</div>', unsafe_allow_html=True)

def get_property_icon(ptype: str) -> str:
    ptype_icons = {
        "Bungalow": "🏠", "Semi D": "🏘️", "Cluster House": "🏡",
        "Terrace House": "🏘️", "Town House": "🏠", "Condominium": "🏢",
        "Service Residence": "🏙️", "Apartment": "🏢", "Flat": "🏬",
    }
    return ptype_icons.get(ptype, "🏠")

def get_property_label(ptype: str) -> str:
    return f"{get_property_icon(ptype)}\n{ptype}"

@st.cache_data(show_spinner=False)
def get_dataset_areas_for_state(map_data: pd.DataFrame, state_name: str):
    return sorted(
        map_data.loc[map_data["State"] == state_name, "Area"]
        .dropna()
        .astype(str)
        .unique()
    )

@st.cache_data(show_spinner=False)
def get_areas_for_state(map_data: pd.DataFrame, state_name: str):
    areas = set(get_dataset_areas_for_state(map_data, state_name))
    areas.update(OFFICIAL_DISTRICTS.get(state_name, []))
    areas.update(SPECIAL_DISPLAY_AREAS.get(state_name, []))
    return sorted(areas)

def get_area_source(map_data: pd.DataFrame, state_name: str, area_name: str) -> str:
    dataset_areas = set(get_dataset_areas_for_state(map_data, state_name))
    in_dataset = area_name in dataset_areas
    in_districts = area_name in set(OFFICIAL_DISTRICTS.get(state_name, []))
    in_special = area_name in set(SPECIAL_DISPLAY_AREAS.get(state_name, []))

    if in_dataset and in_districts:
        return "Dataset locality · Official district"
    if in_dataset:
        return "Housing dataset locality"
    if in_districts:
        return "Official administrative district"
    if in_special:
        return SPECIAL_AREA_LABELS.get(state_name, "Official local area")
    return "Malaysia map area"

def is_dataset_area(map_data: pd.DataFrame, state_name: str, area_name: str) -> bool:
    return area_name in set(get_dataset_areas_for_state(map_data, state_name))

@st.cache_data(show_spinner=False)
def map_coverage_summary(map_data: pd.DataFrame):
    dataset_pairs = map_data[["State", "Area"]].drop_duplicates()
    nationwide_pairs = []
    for state_name in STATE_COORDS:
        for area_name in get_areas_for_state(map_data, state_name):
            nationwide_pairs.append((state_name, area_name))
    return {
        "states": sorted(STATE_COORDS),
        "state_count": len(STATE_COORDS),
        "dataset_pairs": int(len(dataset_pairs)),
        "official_district_units": int(sum(len(v) for v in OFFICIAL_DISTRICTS.values())),
        "nationwide_state_area_pairs": int(len(set(nationwide_pairs))),
    }

# ---------------------------------------------------------------------------
# LOGIC CONTROLLERS
# ---------------------------------------------------------------------------
def reset_location_state():
    st.session_state["selected_state"] = None
    st.session_state["selected_area"] = None
    st.session_state["map_center"] = [4.2105, 108.9758]
    st.session_state["map_zoom"] = 6
    st.session_state["address_input"] = "" 

def analyze_address(map_data, available_states):
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
                if raw_area and matched_state:
                    valid_areas = get_areas_for_state(map_data, matched_state)
                    normalized = {clean_area_name(a): a for a in valid_areas}
                    matched_area = normalized.get(clean_area_name(raw_area))
        except: pass

    if not matched_state:
        for st_name in sorted(available_states, key=len, reverse=True):
            if st_name.lower() in addr:
                matched_state = st_name
                break
                
    if matched_state and not matched_area:
        valid_areas = get_areas_for_state(map_data, matched_state)
        for a in sorted(valid_areas, key=len, reverse=True):
            if a.lower() in addr:
                matched_area = a
                break

    if matched_state:
        st.session_state["selected_state"] = matched_state
        if matched_area:
            st.session_state["selected_area"] = matched_area
            coords, _ = get_area_map_coords(matched_area, matched_state)
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
def prediction_page(data, results, map_data):
    recommended = results.iloc[0]["Model"]
    available_states = sorted(STATE_COORDS.keys())
    ptypes = sorted(data["Primary_Type"].unique())

    if "selected_state" not in st.session_state: st.session_state["selected_state"] = None
    if "selected_area" not in st.session_state: st.session_state["selected_area"] = None
    if "selected_ptype" not in st.session_state: st.session_state["selected_ptype"] = ptypes[0]
    if "address_input" not in st.session_state: st.session_state["address_input"] = ""

    current_state = st.session_state["selected_state"]
    current_area = st.session_state["selected_area"]
    
    st.markdown(
        "<div class='mh-section-head'><div class='mh-step'>1</div><div>"
        "<h3 class='mh-section-title'>Location Selection</h3>"
        "<p class='mh-section-note'>Choose a state, then select an area from the nationwide Malaysia coverage.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    st.text_input("Enter your address or postcode to auto-detect location, or click the map below:", 
                  placeholder="e.g. 45400 or Sekinchan, Selangor",
                  key="address_input", on_change=analyze_address, args=(map_data, available_states))

    map_center = st.session_state.get("map_center", [4.2105, 108.9758] if not current_state else STATE_COORDS.get(current_state, [4.2105, 108.9758]))
    map_zoom = st.session_state.get("map_zoom", 6 if not current_state else 9)
    
    m = folium.Map(location=map_center, zoom_start=map_zoom, tiles="OpenStreetMap", control_scale=True, zoom_control=True, prefer_canvas=True)

    if not current_state:
        for st_name in available_states:
            if st_name in STATE_COORDS:
                folium.CircleMarker(
                    location=STATE_COORDS[st_name], radius=11, color="#18392F", weight=2,
                    fill=True, fill_color="#3E8169", fill_opacity=0.88,
                    tooltip=folium.Tooltip(f"<b>{st_name}</b> (Click to select state)", sticky=True),
                    popup=f"STATE:{st_name}"
                ).add_to(m)
                
    else:
        # Nationwide coverage: dataset localities + official districts + special local subdivisions.
        areas_to_plot = set(get_areas_for_state(map_data, current_state))

        if current_area:
            areas_to_plot.add(current_area)
            
        marker_cluster = MarkerCluster(name="Areas").add_to(m)
        
        for disp_area in sorted(areas_to_plot):
            coords, is_approx = get_area_map_coords(disp_area, current_state)
            if coords:
                is_sel = (disp_area == current_area)
                area_kind = get_area_source(map_data, current_state, disp_area)
                pin_note = "Approximate position · click to select" if is_approx else "Click to select"
                folium.CircleMarker(
                    location=coords, radius=12 if is_sel else 8,
                    color="#247653" if is_sel else ("#98A2B3" if is_approx else "#A86F24"),
                    weight=3 if is_sel else 2,
                    fill=True,
                    fill_color="#2F8F68" if is_sel else ("#D0D5DD" if is_approx else "#D9A04B"),
                    fill_opacity=0.9 if is_sel else 0.75,
                    tooltip=folium.Tooltip(f"<b>{disp_area}</b><br>{area_kind} · {pin_note}", sticky=True),
                    popup=f"AREA:{disp_area}"
                ).add_to(marker_cluster)

    st.markdown("<div class='mh-map-wrap'>", unsafe_allow_html=True)
    map_event = st_folium(m, height=470, use_container_width=True, key="malaysia_map")
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:.82rem;color:#667085;margin-top:6px;'>"
        "<b>Map pins:</b> 🟢 State &nbsp; 🟠 Verified area &nbsp; ⚪ Approximate position &nbsp; ✅ Selected area"
        "</div>",
        unsafe_allow_html=True,
    )
    
    if map_event and map_event.get("last_object_clicked_popup"):
        popup_txt = map_event["last_object_clicked_popup"]
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
            coords, _ = get_area_map_coords(clicked_area, current_state)
            if coords:
                st.session_state["map_center"] = coords
            st.session_state["map_zoom"] = 12
            st.rerun()

    col_loc1, col_loc2 = st.columns([5, 1.25])
    with col_loc1:
        st.markdown(f"<div class='mh-chiprow'><span class='mh-chip'>📍 <strong>State:</strong> {current_state or 'Not Selected'}</span><span class='mh-chip'>🗺️ <strong>Area:</strong> {current_area or 'Not Selected'}</span></div>", unsafe_allow_html=True)
    with col_loc2:
        st.button("Reset Location", use_container_width=True, on_click=reset_location_state)

    st.markdown('<hr class="mh-rule">', unsafe_allow_html=True)
    st.markdown(
        "<div class='mh-section-head'><div class='mh-step'>2</div><div>"
        "<h3 class='mh-section-title'>Property Details</h3>"
        "<p class='mh-section-note'>Choose the property type and tenure, then enter the independently known median price per square foot.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    # ---------------- DIRECT-CLICK PROPERTY TYPE CARDS ----------------
    # One horizontal row; selection logic remains the same.
    field_label("Select Property Type")
    ptype_cols = st.columns(len(ptypes), gap="small")

    for i, pt in enumerate(ptypes):
        with ptype_cols[i]:
            is_sel = (pt == st.session_state["selected_ptype"])
            if st.button(
                get_property_label(pt),
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
        st.markdown("<div class='tenure-anchor'></div>", unsafe_allow_html=True)
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
        dataset_supported = is_dataset_area(map_data, current_state, current_area)
        coverage_note = "" if dataset_supported else (
            '<div class="note">This location is part of the nationwide Malaysia map but is not directly present in the 2025 housing dataset. '
            'The model therefore uses its unseen / infrequent-area handling for this estimate.</div>'
        )
        st.markdown(f'''
        <div class="mh-result" id="estimate-result">
            <div class="mh-result-top">
                <div>
                    <div class="cap">Estimated median price</div>
                    <div class="price">RM {prediction:,.0f}</div>
                    <div class="sub">{current_area}, {current_state} · {ptype} · {tenure}</div>
                    {coverage_note}
                </div>
                <div class="mh-result-badge">
                    <div class="kicker">Recommended model</div>
                    <div class="model">{recommended}</div>
                </div>
            </div>
            <div class="rule"></div>
            <div class="mh-stats">
                <div class="mh-stat"><div class="k">Test MAE</div><div class="v">RM {metrics['MAE_test']/1000:,.1f}K</div></div>
                <div class="mh-stat"><div class="k">Test R²</div><div class="v">{metrics['R2_test']:.3f}</div></div>
                <div class="mh-stat"><div class="k">Market PSF Input</div><div class="v">RM {psf:,.0f}</div></div>
            </div>
        </div>''', unsafe_allow_html=True)

    else:
        st.markdown(
            '<div class="mh-empty"><div class="icon">⌂</div>'
            '<b>Your estimate will appear here.</b><br>'
            '<span>Select location and property details above, then generate the estimate.</span></div>',
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
    missing = [p.name for p in [DATA_PATH, MAP_DATA_PATH, RESULTS_PATH] if not p.exists()]
    if missing:
        st.error("Missing required files: " + ", ".join(missing)); st.stop()
    data = load_data(); map_data = load_map_data(); results = load_results()
    coverage = map_coverage_summary(map_data)
    missing_state_coords = sorted(set(coverage["states"]) - set(STATE_COORDS))
    if missing_state_coords:
        st.error("Missing map coordinates for: " + ", ".join(missing_state_coords)); st.stop()

    pred, insights, report = st.tabs(["Price Prediction", "Market Insights", "Model Report"])
    with pred: prediction_page(data, results, map_data)
    with insights: insights_page(data)
    with report: model_report_page(results)

    st.markdown(
        "<div class='mh-footer'>"
        "<div><div class='brand'>⌂ Malaysia Housing Estimator</div>"
        "<div class='sub'>Nationwide Malaysia map · 2025 housing-market price estimation</div></div>"
        "<div class='sub'>BMDS2003 · Data Science Prototype</div>"
        "</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
