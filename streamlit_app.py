
"""
BMDS2003 Data Science - Deployment Prototype
Malaysia Housing Median Price Estimator

UPDATED:
- Price Prediction page now has an interactive Malaysia map.
- Click a state marker on the map to automatically select the State.
- Area dropdown automatically updates according to the selected State.
- ML model remains unchanged.
- Market Insights remains unchanged.
- Model Report remains unchanged.

MODEL CONTRACT - six trained features:
    State, Area_Key, Tenure, Primary_Type, Median_PSF, Transactions

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path
import inspect

import joblib
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

from area_preprocessing import (
    clean_area_name,
    clean_state_name,
    create_area_key,
    display_name,
)


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

MODEL_FEATURES = [
    "State",
    "Area_Key",
    "Tenure",
    "Primary_Type",
    "Median_PSF",
    "Transactions",
]

SELECTBOX_ACCEPTS_NEW = (
    "accept_new_options" in inspect.signature(st.selectbox).parameters
)

NO_AREA = "— No specific area —"


# ---------------------------------------------------------------------------
# MALAYSIA STATE MAP LOCATIONS
# ---------------------------------------------------------------------------
# These coordinates are used ONLY for the interactive map.
# They are NOT used as ML model features.

STATE_COORDS = {
    "Johor": (1.4927, 103.7414),
    "Kedah": (6.1184, 100.3685),
    "Kelantan": (6.1254, 102.2381),
    "Melaka": (2.1896, 102.2501),
    "Negeri Sembilan": (2.7258, 101.9424),
    "Pahang": (3.8126, 103.3256),
    "Penang": (5.4141, 100.3288),
    "Perak": (4.5975, 101.0901),
    "Perlis": (6.4449, 100.2048),
    "Sabah": (5.9804, 116.0735),
    "Sarawak": (1.5533, 110.3592),
    "Selangor": (3.0738, 101.5183),
    "Terengganu": (5.3117, 103.1324),
    "Kuala Lumpur": (3.1390, 101.6869),
    "Putrajaya": (2.9264, 101.6964),
    "Labuan": (5.2831, 115.2308),
}


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown(
    r"""
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
    --soft:#EEF4FF;
    --radius:14px;
    --shadow:0 6px 22px rgba(24,49,83,.06);
    --mono:ui-monospace,"SF Mono","Cascadia Mono",Menlo,Consolas,monospace;
}

.stApp {
    background:var(--bg);
    color:var(--text);
}

html,body,[class*="css"] {
    font-family:Inter,"Segoe UI",Roboto,Arial,sans-serif
}

h1,h2,h3,h4 {
    color:var(--navy)
}


/* ---------- TOP NAVIGATION ---------- */

.block-container {
    max-width:1180px;
    padding-top:12px!important;
    padding-bottom:2rem;
}

header[data-testid="stHeader"] {
    background:transparent!important;
}

[data-testid="stToolbar"] {
    color:var(--muted)!important;
}

.stTabs [role="tablist"],
.stTabs [data-baseweb="tab-list"] {
    display:flex!important;
    align-items:center!important;
    gap:8px!important;
    min-height:62px;
    padding:0 210px 0 22px!important;
    background:var(--navy)!important;
    border-radius:var(--radius)!important;
    border-bottom:none!important;
    margin-bottom:22px!important;
    overflow:visible!important;
}

.stTabs [role="tablist"]::before,
.stTabs [data-baseweb="tab-list"]::before {
    content:"\2302\00a0\00a0Malaysia Housing Price Estimator";
    font-size:1.04rem;
    font-weight:700;
    color:#FFFFFF;
    white-space:nowrap;
    margin-right:24px;
}

.stTabs [role="tab"],
.stTabs [data-baseweb="tab"] {
    height:38px!important;
    padding:0 18px!important;
    border-radius:999px!important;
    color:#B9C8DF!important;
    font-weight:600;
    font-size:.92rem;
    background:transparent!important;
    border:none!important;
}

.stTabs [role="tab"]:hover,
.stTabs [data-baseweb="tab"]:hover {
    color:#FFFFFF!important;
    background:rgba(255,255,255,.10)!important;
}

.stTabs [role="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs [aria-selected="true"] {
    color:var(--navy)!important;
    background:#FFFFFF!important;
    border-bottom:none!important;
}

.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"],
.stTabs [role="tablist"] + div[data-baseweb="tab-highlight"] {
    display:none!important;
    background:transparent!important;
}


/* ---------- INPUT PANEL ---------- */

.mh-panel-title {
    display:flex;
    align-items:center;
    gap:9px;
    font-size:1rem;
    font-weight:700;
    color:var(--navy);
    margin:2px 0 4px 0;
}

.mh-panel-sub {
    color:var(--muted);
    font-size:.86rem;
    margin:0 0 14px 0;
    line-height:1.45;
}

.mh-label {
    display:flex;
    align-items:baseline;
    justify-content:space-between;
    font-family:var(--mono);
    font-size:.72rem;
    letter-spacing:.1em;
    color:var(--muted);
    margin:2px 0 5px 0;
    text-transform:uppercase;
}

.mh-label .hint {
    font-family:Inter,sans-serif;
    letter-spacing:0;
    text-transform:none;
    font-size:.78rem;
}

.mh-rule {
    border:none;
    border-top:1px solid var(--border);
    margin:16px 0 14px 0;
}


/* ---------- RESULT ---------- */

.mh-result {
    background:var(--navy);
    border-radius:var(--radius);
    padding:26px 26px 24px;
    box-shadow:var(--shadow);
}

.mh-result .cap {
    font-family:var(--mono);
    font-size:.72rem;
    letter-spacing:.14em;
    color:#9FB4D4;
    text-transform:uppercase;
}

.mh-result .price {
    font-size:2.7rem;
    font-weight:750;
    color:#FFFFFF;
    margin:8px 0 6px;
    line-height:1.05;
    letter-spacing:-.01em;
}

.mh-result .sub {
    color:#C9D7EC;
    font-size:.95rem;
}

.mh-result .rule {
    border-top:1px solid rgba(255,255,255,.16);
    margin:18px 0 14px;
}

.mh-stats {
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:14px;
}

.mh-stats .k {
    font-family:var(--mono);
    font-size:.68rem;
    letter-spacing:.12em;
    color:#9FB4D4;
    text-transform:uppercase;
    margin-bottom:3px;
}

.mh-stats .v {
    font-family:var(--mono);
    font-size:.94rem;
    font-weight:700;
    color:#FFFFFF;
}


/* ---------- USED INPUTS ---------- */

.mh-used {
    background:var(--card);
    border:1px solid var(--border);
    border-radius:var(--radius);
    padding:16px 18px;
    box-shadow:var(--shadow);
    margin-top:12px;
}

.mh-used h4 {
    margin:0 0 10px 0;
    font-size:.94rem;
    font-weight:700;
    color:var(--navy);
}

.mh-used .row {
    display:flex;
    justify-content:space-between;
    align-items:baseline;
    padding:6px 0;
    font-size:.9rem;
}

.mh-used .row + .row {
    border-top:1px dashed var(--border);
}

.mh-used .k {
    color:var(--muted);
}

.mh-used .v {
    font-weight:650;
    color:var(--text);
    font-family:var(--mono);
}


/* ---------- MAP ---------- */

.map-title {
    font-size:1rem;
    font-weight:700;
    color:var(--navy);
    margin-bottom:3px;
}

.map-subtitle {
    color:var(--muted);
    font-size:.86rem;
    margin-bottom:12px;
}

.map-selected {
    background:var(--soft);
    border:1px solid var(--border);
    border-radius:12px;
    padding:10px 13px;
    margin-top:10px;
    font-size:.88rem;
    color:var(--text);
}

.map-selected strong {
    color:var(--navy);
}


/* ---------- BUTTONS ---------- */

.stButton>button {
    min-height:44px;
    border-radius:11px;
    font-weight:650;
}

.stButton>button[kind="primary"] {
    background:var(--blue);
    border-color:var(--blue);
}

.stButton>button[kind="secondary"] {
    background:#FFFFFF;
    color:var(--text);
    border:1px solid var(--border);
}


/* ---------- METRICS ---------- */

div[data-testid="stMetric"] {
    background:white;
    border:1px solid var(--border);
    border-radius:13px;
    padding:13px 15px;
}

.mh-fig-caption {
    font-size:.86rem;
    color:var(--muted);
    margin:2px 0 18px 0;
    line-height:1.4;
}


/* ---------- RESPONSIVE ---------- */

@media(max-width:1400px){

    .stTabs [role="tablist"]::after,
    .stTabs [data-baseweb="tab-list"]::after {
        display:none;
    }

    .stTabs [role="tablist"],
    .stTabs [data-baseweb="tab-list"] {
        padding-right:150px!important;
    }
}

@media(max-width:820px){

    .block-container {
        padding-left:12px;
        padding-right:12px;
    }

    .stTabs [role="tablist"],
    .stTabs [data-baseweb="tab-list"] {
        min-height:56px;
        padding:0 12px!important;
    }

    .stTabs [role="tablist"]::before,
    .stTabs [data-baseweb="tab-list"]::before {
        font-size:.9rem;
        margin-right:12px;
    }

    .stTabs [role="tab"],
    .stTabs [data-baseweb="tab"] {
        padding:0 12px!important;
        font-size:.84rem;
    }

    .mh-result .price {
        font-size:2.05rem;
    }

    .mh-stats {
        grid-template-columns:1fr 1fr;
    }
}

@media(max-width:620px){

    .stTabs [role="tablist"]::before,
    .stTabs [data-baseweb="tab-list"]::before {
        display:none;
    }
}

@media(prefers-reduced-motion:reduce){

    *,*::before,*::after {
        animation-duration:.01ms!important;
        transition-duration:.01ms!important;
    }
}

</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# DATA / MODEL LOADING
# ---------------------------------------------------------------------------

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
    filename = name.split(" (")[0].lower().replace(" ", "_") + ".pkl"
    return joblib.load(MODELS_DIR / filename)


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def field_label(text: str, hint: str = "") -> None:
    right = (
        f'<span class="hint">{hint}</span>'
        if hint
        else ""
    )

    st.markdown(
        f'<div class="mh-label">'
        f'<span>{text}</span>{right}'
        f'</div>',
        unsafe_allow_html=True,
    )


def reset_prediction_form():
    """
    Clear prediction widgets and previous map selection.
    """
    for key in list(st.session_state.keys()):
        if key.startswith("pred_"):
            del st.session_state[key]


# ---------------------------------------------------------------------------
# AREA RESOLUTION
# ---------------------------------------------------------------------------

def known_areas_for_state(
    data: pd.DataFrame,
    state: str
) -> list[str]:

    subset = data.loc[
        data["State"] == state,
        "Area_Clean"
    ].dropna().unique()

    return sorted(
        {
            display_name(a)
            for a in subset
        }
    )


def resolve_known_area(
    data,
    state,
    area_text
):

    clean = clean_area_name(area_text)

    subset = data[
        data["State"] == state
    ]

    match = subset[
        subset["Area_Clean"] == clean
    ]

    return (
        clean if len(match) else None,
        match
    )


def derive_reference(
    data,
    state,
    area_clean,
    ptype,
    tenure
):

    """
    Median PSF / Transactions for the closest matching group.

    INFORMATION ONLY.
    These values are NOT automatically changing the widgets.
    """

    s = data["State"].eq(state)
    t = data["Primary_Type"].eq(ptype)
    n = data["Tenure"].eq(tenure)

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

            return {
                "label": label,
                "psf": int(
                    round(
                        pool["Median_PSF"].median()
                    )
                ),
                "transactions": int(
                    round(
                        pool["Transactions"].median()
                    )
                ),
                "n": len(pool),
                "pool": pool,
            }

    raise ValueError(
        "No reference data available"
    )


# ---------------------------------------------------------------------------
# INTERACTIVE MALAYSIA MAP
# ---------------------------------------------------------------------------

def create_malaysia_map(data):

    """
    Create an interactive Malaysia map.

    Each state with available dataset records receives a clickable marker.

    Clicking a marker returns the state name through st_folium().
    """

    malaysia_map = folium.Map(
        location=[4.2, 109.0],
        zoom_start=5,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    available_states = sorted(
        data["State"].dropna().unique()
    )

    for state in available_states:

        if state not in STATE_COORDS:
            continue

        lat, lon = STATE_COORDS[state]

        area_count = data[
            data["State"] == state
        ]["Area_Clean"].nunique()

        popup_html = f"""
        <div style="font-family:Arial; min-width:160px;">
            <b style="font-size:15px;">{state}</b>
            <br>
            <span style="color:#667085;">
                {area_count:,} areas in dataset
            </span>
            <br><br>
            <span style="color:#2F6FED;">
                Click this marker to select
            </span>
        </div>
        """

        folium.Marker(
            location=[lat, lon],
            tooltip=f"Select {state}",
            popup=folium.Popup(
                popup_html,
                max_width=250,
            ),
            icon=folium.Icon(
                color="blue",
                icon="home",
                prefix="fa",
            ),
        ).add_to(malaysia_map)

    return malaysia_map


# ---------------------------------------------------------------------------
# PAGE 1 - PRICE PREDICTION
# ---------------------------------------------------------------------------

def prediction_page(
    data,
    results
):

    recommended = results.iloc[0]["Model"]

    # Fixed dataset-wide defaults.
    default_psf = int(
        round(
            data["Median_PSF"].median()
        )
    )

    default_txn = int(
        round(
            data["Transactions"].median()
        )
    )

    psf_min = int(
        data["Median_PSF"].min()
    )

    psf_max = int(
        data["Median_PSF"].max()
    )


    # -------------------------------------------------------
    # HANDLE MAP SELECTION
    # -------------------------------------------------------

    # Create initial map selection only once.
    if "pred_map_state" not in st.session_state:
        st.session_state["pred_map_state"] = None


    # -------------------------------------------------------
    # TWO-COLUMN LAYOUT
    # -------------------------------------------------------

    left, right = st.columns(
        [1, 1],
        gap="large"
    )


    # =======================================================
    # LEFT SIDE - INPUTS
    # =======================================================

    with left:

        with st.container(border=True):

            st.markdown(
                '<div class="mh-panel-title">'
                '📍 Location'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="mh-panel-sub">'
                'Select a location from the map or choose it manually.'
                '</div>',
                unsafe_allow_html=True,
            )


            # ------------------------------------------------
            # STATE
            # ------------------------------------------------

            field_label("State")

            states = sorted(
                data["State"].dropna().unique()
            )

            # If map selected a state, use it.
            map_state = st.session_state.get(
                "pred_map_state"
            )

            if (
                map_state in states
                and map_state != st.session_state.get(
                    "pred_state"
                )
            ):

                st.session_state["pred_state"] = map_state


            # Find current index safely.
            current_state = st.session_state.get(
                "pred_state",
                states[0]
            )

            if current_state not in states:
                current_state = states[0]

            state = st.selectbox(
                "State",
                states,
                index=states.index(
                    current_state
                ),
                key="pred_state",
                label_visibility="collapsed",
            )


            # ------------------------------------------------
            # AREA
            # ------------------------------------------------

            field_label("Area")

            area_options = [
                NO_AREA
            ] + known_areas_for_state(
                data,
                state
            )

            # If the state changed, make sure the
            # previous area is still valid.
            current_area = st.session_state.get(
                "pred_area"
            )

            if (
                current_area not in area_options
                and SELECTBOX_ACCEPTS_NEW
            ):
                st.session_state["pred_area"] = (
                    NO_AREA
                )

            if SELECTBOX_ACCEPTS_NEW:

                area_choice = st.selectbox(
                    "Area",
                    area_options,
                    key="pred_area",
                    label_visibility="collapsed",
                    accept_new_options=True,
                    help=(
                        "Choose a known area or type "
                        "an area that is not listed."
                    ),
                )

            else:

                OTHER = "Other (type below)"

                picked = st.selectbox(
                    "Area",
                    area_options + [OTHER],
                    key="pred_area_select",
                    label_visibility="collapsed",
                )

                if picked == OTHER:

                    area_choice = st.text_input(
                        "Type the area",
                        key="pred_area_text",
                    )

                else:

                    area_choice = picked


            area_text = (
                ""
                if area_choice == NO_AREA
                else str(
                    area_choice or ""
                )
            )


            # ------------------------------------------------
            # PROPERTY TYPE
            # ------------------------------------------------

            c1, c2 = st.columns(2)

            with c1:

                field_label(
                    "Property type"
                )

                ptype = st.selectbox(
                    "Property type",
                    sorted(
                        data[
                            "Primary_Type"
                        ].unique()
                    ),
                    key="pred_type",
                    label_visibility="collapsed",
                )


            # ------------------------------------------------
            # TENURE
            # ------------------------------------------------

            with c2:

                field_label(
                    "Tenure"
                )

                tenure = st.selectbox(
                    "Tenure",
                    sorted(
                        data[
                            "Tenure"
                        ].unique()
                    ),
                    key="pred_tenure",
                    label_visibility="collapsed",
                )


            # ------------------------------------------------
            # AREA CHECK
            # ------------------------------------------------

            known_area, _ = resolve_known_area(
                data,
                state,
                area_text
            )

            reference = derive_reference(
                data,
                state,
                known_area,
                ptype,
                tenure
            )


            st.markdown(
                '<hr class="mh-rule">',
                unsafe_allow_html=True,
            )


            # ------------------------------------------------
            # MEDIAN PSF
            # ------------------------------------------------

            field_label(
                "Median price per sq ft (RM)"
            )

            psf = st.number_input(
                "Median price per square foot (RM)",
                min_value=1,
                step=10,
                value=default_psf,
                key="pred_psf",
                label_visibility="collapsed",
            )


            # ------------------------------------------------
            # TRANSACTIONS
            # ------------------------------------------------

            field_label(
                "Transactions"
            )

            transactions = st.number_input(
                "Transactions",
                min_value=0,
                step=1,
                value=default_txn,
                key="pred_txn",
                label_visibility="collapsed",
            )


            # ------------------------------------------------
            # MODEL
            # ------------------------------------------------

            field_label("Model")

            labels = []
            mapping = {}

            for _, row in results.iterrows():

                suffix = (
                    " — Recommended"
                    if row["Model"] == recommended
                    else ""
                )

                label = (
                    row["Model"] + suffix
                )

                labels.append(label)
                mapping[label] = row["Model"]


            picked_model = st.selectbox(
                "Model",
                labels,
                index=0,
                key="pred_model",
                label_visibility="collapsed",
            )

            model_name = mapping[
                picked_model
            ]


            # ------------------------------------------------
            # WARNING
            # ------------------------------------------------

            if (
                psf < psf_min
                or psf > psf_max
            ):

                st.warning(
                    f"RM {psf:,} is outside the observed "
                    f"range (RM {psf_min:,}–"
                    f"{psf_max:,}). Tree-based models "
                    f"do not extrapolate beyond values "
                    f"seen in training, so treat an "
                    f"extreme input as unreliable."
                )


            # ------------------------------------------------
            # BUTTONS
            # ------------------------------------------------

            b1, b2 = st.columns(
                [2, 1]
            )

            predict = b1.button(
                "Predict Price  →",
                type="primary",
                use_container_width=True,
            )

            b2.button(
                "Reset",
                type="secondary",
                use_container_width=True,
                on_click=reset_prediction_form,
                key="reset_prediction",
            )


    # =======================================================
    # RIGHT SIDE - MAP + RESULT
    # =======================================================

    with right:

        # ------------------------------------------------
        # MAP
        # ------------------------------------------------

        st.markdown(
            '<div class="map-title">'
            '🗺️ Select Location'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="map-subtitle">'
            'Click a state marker to automatically select the State.'
            '</div>',
            unsafe_allow_html=True,
        )


        malaysia_map = create_malaysia_map(
            data
        )


        map_result = st_folium(
            malaysia_map,
            width=None,
            height=430,
            key="malaysia_prediction_map",
            returned_objects=[
                "last_object_clicked",
            ],
        )


        # ------------------------------------------------
        # GET CLICKED MARKER
        # ------------------------------------------------

        clicked = (
            map_result.get(
                "last_object_clicked"
            )
            if map_result
            else None
        )


        if clicked:

            clicked_lat = clicked.get(
                "lat"
            )

            clicked_lon = clicked.get(
                "lng"
            )

            # Find nearest state marker.
            nearest_state = None
            nearest_distance = float(
                "inf"
            )

            if (
                clicked_lat is not None
                and clicked_lon is not None
            ):

                for state_name, (
                    state_lat,
                    state_lon,
                ) in STATE_COORDS.items():

                    distance = (
                        (clicked_lat - state_lat) ** 2
                        +
                        (clicked_lon - state_lon) ** 2
                    )

                    if (
                        distance
                        < nearest_distance
                    ):

                        nearest_distance = distance
                        nearest_state = state_name


            # Only accept states existing in dataset.
            if (
                nearest_state in states
            ):

                if (
                    st.session_state.get(
                        "pred_map_state"
                    )
                    != nearest_state
                ):

                    st.session_state[
                        "pred_map_state"
                    ] = nearest_state

                    st.session_state[
                        "pred_state"
                    ] = nearest_state

                    # Rerun so dropdown updates immediately.
                    st.rerun()


        # ------------------------------------------------
        # SHOW SELECTED LOCATION
        # ------------------------------------------------

        selected_state = st.session_state.get(
            "pred_state",
            states[0]
        )

        st.markdown(
            f"""
            <div class="map-selected">
                📍 <strong>Selected State:</strong>
                {selected_state}
            </div>
            """,
            unsafe_allow_html=True,
        )


        # ------------------------------------------------
        # PREDICTION RESULT
        # ------------------------------------------------

        if not predict:

            st.markdown(
                '<div class="mh-empty">'
                '<div class="icon">⌂</div>'
                '<b>Your estimate will appear here.</b>'
                'Complete the inputs on the left, '
                'then select Predict Price.'
                '</div>',
                unsafe_allow_html=True,
            )

            return


        # ------------------------------------------------
        # LOAD MODEL
        # ------------------------------------------------

        model = load_model(
            model_name
        )


        # ------------------------------------------------
        # CREATE AREA KEY
        # ------------------------------------------------

        area_key = create_area_key(
            state,
            area_text
        )


        # ------------------------------------------------
        # MODEL INPUT
        # ------------------------------------------------
        # IMPORTANT:
        # This is exactly the same six-feature
        # model contract as before.

        features = pd.DataFrame(
            [
                {
                    "State": state,
                    "Area_Key": area_key,
                    "Tenure": tenure,
                    "Primary_Type": ptype,
                    "Median_PSF": psf,
                    "Transactions": transactions,
                }
            ]
        )[
            MODEL_FEATURES
        ]


        # ------------------------------------------------
        # PREDICT
        # ------------------------------------------------

        with st.spinner(
            "Calculating estimate..."
        ):

            prediction = float(
                model.predict(
                    features
                )[0]
            )


        # ------------------------------------------------
        # MODEL METRICS
        # ------------------------------------------------

        metrics = results[
            results["Model"] == model_name
        ].iloc[0]


        # ------------------------------------------------
        # CHECK AREA
        # ------------------------------------------------

        try:

            encoder = (
                model
                .named_steps[
                    "preprocess"
                ]
                .named_transformers_[
                    "cat"
                ]
                .named_steps[
                    "encoder"
                ]
            )

            area_position = [
                "State",
                "Area_Key",
                "Tenure",
                "Primary_Type",
            ].index(
                "Area_Key"
            )

            area_seen = (
                area_key
                in set(
                    encoder.categories_[
                        area_position
                    ]
                )
            )

        except Exception:

            area_seen = bool(
                known_area
            )


        # ------------------------------------------------
        # LOCATION TEXT
        # ------------------------------------------------

        location = (
            f"{area_text.strip()}, {state}"
            if area_text.strip()
            else state
        )


        # ------------------------------------------------
        # RESULT CARD
        # ------------------------------------------------

        st.markdown(
            f"""
            <div class="mh-result">

                <div class="cap">
                    Estimated median price
                </div>

                <div class="price">
                    RM {prediction:,.0f}
                </div>

                <div class="sub">
                    {location}
                    · {ptype}
                    · {tenure}
                </div>

                <div class="rule"></div>

                <div class="mh-stats">

                    <div>
                        <div class="k">
                            Model
                        </div>

                        <div class="v">
                            {model_name}
                        </div>
                    </div>


                    <div>
                        <div class="k">
                            Test MAE
                        </div>

                        <div class="v">
                            RM {metrics['MAE_test']/1000:,.1f}K
                        </div>
                    </div>


                    <div>
                        <div class="k">
                            Test R²
                        </div>

                        <div class="v">
                            {metrics['R2_test']:.3f}
                        </div>
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        # ------------------------------------------------
        # INPUTS USED
        # ------------------------------------------------

        st.markdown(
            f"""
            <div class="mh-used">

                <h4>
                    Prediction inputs used
                </h4>

                <div class="row">
                    <span class="k">
                        Median PSF
                    </span>

                    <span class="v">
                        RM {psf:,}
                    </span>
                </div>

                <div class="row">
                    <span class="k">
                        Transactions
                    </span>

                    <span class="v">
                        {transactions:,}
                    </span>
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


        # ------------------------------------------------
        # TECHNICAL DETAILS
        # ------------------------------------------------

        with st.expander(
            "Technical details"
        ):

            st.markdown(
                f"""
                - **Reference source:** {reference['label']}
                - **Reference records:** {reference['n']:,}
                - **Area recognised:** {'Yes' if known_area else 'No'}
                - **Area seen during model training:** {'Yes' if area_seen else 'No'}
                - **Broader state reference values used:** {'No' if known_area else 'Yes'}
                """
            )

            st.caption(
                "These describe where the suggested "
                "typical values came from. The "
                "prediction itself used exactly the "
                "Median PSF and Transactions shown above."
            )


# ---------------------------------------------------------------------------
# PAGE 2 - MARKET INSIGHTS
# ---------------------------------------------------------------------------
# UNCHANGED

FIGURE_GROUPS = {
    "Data quality": [
        (
            "fig01_raw_target_distribution.png",
            "House prices are heavily skewed — most townships are affordable, a few are very expensive.",
        ),
        (
            "fig02_raw_numeric_boxplots.png",
            "Boxplots of price, PSF and transactions before cleaning — dots beyond the whiskers are candidate outliers.",
        ),
        (
            "fig10_outlier_before_after.png",
            "Extreme values removed by outlier cleaning, before vs after.",
        ),
        (
            "fig11_price_distribution_before_after.png",
            "Price distribution becomes more balanced after cleaning.",
        ),
    ],

    "Area quality and coverage": [
        (
            "fig09_area_labels_before_after.png",
            "Area name spelling and formatting before vs after standardisation.",
        ),
        (
            "fig12_area_repeated_singleton.png",
            "Many areas appear only once in the data — a real limitation for those locations.",
        ),
        (
            "fig13_area_frequency_distribution.png",
            "How many records each area has — most have very few.",
        ),
        (
            "fig14_top20_areas.png",
            "The 20 areas with the most records in the dataset.",
        ),
        (
            "fig15_area_cleaning_findings.png",
            "Summary of issues found and fixed while cleaning area names.",
        ),
        (
            "fig16_area_price_distribution.png",
            "How median price varies across different areas.",
        ),
    ],

    "Location and property": [
        (
            "fig18_state_counts_clean.png",
            "Number of records per state after cleaning.",
        ),
        (
            "fig19_state_price_distribution.png",
            "Median price differs a lot from state to state.",
        ),
        (
            "fig20_property_type_price.png",
            "Bungalows and semi-detached homes cost more than flats and apartments, on average.",
        ),
        (
            "fig21_tenure_price.png",
            "Freehold properties tend to have a different price profile than leasehold.",
        ),
    ],

    "Relationships": [
        (
            "fig22_psf_price_by_category.png",
            "Price per square foot is one of the strongest single predictors of price.",
        ),
        (
            "fig23_feature_correlation.png",
            "How strongly each feature relates to price and to the other features.",
        ),
        (
            "fig24_top_transactions.png",
            "The most actively traded townships in 2025.",
        ),
    ],
}


def insights_page(data):

    view = st.radio(
        "View",
        [
            "Market Explorer",
            "Visual Insights",
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="insights_view",
    )


    if view == "Market Explorer":

        st.markdown(
            "#### Historical 2025 dataset exploration"
        )

        a, b, c, d = st.columns(4)


        state = a.selectbox(
            "State",
            ["All"]
            + sorted(
                data["State"].unique()
            ),
            key="ex_state",
        )


        area = b.selectbox(
            "Area",
            ["All"]
            + sorted(
                data["Area_Clean"].unique()
            ),
            key="ex_area",
        )


        ptype = c.selectbox(
            "Property type",
            ["All"]
            + sorted(
                data["Primary_Type"].unique()
            ),
            key="ex_type",
        )


        tenure = d.selectbox(
            "Tenure",
            ["All"]
            + sorted(
                data["Tenure"].unique()
            ),
            key="ex_tenure",
        )


        subset = data.copy()


        if state != "All":

            subset = subset[
                subset["State"] == state
            ]


        if area != "All":

            subset = subset[
                subset["Area_Clean"] == area
            ]


        if ptype != "All":

            subset = subset[
                subset["Primary_Type"] == ptype
            ]


        if tenure != "All":

            subset = subset[
                subset["Tenure"] == tenure
            ]


        if len(subset) == 0:

            st.warning(
                "No historical records match these filters."
            )

        else:

            m1, m2, m3, m4 = st.columns(4)


            m1.metric(
                "Records",
                f"{len(subset):,}"
            )


            m2.metric(
                "Median price",
                f"RM {subset['Median_Price'].median()/1000:,.0f}K"
            )


            m3.metric(
                "Median PSF",
                f"RM {subset['Median_PSF'].median():,.0f}"
            )


            m4.metric(
                "Median transactions",
                f"{subset['Transactions'].median():,.0f}"
            )


            with st.expander(
                "View matching historical records"
            ):

                show = subset[
                    [
                        "Township",
                        "Area_Clean",
                        "State",
                        "Primary_Type",
                        "Tenure",
                        "Median_Price",
                        "Median_PSF",
                        "Transactions",
                    ]
                ].copy()

                st.dataframe(
                    show,
                    use_container_width=True,
                    hide_index=True,
                )


    else:

        group = st.selectbox(
            "Insight category",
            list(FIGURE_GROUPS)
        )


        for filename, caption in FIGURE_GROUPS[group]:

            path = FIGURES_DIR / filename

            if path.exists():

                st.image(
                    str(path),
                    use_container_width=True,
                )

                st.markdown(
                    f'<p class="mh-fig-caption">'
                    f'{caption}'
                    f'</p>',
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# PAGE 3 - MODEL REPORT
# ---------------------------------------------------------------------------
# UNCHANGED

def model_report_page(results):

    recommended = results.iloc[0]

    st.subheader(
        "Model Report"
    )


    m1, m2, m3, m4 = st.columns(4)


    m1.metric(
        "Selected model",
        recommended["Model"]
    )


    m2.metric(
        "Group CV RMSE",
        f"RM {recommended['Group_CV_RMSE_mean']/1000:,.1f}K"
    )


    m3.metric(
        "Test MAE",
        f"RM {recommended['MAE_test']/1000:,.1f}K"
    )


    m4.metric(
        "Test R²",
        f"{recommended['R2_test']:.3f}"
    )


    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True,
    )


    sections = {

        "Area ablation": [
            (
                "fig25_area_ablation.png",
                "Does including Area actually improve predictions? Compared with and without it.",
            )
        ],

        "Performance": [
            (
                "fig26_model_test_comparison.png",
                "How the four models compare on data they have not seen.",
            ),
            (
                "fig27_cv_stability_overfitting.png",
                "Checking that each model performs consistently, not just well on one lucky split.",
            ),
        ],

        "Diagnostics": [
            (
                "fig28_prediction_diagnostics.png",
                "Where the selected model's predictions are most and least accurate.",
            )
        ],

        "Importance": [
            (
                "fig29_permutation_importance.png",
                "Which inputs the model actually relies on, tested on unseen data.",
            ),
            (
                "fig30_aggregated_split_importance.png",
                "Which inputs the model used most often while learning.",
            ),
        ],
    }


    section = st.selectbox(
        "Report section",
        list(sections)
    )


    for filename, caption in sections[section]:

        path = FIGURES_DIR / filename

        if path.exists():

            st.image(
                str(path),
                use_container_width=True,
            )

            st.markdown(
                f'<p class="mh-fig-caption">'
                f'{caption}'
                f'</p>',
                unsafe_allow_html=True,
            )


    with st.expander(
        "Key limitations"
    ):

        st.markdown(
            "- Some Areas contain very few records.\n"
            "- Completely unseen Areas are harder than previously observed Areas.\n"
            "- Median PSF remains required market information.\n"
            "- The dataset is a static 2025 snapshot.\n"
            "- Results are township-level medians, not individual-property valuations."
        )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():

    missing = [
        p.name
        for p in [
            DATA_PATH,
            RESULTS_PATH,
        ]
        if not p.exists()
    ]


    if missing:

        st.error(
            "Missing required files: "
            + ", ".join(missing)
        )

        st.stop()


    data = load_data()
    results = load_results()


    pred, insights, report = st.tabs(
        [
            "Price Prediction",
            "Market Insights",
            "Model Report",
        ]
    )


    with pred:

        prediction_page(
            data,
            results
        )


    with insights:

        insights_page(
            data
        )


    with report:

        model_report_page(
            results
        )


main()
```
