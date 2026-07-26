"""
BMDS2003 Data Science - Deployment Prototype
Malaysia Housing Median Price Estimator (market-assisted)

Run locally:      streamlit run streamlit_app.py
Deploy:           Streamlit Community Cloud (all paths are relative)

The app loads the trained pipeline selected by cross-validation RMSE
(training set only) and estimates a township-level median house price.
It does NOT retrain or modify any model.
"""

from __future__ import annotations

from pathlib import Path
import base64

import joblib
import pandas as pd
import streamlit as st

# Plotly is optional: the app degrades gracefully to a Streamlit bar chart.
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Malaysia Housing Median Price Estimator",
    page_icon="🏘️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Relative paths so the app runs identically on GitHub / Streamlit Cloud
RESULTS_PATH = Path("model_results.csv")
DATA_PATH = Path("malaysia_house_price_cleaned.csv")
MODELS_DIR = Path("models")
BANNER_PATH = Path("assets/malaysia_housing_banner.png")

# Palette (fixed by the design brief)
BG = "#0B1220"
CARD = "rgba(30, 41, 59, 0.88)"
PRIMARY = "#1E88E5"
ACCENT = "#FF4B4B"
SUCCESS = "#22C55E"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"

# Feature names expected by the trained pipeline - DO NOT CHANGE
MODEL_FEATURES = ["State", "Tenure", "Primary_Type", "Median_PSF", "Transactions"]


# ---------------------------------------------------------------------------
# STYLING
# ---------------------------------------------------------------------------
def inject_css() -> None:
    """Apply the dark-navy theme, translucent cards and responsive rules."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(160deg, {BG} 0%, #101A2E 55%, #0D1526 100%);
            color: {TEXT};
        }}
        .block-container {{ padding-top: 1.6rem; max-width: 1180px; }}

        /* Cards */
        .mh-card {{
            background: {CARD};
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            padding: 1.35rem 1.5rem;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
            margin-bottom: 1.1rem;
        }}
        .mh-card h3 {{
            margin: 0 0 0.9rem 0;
            font-size: 1.02rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            color: {TEXT};
        }}

        /* Hero */
        .mh-hero {{
            position: relative;
            border-radius: 18px;
            padding: 2.6rem 2rem;
            margin-bottom: 1.4rem;
            border: 1px solid rgba(148, 163, 184, 0.2);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45);
            background-size: cover;
            background-position: center;
        }}
        .mh-eyebrow {{
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: {TEXT};
            background: rgba(30, 136, 229, 0.85);
            padding: 0.32rem 0.7rem;
            border-radius: 999px;
            margin-bottom: 0.9rem;
        }}
        .mh-hero h1 {{
            margin: 0 0 0.5rem 0;
            font-size: 2.05rem;
            font-weight: 700;
            line-height: 1.18;
            color: #FFFFFF;
        }}
        .mh-hero p {{
            margin: 0;
            max-width: 640px;
            font-size: 0.98rem;
            line-height: 1.55;
            color: #E2E8F0;
        }}

        /* Metric cards */
        div[data-testid="stMetric"] {{
            background: {CARD};
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 14px;
            padding: 0.95rem 1.1rem;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3);
        }}
        div[data-testid="stMetricLabel"] p {{
            color: {MUTED} !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}
        div[data-testid="stMetricValue"] {{
            color: {TEXT} !important;
            font-size: 1.42rem !important;
        }}

        /* Result card */
        .mh-result {{
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.16),
                                        rgba(30, 136, 229, 0.14));
            border: 1px solid rgba(34, 197, 94, 0.45);
            border-radius: 16px;
            padding: 1.6rem 1.7rem;
            box-shadow: 0 10px 26px rgba(0, 0, 0, 0.35);
        }}
        .mh-result .label {{
            font-size: 0.8rem; letter-spacing: 0.09em; text-transform: uppercase;
            color: {MUTED}; margin-bottom: 0.35rem;
        }}
        .mh-result .value {{
            font-size: 2.3rem; font-weight: 700; color: {SUCCESS}; line-height: 1.1;
        }}
        .mh-result .sub {{ margin-top: 0.75rem; color: {TEXT}; font-size: 0.94rem; }}
        .mh-result .note {{ margin-top: 0.5rem; color: {MUTED}; font-size: 0.82rem;
                            line-height: 1.5; }}

        /* Inline status banners */
        .mh-note {{ border-radius: 12px; padding: 0.8rem 1rem; font-size: 0.88rem;
                    line-height: 1.5; margin-top: 0.4rem; }}
        .mh-note.ok    {{ background: rgba(30, 136, 229, 0.14);
                          border: 1px solid rgba(30, 136, 229, 0.45); color: #DBEAFE; }}
        .mh-note.warn  {{ background: rgba(255, 75, 75, 0.10);
                          border: 1px solid rgba(255, 75, 75, 0.40); color: #FEE2E2; }}
        .mh-note.alert {{ background: rgba(255, 75, 75, 0.20);
                          border: 1px solid {ACCENT}; color: #FFE4E6; }}

        .mh-source {{ color: {MUTED}; font-size: 0.8rem; line-height: 1.5;
                      margin: 0.2rem 0 0.9rem 0; }}

        /* Footer */
        .mh-footer {{
            margin-top: 2.2rem; padding-top: 1rem;
            border-top: 1px solid rgba(148, 163, 184, 0.2);
            text-align: center; color: {MUTED}; font-size: 0.8rem;
        }}

        /* Native bordered containers used as cards (widgets render inside) */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: rgba(30, 41, 59, 0.88);
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    padding: 0.4rem 0.2rem;
}}

        /* Buttons */
        .stButton > button {{ border-radius: 10px; font-weight: 600; }}

        /* Sidebar */
        section[data-testid="stSidebar"] > div {{
            background: #0D1526;
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }}

        /* Mobile */
        @media (max-width: 640px) {{
            .mh-hero {{ padding: 1.8rem 1.2rem; }}
            .mh-hero h1 {{ font-size: 1.5rem; }}
            .mh-result .value {{ font-size: 1.75rem; }}
            .block-container {{ padding-left: 0.8rem; padding-right: 0.8rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def encode_image(path_str: str) -> str | None:
    """Return a base64 string for a local image, or None if it is missing."""
    path = Path(path_str)
    if not path.exists() or not path.is_file():
        return None
    try:
        return base64.b64encode(path.read_bytes()).decode()
    except OSError:
        return None


def render_hero() -> None:
    """Hero banner. Uses the local image if present, otherwise a gradient."""
    encoded = encode_image(str(BANNER_PATH))
    if encoded:
        layer = (
            f"linear-gradient(rgba(11, 18, 32, 0.78), rgba(11, 18, 32, 0.86)), "
            f"url('data:image/png;base64,{encoded}')"
        )
    else:
        # Clean gradient fallback - the app never crashes on a missing file
        layer = ("linear-gradient(120deg, #0B1220 0%, #14243F 45%, "
                 "#1E3A5F 100%)")

    st.markdown(
        f"""
        <div class="mh-hero" style="background-image: {layer};">
            <span class="mh-eyebrow">BMDS2003 Data Science Group Assignment</span>
            <h1>Malaysia Housing Median Price Estimator</h1>
            <p>A market-assisted township-level housing price prediction prototype
               using Malaysian housing data.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# DATA & MODEL LOADING
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the selected model...")
def load_best_model():
    """Load the pipeline chosen by cross-validation RMSE (training set only).

    The test set is never used for selection - only for reporting final
    performance. Selection order: lowest CV_RMSE_mean, then lowest CV_RMSE_std.
    """
    results = pd.read_csv(RESULTS_PATH)
    best = results.sort_values(["CV_RMSE_mean", "CV_RMSE_std"],
                               ascending=[True, True]).iloc[0]
    filename = best["Model"].split(" (")[0].lower().replace(" ", "_") + ".pkl"
    model_path = MODELS_DIR / filename
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))
    return (joblib.load(model_path), str(best["Model"]),
            float(best["MAE_test"]), float(best["R2_test"]))


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load the cleaned 2025 township dataset."""
    return pd.read_csv(DATA_PATH)


def load_resources():
    """Load data and model with friendly errors instead of a traceback."""
    missing = [str(p) for p in (RESULTS_PATH, DATA_PATH) if not p.exists()]
    if missing:
        st.error(
            "Required file(s) not found: " + ", ".join(missing) + ".  \n"
            "Keep `streamlit_app.py`, `model_results.csv`, "
            "`malaysia_house_price_cleaned.csv` and the `models/` folder "
            "together in the same directory, then reload the app."
        )
        st.stop()
    try:
        model, name, mae, r2 = load_best_model()
        data = load_data()
    except FileNotFoundError as exc:
        st.error(
            f"Model file not found: `{exc}`.  \n"
            "Unzip the `models/` folder next to `streamlit_app.py`, so that the "
            "`.pkl` files sit inside it."
        )
        st.stop()
    except Exception:
        st.error(
            "The model or dataset could not be loaded. Check that "
            "`model_results.csv` and the `.pkl` files come from the same "
            "notebook run, and that the installed library versions match "
            "`requirements.txt`."
        )
        st.stop()
    return model, name, mae, r2, data


# ---------------------------------------------------------------------------
# ANALYSIS HELPERS (pure functions - easy to test)
# ---------------------------------------------------------------------------
def psf_reference(data: pd.DataFrame, state: str) -> dict:
    """Median / min / max observed PSF for one state."""
    psf_series = data.loc[data["State"] == state, "Median_PSF"]
    return {
        "median": float(psf_series.median()),
        "min": float(psf_series.min()),
        "max": float(psf_series.max()),
        "n": int(psf_series.size),
    }


def psf_status(psf: float, reference: dict, state: str) -> tuple[str, str]:
    """Classify the chosen PSF against the state's observed values.

    Returns (level, message) where level is "ok", "warn" or "alert".
    """
    median, low, high = reference["median"], reference["min"], reference["max"]
    deviation = (psf - median) / median if median else 0.0

    if psf < low or psf > high:
        return "alert", (
            f"The selected PSF of RM{psf:,.0f} is outside the range observed in "
            f"{state} (RM{low:,.0f} – RM{high:,.0f}). The model is "
            f"extrapolating beyond its training data, so treat the estimate "
            f"with strong caution."
        )
    if abs(deviation) > 0.30:
        direction = "above" if deviation > 0 else "below"
        segment = ("premium-market" if deviation > 0 else "budget-market")
        return "warn", (
            f"The selected PSF of RM{psf:,.0f} is substantially {direction} the "
            f"{state} median of RM{median:,.0f}. The prediction may represent a "
            f"{segment} property."
        )
    return "ok", (
        f"The selected PSF of RM{psf:,.0f} is within 30% of the {state} median "
        f"of RM{median:,.0f}, which is the range the model has seen most often."
    )


def find_similar_records(data: pd.DataFrame, state: str, ptype: str,
                         psf: float, transactions: int,
                         top_n: int = 5) -> pd.DataFrame:
    """Rank dataset records by similarity to the user's inputs.

    Matching pool : same State AND same Primary_Type.
    Similarity    : absolute % difference in Median_PSF
                    + normalised absolute difference in Transactions.
    Lower score = more similar.
    """
    pool = data[(data["State"] == state) &
                (data["Primary_Type"] == ptype)].copy()
    if pool.empty:
        return pool

    psf_gap = (pool["Median_PSF"] - psf).abs() / max(psf, 1)

    span = pool["Transactions"].max() - pool["Transactions"].min()
    span = span if span > 0 else 1
    txn_gap = (pool["Transactions"] - transactions).abs() / span

    pool["Similarity_Score"] = psf_gap + txn_gap
    return pool.sort_values("Similarity_Score").head(top_n)


def summarise_against_group(prediction: float, group_median: float) -> str:
    """Wording that compares the prediction with the matching-group median."""
    difference = prediction - group_median
    percent = (difference / group_median * 100) if group_median else 0.0

    if abs(percent) < 5:
        relation = "close to the matching-group median"
    elif difference > 0:
        relation = "above the matching-group median"
    else:
        relation = "below the matching-group median"

    return (
        f"The estimate of **RM {prediction:,.0f}** is {relation} "
        f"(**RM {group_median:,.0f}**) — a difference of "
        f"**RM {abs(difference):,.0f}** (**{abs(percent):.1f}%**)."
    )


# ---------------------------------------------------------------------------
# CHART
# ---------------------------------------------------------------------------
def render_comparison_chart(prediction: float, group_median: float,
                            similar: pd.DataFrame) -> None:
    """Bar chart: prediction vs group median vs the five closest records."""
    labels = ["Your estimate", "Matching-group median"]
    values = [prediction, group_median]
    colors = [SUCCESS, PRIMARY]

    for _, row in similar.iterrows():
        name = str(row["Township"]).title()
        labels.append(name if len(name) <= 22 else name[:20] + "…")
        values.append(float(row["Median_Price"]))
        colors.append("#64748B")

    if PLOTLY_AVAILABLE:
        figure = go.Figure(
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f"RM {v:,.0f}" for v in values],
                textposition="outside",
                hovertemplate="%{x}<br>RM %{y:,.0f}<extra></extra>",
            )
        )
        figure.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=30, b=90),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT, size=12),
            yaxis=dict(title="Median price (RM)", gridcolor="rgba(148,163,184,0.18)",
                       tickformat=",.0f"),
            xaxis=dict(tickangle=-30),
            showlegend=False,
        )
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.bar_chart(pd.DataFrame({"Median price (RM)": values}, index=labels))
        st.caption("Install `plotly` for the styled version of this chart.")


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
def render_sidebar(model_name: str, record_count: int) -> None:
    with st.sidebar:
        st.markdown("### About this prototype")
        st.write(
            "Estimates the **median house price of a township** from its state, "
            "tenure, property type, market activity and known median price per "
            "square foot."
        )
        st.markdown("---")
        st.markdown(
            f"""
            **Dataset year** &nbsp;· 2025
            **Records after cleaning** &nbsp;· {record_count:,}
            **Prediction level** &nbsp;· Township-level median
            **Model in use** &nbsp;· {model_name}
            """
        )
        st.markdown("---")
        st.markdown("**Main limitation**")
        st.write(
            "The median price per square foot must already be known for the "
            "area. The model is market-assisted: it refines a known market "
            "rate rather than discovering prices without any market input."
        )
        st.markdown("---")
        st.caption(
            "Academic prototype for BMDS2025 coursework. Built on a static "
            "2025 dataset, not live market data. Estimates are township-level "
            "medians, not valuations of individual properties, and must not be "
            "used for real financial decisions."
        )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    inject_css()
    model, model_name, test_mae, test_r2, data = load_resources()

    render_sidebar(model_name, len(data))
    render_hero()

    # --- C. Model performance cards -------------------------------------
    card_1, card_2, card_3 = st.columns(3)
    card_1.metric("Selected model", model_name)
    card_2.metric("Test R²", f"{test_r2:.3f}")
    card_3.metric("Typical test MAE", f"RM {test_mae:,.0f}")
    st.caption(
        "The model was selected by cross-validation RMSE on the training set. "
        "The test figures above describe its performance on unseen data."
    )

    # Reset support: bumping the counter gives every widget a fresh key,
    # which restores all default values.
    if "reset_counter" not in st.session_state:
        st.session_state.reset_counter = 0
    suffix = st.session_state.reset_counter

    # --- D. Inputs -------------------------------------------------------
    st.markdown("#### Property and market information")
    input_card = st.container(border=True)
    with input_card:
        left, right = st.columns(2)
        with left:
            state = st.selectbox("State", sorted(data["State"].unique()),
                                 key=f"state_{suffix}")
            tenure = st.selectbox("Tenure", sorted(data["Tenure"].unique()),
                                  key=f"tenure_{suffix}")
        with right:
            ptype = st.selectbox("Property type",
                                 sorted(data["Primary_Type"].unique()),
                                 key=f"ptype_{suffix}")
            transactions = st.slider(
                "Transactions in the township", 10,
                int(data["Transactions"].max()), 20,
                key=f"txn_{suffix}",
                help="Recorded market activity: how many transactions the township "
                     "registered in the dataset year. Use a low value for a quiet "
                     "area and a high value for an active one.",
            )

        reference = psf_reference(data, state)
        psf_left, psf_right = st.columns([3, 1])
        with psf_left:
            # Keying on the state resets the slider default when the state changes
            psf = st.slider(
                "Median price per square foot (RM)",
                int(data["Median_PSF"].min()), int(data["Median_PSF"].max()),
                int(reference["median"]),
                key=f"psf_{state}_{suffix}",
                help="The market rate per square foot for the selected area. Look "
                     "this up from recent listings or a property portal.",
            )
        with psf_right:
            st.metric(f"{state} median PSF", f"RM {reference['median']:,.0f}")

        # --- E. PSF validation ----------------------------------------------
        level, message = psf_status(psf, reference, state)
        st.markdown(f'<div class="mh-note {level}">{message}</div>',
                    unsafe_allow_html=True)
    action_left, action_right = st.columns([3, 1])
    with action_left:
        predict = st.button("Estimate median price", type="primary",
                            use_container_width=True)
    with action_right:
        if st.button("Reset inputs", use_container_width=True):
            st.session_state.reset_counter += 1
            st.rerun()

    if not predict:
        return

    # --- F. Prediction ---------------------------------------------------
    features = pd.DataFrame([{
        "State": state,
        "Tenure": tenure,
        "Primary_Type": ptype,
        "Median_PSF": psf,
        "Transactions": transactions,
    }])[MODEL_FEATURES]

    try:
        prediction = float(model.predict(features)[0])
    except Exception:
        st.error(
            "The model could not produce a prediction for these inputs. This "
            "usually means the saved pipeline expects different columns. "
            "Re-export the models from the notebook and try again."
        )
        return

    st.markdown(
        f"""
        <div class="mh-result">
            <div class="label">Estimated median price</div>
            <div class="value">RM {prediction:,.0f}</div>
            <div class="sub">Typical absolute test error (MAE):
                <strong>RM {test_mae:,.0f}</strong></div>
            <div class="note">MAE is the model's average absolute error on the
                test dataset. It is not a confidence interval or statistically
                valid prediction interval.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- G/H/I. Reference records ---------------------------------------
    similar = find_similar_records(data, state, ptype, psf, transactions)

    if similar.empty:
        st.markdown(
            '<div class="mh-note warn">No record in the dataset combines this '
            'state with this property type, so no comparable townships can be '
            'shown. The estimate is an extrapolation — treat it with extra '
            'caution.</div>',
            unsafe_allow_html=True,
        )
        return

    pool = data[(data["State"] == state) & (data["Primary_Type"] == ptype)]
    group_median = float(pool["Median_Price"].median())

    st.markdown("")
    st.markdown("#### Reference from similar 2025 township records")
    ref_card = st.container(border=True)
    with ref_card:
        st.markdown(
        '<p class="mh-source">Source: cleaned Malaysia housing dataset, 2025. '
        'These are historical dataset records and are not live property-market '
        'data.</p>',
            unsafe_allow_html=True,
        )

        # H. Comparison summary
        st.markdown(summarise_against_group(prediction, group_median))
        st.caption(
            f"Matching group: {len(pool)} record(s) in {state} of type {ptype}. "
            "The group median summarises past dataset records; it is not a current "
            "market price."
        )

        # I. Chart
        render_comparison_chart(prediction, group_median, similar)

        # G. Table of the five closest records
        st.markdown("**Five most similar records** (same state and property type, "
                    "closest on price per square foot and market activity)")
        table = (similar[["Township", "Area", "Median_Price",
                          "Median_PSF", "Transactions"]]
                 .reset_index(drop=True))
        # Pre-format as strings so RM values render with separators on every
        # Streamlit version (avoids version-specific column_config formats).
        display_table = pd.DataFrame({
            "Township": table["Township"].str.title(),
            "Area": table["Area"],
            "Median Price (RM)": table["Median_Price"].map(lambda v: f"{v:,.0f}"),
            "Median PSF (RM)": table["Median_PSF"].map(lambda v: f"{v:,.0f}"),
            "Transactions": table["Transactions"],
        })
        st.dataframe(display_table, use_container_width=True, hide_index=True)

        with st.expander("How similarity is calculated"):
            st.write(
                "Records are first filtered to the same state and property type. "
                "Each remaining record is then scored as the absolute percentage "
                "difference in median price per square foot plus the absolute "
                "difference in transactions, normalised by the range within the "
                "matching group. A lower score means a closer match."
            )
            st.dataframe(
                similar[["Township", "Median_PSF", "Transactions",
                         "Similarity_Score"]]
                .rename(columns={"Median_PSF": "Median PSF (RM)",
                                 "Similarity_Score": "Similarity score"})
                .round({"Similarity score": 3})
                .reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )


main()

# --- K. Footer ----------------------------------------------------------
st.markdown(
    '<div class="mh-footer">BMDS2003 Data Science Group Assignment | '
    'Academic Prototype | Data source: Malaysia Housing Prices 2025</div>',
    unsafe_allow_html=True,
)
