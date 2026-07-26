"""
BMDS2003 Data Science - Deployment Prototype
Malaysia Housing Median Price Estimator (market-assisted)
Run locally:  streamlit run streamlit_app.py
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Malaysia House Price Estimator",
                   page_icon="house", layout="centered")

st.title("Malaysia Housing Median Price Estimator")
st.caption("BMDS2003 Data Science - Group Assignment Deployment Prototype")


@st.cache_resource
def load_best_model():
    """Load the model selected by cross-validation RMSE (training set only).

    The test set is NOT used for selection - only for reporting final performance.
    """
    res = pd.read_csv("model_results.csv")
    best = res.sort_values(["CV_RMSE_mean", "CV_RMSE_std"],
                           ascending=[True, True]).iloc[0]
    fname = best["Model"].split(" (")[0].lower().replace(" ", "_") + ".pkl"
    return (joblib.load(f"models/{fname}"), best["Model"],
            float(best["MAE_test"]), float(best["R2_test"]))


@st.cache_data
def load_data():
    return pd.read_csv("malaysia_house_price_cleaned.csv")


model, best_name, TEST_MAE, TEST_R2 = load_best_model()
df = load_data()

st.info(f"Model in use: **{best_name}**  |  Test R2 = {TEST_R2:.2f}  |  "
        f"Typical error (MAE) = RM {TEST_MAE:,.0f}")

st.subheader("1. Property details")
col1, col2 = st.columns(2)
with col1:
    state = st.selectbox("State", sorted(df["State"].unique()))
    tenure = st.selectbox("Tenure", sorted(df["Tenure"].unique()))
with col2:
    ptype = st.selectbox("Property type", sorted(df["Primary_Type"].unique()))
    transactions = st.slider("Transactions in township (market activity)",
                             10, int(df["Transactions"].max()), 20)

psf_default = int(df[df["State"] == state]["Median_PSF"].median())
psf = st.slider("Median price per sq ft (RM) - market rate for the area",
                int(df["Median_PSF"].min()), int(df["Median_PSF"].max()), psf_default,
                help="Required input. Look this up from recent listings or a property "
                     "portal for the township you are pricing.")
st.caption(f"Median PSF across {state}: RM {psf_default:,}")

st.subheader("2. Estimate")
if st.button("Estimate median price", type="primary", use_container_width=True):
    row = pd.DataFrame([{ "State": state, "Tenure": tenure, "Primary_Type": ptype,
                          "Median_PSF": psf, "Transactions": transactions }])
    pred = float(model.predict(row)[0])

    st.success(f"### Estimated median price: RM {pred:,.0f}")
    st.write(f"Typical absolute test error (MAE): **RM {TEST_MAE:,.0f}**")
    st.caption("MAE is the model's average absolute error on the test data. "
               "It is not a confidence interval or a statistically valid "
               "prediction interval.")

    similar = df[(df["State"] == state) & (df["Primary_Type"] == ptype)]
    if len(similar) > 0:
        st.subheader("3. Comparison with real townships")
        st.write(f"{len(similar)} township(s) in **{state}** of type **{ptype}** - "
                 f"actual median price: RM {similar['Median_Price'].median():,.0f}")
        st.dataframe(similar.nlargest(5, "Transactions")
                     [["Township", "Area", "Median_Price", "Median_PSF", "Transactions"]]
                     .reset_index(drop=True), use_container_width=True)
    else:
        st.warning("No township in the data matches this State + Type combination. "
                   "The estimate is an extrapolation - treat it with extra caution.")

st.divider()
st.caption(
    "Academic prototype. Trained on 1,914 Malaysian township records (2025) after "
    "removing statistical outliers, so estimates apply to the mainstream market and are "
    "less reliable for very low-cost or luxury properties. Because median PSF is closely "
    "related to total price, accuracy depends on supplying a realistic PSF value. "
    "Predictions are township-level medians, not valuations of individual properties, "
    "and must not be used for real financial decisions.")
