import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import extract_features_v4  # Required for FunctionTransformer deserialization

st.set_page_config(
    page_title="Used Car Fair-Market Valuation Engine",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Used Car Fair-Market Valuation Engine")
st.markdown(
    "Estimate fair-market vehicle prices with **Duan's Smearing bias correction**, "
    "**Conformalized 80% Prediction Intervals (`CQR`)**, a **Seller Deal Grader**, "
    "and a **5-Year Depreciation Simulator**."
)

BUNDLE_PATH = "models/used_car_valuation_bundle.joblib"


@st.cache_resource
def load_valuation_bundle(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


bundle = load_valuation_bundle(BUNDLE_PATH)
if bundle is None:
    st.error(
        f"Model bundle not found at `{BUNDLE_PATH}`. "
        "Run `python -m src.train --data-path data/raw/used_cars.csv` first, "
        "then launch with `streamlit run app.py`."
    )
    st.stop()
    raise SystemExit("Halting execution in bare mode.")

point_pipe = bundle["point_pipeline"]
q10_model = bundle["q10_model"]
q90_model = bundle["q90_model"]
smearing_factor = bundle.get("smearing_factor", 1.0755)
cqr_offset = bundle.get("cqr_offset", 0.0174)
catalog = bundle.get("catalog", {})

brands_list = catalog.get(
    "brands",
    ["Lexus", "Toyota", "Ford", "BMW", "Mercedes-Benz", "Audi", "Porsche", "Honda", "Chevrolet"]
)
brand_to_models = catalog.get("brand_to_models", {})
model_defaults = catalog.get("model_defaults", {})

# --- Section 1: Cascading Vehicle Specification Inputs ---
st.subheader("1. Configure Vehicle Specifications")
col1, col2, col3 = st.columns(3)

with col1:
    default_brand_idx = brands_list.index("Lexus") if "Lexus" in brands_list else 0
    brand = st.selectbox("Brand", brands_list, index=default_brand_idx)

    available_models = brand_to_models.get(brand, ["Base"])
    default_model_idx = available_models.index("RX 350 Base") if "RX 350 Base" in available_models else 0
    model = st.selectbox("Model & Trim", available_models, index=default_model_idx)

    model_year = st.slider("Model Year", min_value=1990, max_value=2024, value=2018, step=1)
    milage = st.number_input("Odometer Mileage (miles)", min_value=50, max_value=300000, value=48000, step=1000)

# Auto-populate typical engine, transmission & fuel type for the selected (brand, model)
specs = model_defaults.get(f"{brand}||{model}", {})
default_engine = specs.get("engine", "295.0HP 3.5L V6 Cylinder Engine Gasoline Fuel")
default_trans = specs.get("transmission", "8-Speed A/T")
default_fuel = specs.get("fuel_type", "Gasoline")

with col2:
    engine = st.text_input("Engine Specification", value=str(default_engine))
    trans_options = [
        "8-Speed A/T", "6-Speed A/T", "10-Speed A/T", "7-Speed A/T", "9-Speed A/T",
        "A/T", "Transmission w/Dual Shift Mode", "6-Speed M/T", "CVT Transmission",
        "Automatic", "1-Speed A/T"
    ]
    if default_trans not in trans_options:
        trans_options.insert(0, str(default_trans))
    transmission = st.selectbox("Transmission", trans_options, index=trans_options.index(str(default_trans)))

    fuel_options = ["Gasoline", "Hybrid", "Diesel", "E85 Flex Fuel", "Plug-In Hybrid", "Missing"]
    fuel_idx = fuel_options.index(str(default_fuel)) if str(default_fuel) in fuel_options else 0
    fuel_type = st.selectbox("Fuel Type", fuel_options, index=fuel_idx)

with col3:
    accident = st.selectbox(
        "Accident / Damage History",
        ["None reported", "At least 1 accident or damage reported", "Missing"],
        index=0
    )
    clean_title_choice = st.selectbox(
        "Clean Title Status",
        ["Yes", "Not Reported / Missing"],
        index=0
    )
    ext_col = st.selectbox("Exterior Color", ["Black", "White", "Gray", "Silver", "Blue", "Red", "Green", "Brown"])
    int_col = st.selectbox("Interior Color", ["Black", "Beige", "Gray", "Brown", "Red", "White"])

asking_price = st.number_input(
    "Optional: Seller's Asking Price ($) — Enter to grade deal fairness",
    min_value=0, max_value=500000, value=0, step=500
)

st.divider()

# --- Section 2: Inference, Deal Grader & 5-Year Depreciation ---
input_df = pd.DataFrame([{
    "brand": brand,
    "model": model,
    "model_year": int(model_year),
    "milage": int(milage),
    "fuel_type": np.nan if fuel_type == "Missing" else fuel_type,
    "engine": engine,
    "transmission": transmission,
    "ext_col": ext_col,
    "int_col": int_col,
    "accident": np.nan if accident == "Missing" else accident,
    "clean_title": "Yes" if clean_title_choice == "Yes" else np.nan
}])

pred_log = float(point_pipe.predict(input_df)[0])
pred_median = float(np.expm1(pred_log))
pred_mean_smeared = float(np.exp(pred_log) * smearing_factor - 1)

X_trans = point_pipe[:-1].transform(input_df)
raw_q10 = float(q10_model.predict(X_trans)[0])
raw_q90 = float(q90_model.predict(X_trans)[0])
cqr_low = float(np.expm1(min(raw_q10, raw_q90) - cqr_offset))
cqr_high = float(np.expm1(max(raw_q10, raw_q90) + cqr_offset))

st.subheader("2. Fair-Market Valuation & Conformal Uncertainty Bounds")
m1, m2, m3 = st.columns(3)
m1.metric(
    "Expected Market Value (Duan Smeared)",
    f"${pred_mean_smeared:,.0f}",
    help="Expected dollar value corrected for log-normal retransformation bias."
)
m2.metric(
    "Typical Median Listing (Geometric Mean)",
    f"${pred_median:,.0f}",
    help="Conditional median estimate minimizing Mean Absolute Error."
)
m3.metric(
    "80% Conformal Fair-Market Range",
    f"${cqr_low:,.0f} – ${cqr_high:,.0f}",
    help="Calibrated via Split Conformalized Quantile Regression (80.05% empirical coverage)."
)

# Seller Deal Grader
if asking_price > 0:
    diff = asking_price - pred_mean_smeared
    pct_diff = (diff / pred_mean_smeared) * 100
    if asking_price <= cqr_low:
        st.success(
            f"**🟢 Great Deal:** Asking price (**${asking_price:,.0f}**) is **${abs(diff):,.0f} ({abs(pct_diff):.1f}%) below** "
            f"expected market value and sits below the 80% lower bound."
        )
    elif asking_price <= pred_mean_smeared * 1.08:
        st.info(
            f"**🟡 Fair Market Price:** Asking price (**${asking_price:,.0f}**) is within **{pct_diff:+.1f}%** "
            f"of expected market value."
        )
    elif asking_price <= cqr_high:
        st.warning(
            f"**🟠 Above Average:** Asking price (**${asking_price:,.0f}**) is **${diff:,.0f} ({pct_diff:+.1f}%) above** "
            f"expected market value, though still inside the upper 80% range."
        )
    else:
        st.error(
            f"**🔴 Overpriced:** Asking price (**${asking_price:,.0f}**) exceeds expected market value by "
            f"**${diff:,.0f} ({pct_diff:+.1f}%)** and sits above the 80% fair-market ceiling."
        )

# --- Section 3: 5-Year Projected Depreciation Simulator ---
st.subheader("3. 5-Year Projected Depreciation Curve (+12,000 miles/yr)")
sim_rows = []
for yr in range(6):
    row = input_df.copy()
    row["model_year"] = int(model_year) - yr
    row["milage"] = int(milage) + (yr * 12000)
    sim_rows.append(row)

sim_df = pd.concat(sim_rows, ignore_index=True)
sim_preds = np.exp(point_pipe.predict(sim_df)) * smearing_factor - 1

dep_col1, dep_col2 = st.columns([2, 1])
with dep_col1:
    chart_df = pd.DataFrame({
        "Projection Horizon": [f"Now ({int(milage):,} mi)"] + [
            f"Yr +{yr} ({int(milage) + yr * 12000:,} mi)" for yr in range(1, 6)
        ],
        "Expected Value ($)": sim_preds.round(0)
    }).set_index("Projection Horizon")
    st.line_chart(chart_df, y="Expected Value ($)")

with dep_col2:
    five_yr_loss = sim_preds[0] - sim_preds[-1]
    five_yr_pct = (five_yr_loss / sim_preds[0]) * 100
    st.metric("5-Year Projected Value", f"${sim_preds[-1]:,.0f}", delta=f"-${five_yr_loss:,.0f} (-{five_yr_pct:.1f}%)")
    with st.expander("Inspect Engineered Features"):
        st.dataframe(
            extract_features_v4(input_df)[
                ["vehicle_age", "miles_per_year", "horsepower", "displacement", "cylinders", "is_turbo", "is_luxury"]
            ],
            use_container_width=True
        )