import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered",
)

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #f7f8fa; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 780px; }

    h1 { font-weight: 700; color: #1a1a2e; }

    .subtitle {
        color: #6b7280;
        font-size: 0.95rem;
        margin-top: -0.6rem;
        margin-bottom: 1.6rem;
    }

    .section-label {
        font-weight: 600;
        font-size: 0.95rem;
        color: #374151;
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #2563eb, #1d4ed8);
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.7rem 0;
        border-radius: 10px;
        border: none;
        margin-top: 1.5rem;
        transition: transform 0.05s ease-in-out;
    }
    div.stButton > button:hover {
        transform: scale(1.01);
        background: linear-gradient(90deg, #1d4ed8, #1e40af);
    }

    .result-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        color: white;
        border-radius: 16px;
        padding: 1.8rem;
        text-align: center;
        margin-top: 1.5rem;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.25);
    }
    .result-card .label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .result-card .price {
        font-size: 2.4rem;
        font-weight: 700;
        margin-top: 0.3rem;
        color: #4ade80;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🏠 House Price Prediction")
st.markdown('<p class="subtitle">Enter the property details below to get an estimated market price.</p>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("house_price_model.pkl")

try:
    model = load_model()
    st.success("Model loaded successfully ✅", icon="✅")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
with st.form("prediction_form"):

    st.markdown('<div class="section-label">🛏️ Rooms</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=15, value=3, step=1)
    with c2:
        bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=0.25)

    st.markdown('<div class="section-label">📐 Size (sqft)</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        sqft_living = st.number_input("Living area", min_value=100, max_value=15000, value=1800, step=50)
        sqft_above = st.number_input("Above ground", min_value=0, max_value=15000, value=1500, step=50)
    with c2:
        sqft_lot = st.number_input("Lot size", min_value=100, max_value=200000, value=5000, step=100)
        sqft_basement = st.number_input("Basement", min_value=0, max_value=10000, value=300, step=50)

    st.markdown('<div class="section-label">🏗️ Structure</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        floors = st.selectbox("Floors", [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], index=1)
        condition = st.slider("Condition (1=poor, 5=excellent)", 1, 5, 3)
    with c2:
        view = st.slider("View quality (0=none, 4=excellent)", 0, 4, 0)
        waterfront = st.radio("Waterfront?", ["No", "Yes"], horizontal=True)

    st.markdown('<div class="section-label">📅 Year</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        yr_built = st.number_input("Year built", min_value=1900, max_value=2026, value=1990, step=1)
    with c2:
        yr_renovated = st.number_input("Year renovated (0 if never)", min_value=0, max_value=2026, value=0, step=1)

    st.markdown('<div class="section-label">📍 Location</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        sale_date = st.date_input("Sale date")
        street = st.text_input("Street address", value="123 Main St")
    with c2:
        city = st.text_input("City", value="Seattle")
        statezip = st.text_input("State + ZIP", value="WA 98103")
    country = st.text_input("Country", value="USA")

    submitted = st.form_submit_button("🔮 Predict Price")

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submitted:
    waterfront_val = 1 if waterfront == "Yes" else 0

    # NOTE: feature order/names below mirror the raw CSV columns
    # (minus "price"). The model's pipeline appears to expect the
    # full original column set, including date/location fields.
    # If prediction still fails, share your training script so the
    # exact preprocessing (encoders, date parsing, etc.) can be matched.
    features = pd.DataFrame([{
        "date": sale_date.strftime("%Y-%m-%d 00:00:00"),
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront_val,
        "view": view,
        "condition": condition,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated,
        "street": street,
        "city": city,
        "statezip": statezip,
        "country": country,
    }])

    try:
        prediction = model.predict(features)[0]
        prediction = max(prediction, 0)  # guard against negative output

        st.markdown(f"""
        <div class="result-card">
            <div class="label">Estimated Price</div>
            <div class="price">${prediction:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("See the input values used"):
            st.dataframe(features, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "This usually means the input features don't match what the "
            "model was trained on. Check the feature names/order in your "
            "training script against the `features` DataFrame in app.py."
        )

st.markdown("---")
st.caption("Built with Streamlit · Model: house_price_model.pkl")
