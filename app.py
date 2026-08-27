import streamlit as st
import joblib

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠"
)

st.title("🏠 House Price Prediction")

# Load the trained model
model = joblib.load("house_price_model.pkl")

st.success("Model loaded successfully! ✅")

st.write("Welcome to the House Price Prediction App")
