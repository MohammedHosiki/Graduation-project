import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# --- Page Config ---
st.set_page_config(
    page_title="Jeddah Real Estate Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# --- Load Model and Data ---
@st.cache_resource
def load_model():
    return joblib.load('models/best_model.joblib')

@st.cache_data
def load_districts():
    with open('models/district_list.json', 'r', encoding='utf-8') as f:
        return json.load(f)

@st.cache_data
def load_feature_columns():
    return joblib.load('models/feature_columns.joblib')

model = load_model()
districts = load_districts()
feature_columns = load_feature_columns()

# --- Header ---
st.markdown(
    """
    <div style="
        background-color: #1a3c2e;
        padding: 20px 30px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 25px;
    ">
        <img src="data:image/png;base64,{logo_b64}" style="height: 80px;">
        <div>
            <h1 style="color: white; margin: 0; font-size: 1.8em;">Jeddah Real Estate Rental Price Predictor</h1>
            <p style="color: #cce8d4; margin: 4px 0 0 0;">King Abdulaziz University — Graduation Project</p>
        </div>
    </div>
    """.replace("{logo_b64}", __import__('base64').b64encode(open('logo.png','rb').read()).decode()),
    unsafe_allow_html=True
)
st.divider()

# --- Input Form ---
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        district = st.selectbox("District", options=districts)
        area = st.number_input("Area (sqm)", min_value=10, max_value=500, value=100, step=5)
        age = st.number_input("Property Age (years)", min_value=0, max_value=50, value=0, step=1)
        street_width = st.slider("Street Width (m)", min_value=5, max_value=60, value=15)

    with col2:
        beds = st.selectbox("Bedrooms", options=list(range(1, 8)), index=1)
        wc = st.selectbox("Bathrooms", options=list(range(1, 6)), index=1)
        livings = st.selectbox("Living Rooms", options=list(range(0, 3)), index=1)
        furnished = st.checkbox("Furnished")
        ac = st.checkbox("Air Conditioning")
        kitchen = st.checkbox("Kitchen")

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

# --- Prediction ---
if submitted:
    input_data = pd.DataFrame([{
        'district': district,
        'furnished': int(furnished),
        'ac': int(ac),
        'kitchen': int(kitchen),
        'age': age,
        'street_width': street_width,
        'area': area,
        'wc': wc,
        'livings': livings,
        'beds': beds,
        'year': 2022,
        'month': 7,
        'total_rooms': beds + livings + wc
    }])

    # Reorder columns to match training data
    input_data = input_data[feature_columns]

    prediction = model.predict(input_data)[0]
    prediction = max(prediction, 0)  # ensure non-negative

    st.divider()
    st.success(f"Estimated Monthly Rent: **{prediction:,.0f} SAR**")

    st.caption(
        "This prediction is based on historical data from 2021-2022 "
        "and is for educational purposes only."
    )

# --- Footer ---
st.divider()
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.85em;'>"
    "Jeddah Real Estate Price Prediction — Graduation Project"
    "</div>",
    unsafe_allow_html=True
)
