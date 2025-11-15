%%writefile app_streamlit.py
import streamlit as st
import joblib
import numpy as np

# -------------------------
# Load trained model
# -------------------------
model = joblib.load("/content/drive/MyDrive/AI ML/model.pkl")

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(page_title="ShipmentSure", layout="centered")
st.title("🚚 ShipmentSure - Delivery Prediction App")
st.markdown("### Predict whether your shipment will arrive on time!")

# -------------------------
# Input Fields
# -------------------------
st.subheader("📦 Enter Shipment Details")

# Categorical mappings (same as training)
warehouse_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 4}
shipment_mode_map = {'Ship': 0, 'Flight': 1, 'Road': 2}
importance_map = {'low': 0, 'medium': 1, 'high': 2}
gender_map = {'M': 0, 'F': 1}

# Collect inputs
ID = st.number_input("Shipment ID", min_value=1, step=1)
Warehouse_block = st.selectbox("Warehouse Block", list(warehouse_map.keys()))
Mode_of_Shipment = st.selectbox("Mode of Shipment", list(shipment_mode_map.keys()))
Customer_care_calls = st.slider("Customer Care Calls", 1, 10, 3)
Customer_rating = st.slider("Customer Rating", 1, 5, 4)
Cost_of_the_Product = st.number_input("Cost of Product", min_value=1.0, step=1.0)
Prior_purchases = st.slider("Prior Purchases", 0, 10, 2)
Product_importance = st.selectbox("Product Importance", list(importance_map.keys()))
Gender = st.selectbox("Customer Gender", list(gender_map.keys()))
Discount_offered = st.number_input("Discount Offered (%)", min_value=0.0, step=0.5)
Weight_in_gms = st.number_input("Weight (grams)", min_value=100.0, step=50.0)
Cost_to_Weight_Ratio = st.number_input("Cost to Weight Ratio", min_value=0.0, step=0.001)

# -------------------------
# Prediction Logic
# -------------------------
if st.button("🚀 Predict Delivery Status"):
    X = np.array([[
        ID,
        warehouse_map[Warehouse_block],
        shipment_mode_map[Mode_of_Shipment],
        Customer_care_calls,
        Customer_rating,
        Cost_of_the_Product,
        Prior_purchases,
        importance_map[Product_importance],
        gender_map[Gender],
        Discount_offered,
        Weight_in_gms,
        Cost_to_Weight_Ratio
    ]])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1] if hasattr(model, "predict_proba") else None

    result = "🟢 On Time" if prediction == 1 else "🔴 Delayed"
    st.subheader("📊 Prediction Result:")
    st.write(f"**Status:** {result}")
    if probability:
        st.write(f"**Confidence:** {probability:.2%}")
