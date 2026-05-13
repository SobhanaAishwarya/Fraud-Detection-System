import streamlit as st
import numpy as np
import joblib

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💳 Fraud Detection System")

inputs = []

for i in range(1, 29):
    val = st.number_input(f"V{i}", value=0.0)
    inputs.append(val)

time = st.number_input("Time", value=0.0)
amount = st.number_input("Amount", value=0.0)

inputs = [time] + inputs + [amount]
inputs = np.array(inputs).reshape(1, -1)

inputs_scaled = scaler.transform(inputs)

if st.button("Predict"):
    result = model.predict(inputs_scaled)

    if result[0] == 1:
        st.error("⚠ Fraud Detected")
    else:
        st.success("✅ Legit Transaction")