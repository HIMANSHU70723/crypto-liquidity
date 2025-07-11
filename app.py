import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Cryptocurrency Liquidity Predictor")

st.markdown("Enter the following features to predict liquidity score:")

avg_price = st.number_input("Average Price", value=100.0)
return_pct = st.number_input("Return (%)", value=0.01)
volatility = st.number_input("Volatility", value=5.0)
volume = st.number_input("Volume", value=100000.0)

if st.button("Predict Liquidity Score"):
    inputs = np.array([[avg_price, return_pct, volatility, volume]])
    scaled_inputs = scaler.transform(inputs)
    prediction = model.predict(scaled_inputs)
    st.success(f"Predicted Liquidity Score: {prediction[0]:.2f}")
