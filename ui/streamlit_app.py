import streamlit as st
import pandas as pd
import requests
import joblib

# Load feature list
FEATURE_LIST = joblib.load("feature_list.pkl")

st.title("Battery SoH/SoC Prediction - Batch Demo")
st.write("Upload a CSV with feature columns. One row = one prediction.")

# Use session state to store the uploaded file's content
if "data" not in st.session_state:
    st.session_state["data"] = None
if "results" not in st.session_state:
    st.session_state["results"] = None

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.session_state["data"] = df
    st.session_state["results"] = None  # Clear results if new file uploaded

target = st.selectbox("Prediction Target", ["SoH_%", "SoC_Progress_%"])
model_type = st.selectbox("Model Type", ["xgboost", "lightgbm", "mlp"])

# Predict button (always available if data uploaded)
if st.session_state["data"] is not None:
    df = st.session_state["data"]
    missing_cols = set(FEATURE_LIST) - set(df.columns)
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
    else:
        if st.button("Predict with selected model"):
            results = []
            with st.spinner("Predicting..."):
                for idx, row in df.iterrows():
                    input_dict = row[FEATURE_LIST].to_dict()
                    payload = {
                        "features": input_dict,
                        "target": target,
                        "model_type": model_type
                    }
                    r = requests.post("http://fastapi:8080/predict", json=payload)
                    if r.status_code == 200:
                        result = r.json()["prediction"]
                    else:
                        result = f"Error: {r.status_code}"
                    results.append(result)
            # Insert Prediction as first column
            result_df = df.copy()
            result_df.insert(0, "Prediction", results)
            st.session_state["results"] = result_df
            st.success("Predictions completed.")
        # Display last results, if available
        if st.session_state["results"] is not None:
            st.dataframe(st.session_state["results"])
            st.download_button(
                label="Download predictions as CSV",
                data=st.session_state["results"].to_csv(index=False),
                file_name="predictions.csv"
            )
else:
    st.info("Please upload a CSV with the exact feature columns expected by the model.")
