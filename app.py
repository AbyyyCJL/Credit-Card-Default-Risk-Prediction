import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from utils.explain import get_top_features, plot_permutation_importance, generate_gemini_explanation, plot_top_features_bar
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Load all models
model_files = {
    "XGBoost": "models/xgboost_model.pkl",
    "Random Forest": "models/randomforest_model.pkl",
    "Logistic Regression": "models/logisticregression_model.pkl"
}
models = {name: load(path) for name, path in model_files.items()}

# Load the scaler
scaler = load("models/scaler.pkl")

st.set_page_config(page_title="Credit Default Predictor", layout="wide")
st.title("💳 Credit Card Default Risk Predictor")

st.markdown("""
This app predicts whether a customer is likely to default on their credit card next month, using:
- **Multiple ML Models (XGBoost, RF, Logistic Regression)**
- **GenAI (Gemini) for Explanation**
- **Fairness & Transparency**
""")

st.sidebar.header("📋 Input Customer Details")

def user_input_features():
    limit = st.sidebar.number_input("Credit Limit (LIMIT_BAL)", min_value=10000, max_value=1000000, value=200000, step=1000)
    age = st.sidebar.slider("Age", 18, 80, 30)
    sex = st.sidebar.selectbox("Gender", ["Male", "Female"])
    education = st.sidebar.selectbox("Education", ["Graduate School", "University", "High School", "Others"])
    marriage = st.sidebar.selectbox("Marital Status", ["Married", "Single", "Others"])

    pay_0 = st.sidebar.slider("PAY_0 (Last Month Repayment Status)", -1, 8, 0)
    pay_2 = st.sidebar.slider("PAY_2 (Two Months Ago)", -1, 8, 0)

    bill_amt1 = st.sidebar.number_input("Last Bill Amount (BILL_AMT1)", min_value=0, value=50000)
    pay_amt1 = st.sidebar.number_input("Last Payment Made (PAY_AMT1)", min_value=0, value=10000)

    # Mapping
    sex_map = {"Male": 1, "Female": 2}
    edu_map = {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}
    mar_map = {"Married": 1, "Single": 2, "Others": 3}

    # Base Features
    base = {
        'LIMIT_BAL': limit,
        'SEX': sex_map[sex],
        'EDUCATION': edu_map[education],
        'MARRIAGE': mar_map[marriage],
        'AGE': age,
        'PAY_0': pay_0, 'PAY_2': pay_2,
        'PAY_3': 0, 'PAY_4': 0, 'PAY_5': 0, 'PAY_6': 0,
        'BILL_AMT1': bill_amt1, 'BILL_AMT2': bill_amt1, 'BILL_AMT3': bill_amt1,
        'BILL_AMT4': bill_amt1, 'BILL_AMT5': bill_amt1, 'BILL_AMT6': bill_amt1,
        'PAY_AMT1': pay_amt1, 'PAY_AMT2': pay_amt1, 'PAY_AMT3': pay_amt1,
        'PAY_AMT4': pay_amt1, 'PAY_AMT5': pay_amt1, 'PAY_AMT6': pay_amt1,
    }

    df = pd.DataFrame([base])

    # Feature Engineering (must match train_model.py)
    df["TOTAL_BILL"] = df[[f"BILL_AMT{i}" for i in range(1, 7)]].sum(axis=1)
    df["TOTAL_PAYMENT"] = df[[f"PAY_AMT{i}" for i in range(1, 7)]].sum(axis=1)
    df["PAYMENT_RATIO"] = df["TOTAL_PAYMENT"] / df["TOTAL_BILL"].replace(0, 1)
    df["AVG_DELAY"] = df[[f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]]].mean(axis=1)

    return df

df_input = user_input_features()

if st.button("Predict"):
    st.session_state.predict = True

if st.session_state.get("predict", False):
    scaled_input = scaler.transform(df_input)

    model_probs = {}
    for name, model in models.items():
        prob = model.predict_proba(scaled_input)[0][1]
        model_probs[name] = prob

    sorted_models = sorted(model_probs.items(), key=lambda x: x[1], reverse=True)
    best_model_name, best_prob = sorted_models[0]
    best_model = models[best_model_name]
    risk = "High ⚠️" if best_prob >= 0.75 else "Medium ⚠️" if best_prob >= 0.5 else "Low ✅"

    st.subheader("📈 Model Comparison")
    for model_name, prob in model_probs.items():
        st.write(f"🔹 **{model_name}** → Default Probability: `{prob:.2f}`")

    st.success(f"🏆 Best Model: **{best_model_name}**")
    st.metric("Default Probability", f"{best_prob:.2f}")
    st.metric("Risk Label", risk)

    # Top Features & GenAI
    top_feats, result = get_top_features(df_input, best_model, [0], df_input.columns)
    explanation = generate_gemini_explanation(top_feats, best_prob)

    st.subheader("🧠 GenAI Explanation")
    st.write(explanation)

    if "error" in top_feats:
        st.warning(f"⚠️ Could not compute feature importances: {top_feats['error']}")
    else:
        st.subheader("📊 Top Contributing Features")
        st.json(top_feats)

    st.subheader("📊 Top Features Visualization")
    fig = plot_top_features_bar(top_feats)
    if isinstance(fig, str):
        st.warning(fig)
    else:
        st.pyplot(fig)
