from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score, recall_score, precision_score
import pandas as pd
import streamlit as st

def run_fairness_check(y_true, y_pred, sensitive_features):
    metrics = {
        'accuracy': accuracy_score,
        'selection_rate': selection_rate,
        'precision': precision_score,
        'recall': recall_score
    }

    mf = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)

    st.subheader("⚖️ Fairness Check Results")
    st.dataframe(mf.by_group.round(3))
    
    disparity = mf.difference(method='between_groups')
    st.subheader("🔍 Disparity Between Groups")
    st.json(disparity.round(3).to_dict())

    return mf
