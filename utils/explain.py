# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import streamlit as st
# from sklearn.inspection import permutation_importance
# from google.generativeai import configure, GenerativeModel
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# # Configure Gemini
# configure(api_key=GEMINI_API_KEY)
# model = GenerativeModel("gemini-1.5-flash")

# def get_top_features(X, model, y, feature_names):
#     try:
#         if hasattr(model, "feature_importances_"):
#             importances = model.feature_importances_
#             top_indices = np.argsort(importances)[::-1][:5]
#             top_features = {
#                 feature_names[i]: round(importances[i], 4) for i in top_indices
#             }
#             return top_features, importances

#         elif hasattr(model, "coef_"):
#             importances = np.abs(model.coef_[0])
#             top_indices = np.argsort(importances)[::-1][:5]
#             top_features = {
#                 feature_names[i]: round(importances[i], 4) for i in top_indices
#             }
#             return top_features, importances

#         else:
#             # Try permutation importance as fallback
#             try:
#                 perm_result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
#                 importances = perm_result.importances_mean
#                 top_indices = np.argsort(importances)[::-1][:5]
#                 top_features = {
#                     feature_names[i]: round(importances[i], 4) for i in top_indices
#                 }
#                 return top_features, perm_result
#             except Exception as perm_e:
#                 return {"error": f"Model does not support feature importances and permutation importance failed: {perm_e}"}, None

#     except Exception as e:
#         return {"error": str(e)}, None

# def plot_permutation_importance(perm_result, feature_names):
#     """
#     Plot bar chart for permutation importance results.
#     """
#     try:
#         importances = perm_result.importances_mean
#         sorted_idx = np.argsort(importances)

#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.barh(range(len(importances)), importances[sorted_idx], align='center')
#         ax.set_yticks(range(len(importances)))
#         ax.set_yticklabels(np.array(feature_names)[sorted_idx])
#         ax.set_title("Permutation Importance (Top Features)")
#         st.pyplot(fig)
#         plt.clf()
#     except Exception as e:
#         st.warning(f"⚠️ Could not plot feature importances: {e}")

# def generate_gemini_explanation(top_features, prob):
#     """
#     Use Gemini to explain top contributing features.
#     """
#     prompt = f"""
#     A customer has a predicted default probability of {prob:.2f} on their credit card bill.

#     The top contributing features are:
#     {top_features}

#     Explain in simple terms why these features might increase or decrease default risk.
#     Keep it brief and user-friendly.
#     """

#     response = model.generate_content(prompt)
#     return response.text

# def plot_top_features_bar(top_features_dict):
#     try:
#         fig, ax = plt.subplots(figsize=(8, 4))
#         features = list(top_features_dict.keys())
#         importances = list(top_features_dict.values())

#         ax.barh(features[::-1], importances[::-1], color='skyblue')
#         ax.set_xlabel("Importance")
#         ax.set_title("Top Contributing Features")
#         plt.tight_layout()

#         return fig
#     except Exception as e:
#         return f"Plotting error: {e}"



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.inspection import permutation_importance
from google.generativeai import configure, GenerativeModel
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini
configure(api_key=GEMINI_API_KEY)
model = GenerativeModel("gemini-1.5-flash")

def get_top_features(X, model, y, feature_names):
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:5]
            top_features = {
                feature_names[i]: round(importances[i], 4) for i in top_indices
            }
            return top_features, importances

        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
            top_indices = np.argsort(importances)[::-1][:5]
            top_features = {
                feature_names[i]: round(importances[i], 4) for i in top_indices
            }
            return top_features, importances

        else:
            try:
                perm_result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
                importances = perm_result.importances_mean
                top_indices = np.argsort(importances)[::-1][:5]
                top_features = {
                    feature_names[i]: round(importances[i], 4) for i in top_indices
                }
                return top_features, perm_result
            except Exception as perm_e:
                return {"error": f"Model does not support feature importances and permutation importance failed: {perm_e}"}, None

    except Exception as e:
        return {"error": str(e)}, None

def plot_permutation_importance(perm_result, feature_names):
    try:
        importances = perm_result.importances_mean
        sorted_idx = np.argsort(importances)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(importances)), importances[sorted_idx], align='center')
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels(np.array(feature_names)[sorted_idx])
        ax.set_title("Permutation Importance (Top Features)")
        st.pyplot(fig)
        plt.clf()
    except Exception as e:
        st.warning(f"⚠️ Could not plot feature importances: {e}")

def generate_gemini_explanation(top_features, prob):
    prompt = f"""
    A customer has a predicted default probability of {prob:.2f} on their credit card bill.
    The top contributing features are:
    {top_features}
    Explain in simple terms why these features might increase or decrease default risk.
    Keep it brief and user-friendly.
    """
    response = model.generate_content(prompt)
    return response.text

def plot_top_features_bar(top_features_dict):
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        features = list(top_features_dict.keys())
        importances = list(top_features_dict.values())
        ax.barh(features[::-1], importances[::-1], color='skyblue')
        ax.set_xlabel("Importance")
        ax.set_title("Top Contributing Features")
        plt.tight_layout()
        return fig
    except Exception as e:
        return f"Plotting error: {e}"

