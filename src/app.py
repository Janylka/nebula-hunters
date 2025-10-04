import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go

# ===============================
# Paths
# ===============================
BASE_DIR = os.path.dirname(__file__)
LSTM_DATA_FILE = os.path.join(BASE_DIR, "../data/processed/all_light_curves_full.csv")
TABULAR_DATA_FILE = os.path.join(BASE_DIR, "../data/processed/processed_exoplanets.csv")
TABULAR_MODEL_FILE = os.path.join(BASE_DIR, "../data/model.pkl")
LSTM_MODEL_FILE = os.path.join(BASE_DIR, "../models/lstm_final_model.h5")
LSTM_CLASSIFIER_FILE = os.path.join(BASE_DIR, "../models/lstm_classifier.h5")

# ===============================
# Load Data / Models
# ===============================
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"Data file not found: {file_path}")
        return None
    return pd.read_csv(file_path)

@st.cache_resource
def load_tabular_model(file_path):
    if not os.path.exists(file_path):
        st.error(f"Tabular model not found: {file_path}")
        return None
    return joblib.load(file_path)

@st.cache_resource
def load_lstm_model(file_path):
    if not os.path.exists(file_path):
        st.error(f"LSTM model not found: {file_path}")
        return None
    return load_model(file_path, compile=False)

# ===============================
# Prediction functions
# ===============================
def tabular_predict(df, model):
    features = ['pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_eqt']
    missing = [f for f in features if f not in df.columns]
    if missing:
        st.error(f"Missing columns for tabular prediction: {missing}")
        return df
    X = df[features]
    df["predicted_candidate"] = model.predict(X)
    return df

def lstm_predict_sequence(lstm_model, sequence):
    """sequence: numpy array of shape (timesteps, 1)"""
    return lstm_model.predict(sequence[np.newaxis, :, :], verbose=0).flatten()

def classify_sequence(lstm_classifier, sequence):
    """Return classification probabilities for each timestep"""
    return lstm_classifier.predict(sequence[np.newaxis, :, :], verbose=0).flatten()

# ===============================
# Streamlit App
# ===============================
st.set_page_config(page_title="Nebula Hunters 🚀", layout="wide")
st.title("Nebula Hunters: Hunting for Exoplanets with AI 🚀")
st.markdown("""
Predict potential exoplanet candidates from NASA Kepler light curves.  
Select a TIC to view historical data, LSTM predictions, and Tabular ML predictions.
""")

# Load datasets and models
lstm_df = load_data(LSTM_DATA_FILE)
tabular_df = load_data(TABULAR_DATA_FILE)
tab_model = load_tabular_model(TABULAR_MODEL_FILE)
lstm_model = load_lstm_model(LSTM_MODEL_FILE)
lstm_classifier = load_lstm_model(LSTM_CLASSIFIER_FILE)

if lstm_df is not None and tabular_df is not None and tab_model is not None and lstm_model is not None and lstm_classifier is not None:
    st.success("✅ Data and models loaded successfully!")

    # Select TIC
    tic_list = lstm_df['tic_id'].unique()
    selected_tic = st.selectbox("Select TIC", tic_list)

    # Show LSTM data
    tic_data = lstm_df[lstm_df['tic_id'] == selected_tic].sort_values("time").head(10)
    st.subheader("Sample Light Curve Data")
    st.dataframe(tic_data)

    # Prepare LSTM sequence
    flux_sequence = tic_data['flux_norm'].values.reshape(-1, 1)
    lstm_pred = lstm_predict_sequence(lstm_model, flux_sequence)
    lstm_class_prob = classify_sequence(lstm_classifier, flux_sequence)

    # Plot interactive graph
    fig = go.Figure()
    fig.add_scatter(x=tic_data['time'], y=tic_data['flux_norm'], mode='markers+lines', name='Historical Flux', marker=dict(color='blue', size=8))
    fig.add_scatter(x=tic_data['time'], y=lstm_pred, mode='lines+markers', name='LSTM Prediction', marker=dict(color='orange', size=8))
    fig.add_scatter(
        x=tic_data['time'], y=tic_data['flux_norm'], mode='markers',
        name='Predicted Candidate', marker=dict(color=['red' if p > 0.5 else 'green' for p in lstm_class_prob], size=12, symbol='circle-open')
    )
    fig.update_layout(title=f"TIC {selected_tic} Light Curve & Predictions", xaxis_title="Time (days)", yaxis_title="Normalized Flux", template="plotly_dark", legend=dict(y=0.99, x=0.01))
    st.plotly_chart(fig, use_container_width=True)

    # Tabular ML prediction
    if st.button("Run Tabular ML Model Prediction"):
        df_tab_pred = tabular_predict(tabular_df.copy(), tab_model)
        st.subheader("Tabular Model Predictions")
        display_cols = ['tic_id', 'time', 'flux_norm', 'predicted_candidate']
        # Filter columns if they exist
        display_cols = [col for col in display_cols if col in df_tab_pred.columns]
        st.dataframe(df_tab_pred[display_cols].head(10))

else:
    st.warning("App cannot run without data and models. Please check paths.")
