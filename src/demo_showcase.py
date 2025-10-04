import pandas as pd
import numpy as np
import plotly.express as px
from tensorflow.keras.models import load_model

# -------------------------------
# 1. Config
# -------------------------------
FULL_DATA_PATH = "../data/processed/all_light_curves_full.csv"
LABELED_DATA_PATH = "../data/processed/light_curves_with_labels.csv"
MODEL_PATH = "../models/lstm_classifier.h5"
N_TICS_TO_SHOW = 10
SEQ_LEN = 50

# -------------------------------
# 2. Load data
# -------------------------------
df_full = pd.read_csv(FULL_DATA_PATH)
df_labels = pd.read_csv(LABELED_DATA_PATH)
print(f"Loaded {len(df_full)} rows from {FULL_DATA_PATH}")

# -------------------------------
# 3. Select random TICs for demo
# -------------------------------
unique_tics = df_full['tic_id'].unique()
selected_tics = np.random.choice(unique_tics, size=min(N_TICS_TO_SHOW, len(unique_tics)), replace=False)
df_demo = df_full[df_full['tic_id'].isin(selected_tics)]

# -------------------------------
# 4. Plot interactive light curves
# -------------------------------
fig = px.line(
    df_demo,
    x='time',
    y='flux_norm',
    color='tic_id',
    title="Demo: Light Curves + Predicted Flux",
    hover_data=['predicted_flux_norm', 'future_step']
)
fig.show()

# -------------------------------
# 5. Load classifier model
# -------------------------------
model = load_model(MODEL_PATH)
print(f"Classifier model loaded from {MODEL_PATH}")

# -------------------------------
# 6. Predict for last SEQ_LEN points of each TIC
# -------------------------------
predictions = []
for tic_id in selected_tics:
    group = df_full[df_full['tic_id'] == tic_id].sort_values('time')
    if len(group) < SEQ_LEN:
        continue
    seq = group['flux_norm'].values[-SEQ_LEN:].reshape(1, SEQ_LEN, 1)
    pred_prob = model.predict(seq, verbose=0)[0][0]
    predictions.append({'tic_id': tic_id, 'transit_prob': float(pred_prob)})

# -------------------------------
# 7. Show predictions
# -------------------------------
print("\n=== LSTM Transit Probabilities ===")
for p in predictions:
    print(f"TIC {p['tic_id']}: {p['transit_prob']*100:.2f}% chance of transit/event")
