import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import plotly.express as px
from tqdm import tqdm

# -------------------------------
# 1. Config
# -------------------------------
SEQ_LEN = 50
MODEL_PATH = "../models/lstm_classifier.h5"
DATA_PATH = "../data/processed/all_light_curves_clean.csv"
OUTPUT_PATH = "../data/processed/all_light_curves_with_ai_predictions.csv"

# -------------------------------
# 2. Load cleaned light curves
# -------------------------------
print("🚀 Loading cleaned light curves...")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} rows from {DATA_PATH}")

# -------------------------------
# 3. Load trained classifier
# -------------------------------
model = load_model(MODEL_PATH)
print(f"LSTM classifier loaded from {MODEL_PATH}")

# -------------------------------
# 4. Predict AI labels for each TIC
# -------------------------------
predictions = []

for tic_id, group in tqdm(df.groupby("tic_id"), desc="Predicting AI labels"):
    group_sorted = group.sort_values("time")
    flux_values = group_sorted["flux_norm"].values

    # Prepare sequences
    if len(flux_values) < SEQ_LEN:
        continue
    seqs = []
    for i in range(len(flux_values) - SEQ_LEN + 1):
        seqs.append(flux_values[i:i+SEQ_LEN])
    seqs = np.array(seqs).reshape(-1, SEQ_LEN, 1)

    # Predict
    preds = model.predict(seqs, verbose=0).squeeze()

    # Align predictions with original times
    preds_full = np.zeros(len(flux_values))
    preds_full[SEQ_LEN-1:] = preds  # first SEQ_LEN-1 points have no prediction
    predictions.append(pd.DataFrame({
        "tic_id": tic_id,
        "time": group_sorted["time"].values,
        "flux_norm": flux_values,
        "ai_label": preds_full
    }))

# Combine all TIC predictions
df_pred = pd.concat(predictions, ignore_index=True)
df_pred.to_csv(OUTPUT_PATH, index=False)
print(f"✅ AI predictions saved to {OUTPUT_PATH} with {len(df_pred)} rows")

# -------------------------------
# 5. Interactive visualization for a few random TICs
# -------------------------------
sample_tics = df_pred["tic_id"].drop_duplicates().sample(10, random_state=42).values
for tic in sample_tics:
    df_tic = df_pred[df_pred["tic_id"] == tic]
    fig = px.scatter(df_tic, x="time", y="flux_norm",
                     color="ai_label", color_continuous_scale="Viridis",
                     hover_data={"ai_label": True, "time": True, "flux_norm": True},
                     title=f"TIC {tic} - Light Curve with AI Predictions")
    fig.show()
