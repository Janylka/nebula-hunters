import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tqdm import tqdm

# -------------------------------
# 1. Config
# -------------------------------
SEQ_LEN = 50
N_FUTURE = 30  # number of steps to predict ahead
MODEL_PATH = "lstm_final_model.h5"
CLEAN_DATA_PATH = "../data/processed/all_light_curves_clean.csv"
OUTPUT_PATH = "../data/processed/all_light_curves_predicted.csv"

# -------------------------------
# 2. Load cleaned data
# -------------------------------
df = pd.read_csv(CLEAN_DATA_PATH)
print(f"Loaded {len(df)} rows from {CLEAN_DATA_PATH}")

# -------------------------------
# 3. Load trained LSTM model
# -------------------------------
model = load_model(MODEL_PATH)
print(f"LSTM model loaded from {MODEL_PATH}")

# -------------------------------
# 4. Predict future flux for each TIC
# -------------------------------
all_predictions = []

for tic_id, group in tqdm(df.groupby('tic_id'), desc="Predicting TICs"):
    flux_seq = group['flux_norm'].values.tolist()

    # Only predict if we have at least SEQ_LEN points
    if len(flux_seq) < SEQ_LEN:
        continue

    input_seq = flux_seq[-SEQ_LEN:]  # last SEQ_LEN points
    future_preds = []

    for _ in range(N_FUTURE):
        seq_array = np.array(input_seq[-SEQ_LEN:]).reshape(1, SEQ_LEN, 1)
        pred = model.predict(seq_array, verbose=0)[0][0]
        future_preds.append(pred)
        input_seq.append(pred)

    # Save predictions with TIC ID and step index
    for step, pred_flux in enumerate(future_preds, 1):
        all_predictions.append({
            'tic_id': tic_id,
            'future_step': step,
            'predicted_flux_norm': pred_flux
        })

# -------------------------------
# 5. Save all predictions to CSV
# -------------------------------
df_pred = pd.DataFrame(all_predictions)
df_pred.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Predicted future flux saved to {OUTPUT_PATH} with {len(df_pred)} rows")
