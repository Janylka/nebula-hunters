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
OUTPUT_PATH = "../data/processed/all_light_curves_predicted_batch.csv"

# -------------------------------
# 2. Load cleaned data
# -------------------------------
df = pd.read_csv(CLEAN_DATA_PATH)
print(f"Loaded {len(df)} rows from {CLEAN_DATA_PATH}")

# -------------------------------
# 3. Prepare sequences for all TICs
# -------------------------------
tic_groups = df.groupby('tic_id')
tic_sequences = []
tic_times = []
tic_ids = []

for tic_id, group in tic_groups:
    group_sorted = group.sort_values('time')
    if len(group_sorted) < SEQ_LEN:
        continue
    flux_seq = group_sorted['flux_norm'].values[-SEQ_LEN:]
    time_seq = group_sorted['time'].values[-SEQ_LEN:]
    tic_sequences.append(flux_seq.reshape(SEQ_LEN, 1))
    tic_times.append(time_seq)
    tic_ids.append(tic_id)

tic_sequences = np.array(tic_sequences)
print(f"Prepared {len(tic_sequences)} TIC sequences for batch prediction")

# -------------------------------
# 4. Load trained LSTM model
# -------------------------------
model = load_model(MODEL_PATH)
print(f"LSTM model loaded from {MODEL_PATH}")

# -------------------------------
# 5. Predict N_FUTURE steps in batch
# -------------------------------
all_predictions = []

for step in tqdm(range(1, N_FUTURE + 1), desc="Predicting future steps"):
    preds = model.predict(tic_sequences, verbose=0)
    preds = preds.squeeze()  # shape: (num_TICs,)

    for i, tic_id in enumerate(tic_ids):
        last_time = tic_times[i][-1]
        mean_delta = np.mean(np.diff(tic_times[i]))
        future_time = last_time + step * mean_delta
        all_predictions.append({
            'tic_id': tic_id,
            'future_step': step,
            'predicted_flux_norm': preds[i],
            'predicted_time': future_time
        })

    # Append predicted step to sequence for next iteration
    tic_sequences = np.concatenate([tic_sequences[:, 1:, :], preds.reshape(-1, 1, 1)], axis=1)

# -------------------------------
# 6. Save all predictions to CSV
# -------------------------------
df_pred = pd.DataFrame(all_predictions)
df_pred.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Batch predicted future flux saved to {OUTPUT_PATH} with {len(df_pred)} rows")
