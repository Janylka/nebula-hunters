import pandas as pd

# -------------------------------
# 1. Config
# -------------------------------
CLEAN_DATA_PATH = "../data/processed/all_light_curves_clean.csv"
PREDICTIONS_PATH = "../data/processed/all_light_curves_predicted_batch.csv"
OUTPUT_PATH = "../data/processed/all_light_curves_full.csv"

# -------------------------------
# 2. Load data
# -------------------------------
df_history = pd.read_csv(CLEAN_DATA_PATH)
df_pred = pd.read_csv(PREDICTIONS_PATH)

print(f"Loaded {len(df_history)} rows of historical data")
print(f"Loaded {len(df_pred)} rows of predictions")

# -------------------------------
# 3. Rename columns for clarity
# -------------------------------
df_history = df_history.rename(columns={
    'flux_norm': 'flux_norm_history',
    'time': 'time_history'
})

df_pred = df_pred.rename(columns={
    'predicted_flux_norm': 'flux_norm_pred',
    'predicted_time': 'time_pred'
})

# -------------------------------
# 4. Merge predictions with history
# -------------------------------
# We'll concatenate history and predictions for each TIC
dfs = []

for tic_id, group_hist in df_history.groupby('tic_id'):
    group_pred = df_pred[df_pred['tic_id'] == tic_id]

    # Historical data
    df_hist_tic = pd.DataFrame({
        'tic_id': tic_id,
        'time': group_hist['time_history'].values,
        'flux_norm': group_hist['flux_norm_history'].values,
        'type': 'history'
    })

    # Predicted data
    df_pred_tic = pd.DataFrame({
        'tic_id': tic_id,
        'time': group_pred['time_pred'].values,
        'flux_norm': group_pred['flux_norm_pred'].values,
        'type': 'prediction'
    })

    # Combine
    dfs.append(pd.concat([df_hist_tic, df_pred_tic], ignore_index=True))

df_full = pd.concat(dfs, ignore_index=True)
print(f"Combined history + predictions: {len(df_full)} rows")

# -------------------------------
# 5. Save to CSV
# -------------------------------
df_full.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Full light curves saved to {OUTPUT_PATH}")
