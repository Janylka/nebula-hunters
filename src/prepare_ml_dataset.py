import pandas as pd
import numpy as np
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
CLEAN_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "all_light_curves_clean.csv")
TRANSIT_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "transits.csv")  # optional
ML_DIR = os.path.join(PROJECT_ROOT, "data", "ml_ready")
os.makedirs(ML_DIR, exist_ok=True)
ML_DATA_FILE = os.path.join(ML_DIR, "ml_light_curves.npy")
ML_LABELS_FILE = os.path.join(ML_DIR, "ml_labels.npy")
ML_TIC_FILE = os.path.join(ML_DIR, "ml_tic_ids.npy")

# Parameters
WINDOW_SIZE = 200  # number of time points per segment
STEP_SIZE = 100  # overlap between windows

print("🚀 Preparing ML dataset with features and labels...")

# Load cleaned data
df = pd.read_csv(CLEAN_CSV)
print(f"Loaded {len(df)} rows from cleaned light curves.")

# Load transit info if available
if os.path.exists(TRANSIT_CSV):
    df_transits = pd.read_csv(TRANSIT_CSV)
else:
    df_transits = pd.DataFrame(columns=['tic_id', 'transit_start', 'transit_end'])

X = []  # flux windows
y = []  # labels (0/1)
tic_ids = []

# Group by TIC ID
for tic, group in df.groupby('tic_id'):
    flux = group['flux_norm'].values
    flux_minmax = group['flux_minmax'].values
    flux_smooth = group['flux_smooth'].values
    time = group['time'].values

    # Optional: compute flux derivative
    flux_deriv = np.gradient(flux_smooth)

    # Combine features: flux_norm, flux_minmax, flux_smooth, flux_deriv
    features = np.stack([flux, flux_minmax, flux_smooth, flux_deriv], axis=1)  # shape: (n_points, 4)

    # Prepare transit labels
    labels = np.zeros(len(group), dtype=int)
    tic_transits = df_transits[df_transits['tic_id'] == tic]
    for _, row in tic_transits.iterrows():
        mask = (time >= row['transit_start']) & (time <= row['transit_end'])
        labels[mask] = 1

    # Slide window
    n_points = len(group)
    for start in range(0, n_points - WINDOW_SIZE + 1, STEP_SIZE):
        X.append(features[start:start + WINDOW_SIZE])
        y.append(labels[start:start + WINDOW_SIZE])
        tic_ids.append(tic)

X = np.array(X)
y = np.array(y)
tic_ids = np.array(tic_ids)

print(f"Generated {len(X)} windows with {X.shape[2]} features each.")

# Save ML dataset
np.save(ML_DATA_FILE, X)
np.save(ML_LABELS_FILE, y)
np.save(ML_TIC_FILE, tic_ids)

print(f"✅ ML dataset saved as {ML_DATA_FILE}")
print(f"✅ Labels saved as {ML_LABELS_FILE}")
print(f"✅ TIC IDs saved as {ML_TIC_FILE}")
