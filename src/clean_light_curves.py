import pandas as pd
import os

# Define paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
ALL_LC_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "all_light_curves.csv")
CLEAN_LC_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "all_light_curves_clean.csv")

# Load combined light curves
print("🚀 Loading combined light curves...")
df = pd.read_csv(ALL_LC_PATH)
print(f"Initial rows: {len(df)}")

# -----------------------------
# 1️⃣ Filter out invalid/outlier flux values
# Keep only rows with positive flux values
df_filtered = df[df['flux'].notna() & (df['flux'] > 0)]
print(f"After outlier removal: {len(df_filtered)} rows")

# -----------------------------
# 2️⃣ Normalize flux values
# a) Z-score normalization per TIC
df_filtered.loc[:, 'flux_norm'] = df_filtered.groupby('tic_id')['flux'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# b) Min-max scaling per TIC
df_filtered.loc[:, 'flux_minmax'] = df_filtered.groupby('tic_id')['flux'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
)

# -----------------------------
# 3️⃣ Smooth flux using rolling window (5 points)
df_filtered.loc[:, 'flux_smooth'] = df_filtered.groupby('tic_id')['flux_norm'].transform(
    lambda x: x.rolling(window=5, min_periods=1, center=True).mean()
)

# -----------------------------
# 4️⃣ Save cleaned dataset
df_filtered.to_csv(CLEAN_LC_PATH, index=False)
print(f"✅ Cleaned light curves saved to {CLEAN_LC_PATH} with {len(df_filtered)} rows")
