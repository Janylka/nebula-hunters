import pandas as pd
import numpy as np
import random
import os

# -------------------------------
# 1. Config
# -------------------------------
INPUT_PATH = "../data/processed/all_light_curves_clean.csv"
OUTPUT_PATH = "../data/processed/light_curves_with_labels.csv"

# Parameters for synthetic transit
TRANSIT_DEPTH = 0.05   # 5% dip
TRANSIT_WIDTH = 10     # number of points per dip
TRANSIT_PROB = 0.3     # probability that a TIC gets a synthetic transit

# -------------------------------
# 2. Load data
# -------------------------------
print("🚀 Loading cleaned light curves...")
df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} rows from {INPUT_PATH}")

# -------------------------------
# 3. Add synthetic transits
# -------------------------------
all_labeled = []

for tic_id, group in df.groupby("tic_id"):
    group_sorted = group.sort_values("time").copy()
    group_sorted["label"] = 0  # default: no transit

    if random.random() < TRANSIT_PROB and len(group_sorted) > TRANSIT_WIDTH * 2:
        # pick random start index for transit
        start = random.randint(0, len(group_sorted) - TRANSIT_WIDTH - 1)
        end = start + TRANSIT_WIDTH

        # add flux dip
        group_sorted.loc[start:end, "flux_norm"] *= (1 - TRANSIT_DEPTH)
        group_sorted.loc[start:end, "label"] = 1  # mark as transit

    all_labeled.append(group_sorted)

df_labeled = pd.concat(all_labeled, ignore_index=True)

# -------------------------------
# 4. Save dataset
# -------------------------------
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_labeled.to_csv(OUTPUT_PATH, index=False)

print(f"✅ Saved labeled dataset with synthetic transits to {OUTPUT_PATH}")
print(f"Total rows: {len(df_labeled)}, with transits: {df_labeled['label'].sum()}")
