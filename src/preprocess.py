import pandas as pd
import os

# Get project root (one level up from src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Define file paths
RAW_DATA = os.path.join(PROJECT_ROOT, "data", "raw", "kepler_exoplanets.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PROCESSED_DATA = os.path.join(PROCESSED_DIR, "processed_exoplanets.csv")

def preprocess_data():
    print("🚀 Preprocessing dataset...")

    # 1. Load raw dataset
    df = pd.read_csv(RAW_DATA)
    print(f"Initial dataset shape: {df.shape}")

    # 2. Select useful columns
    useful_columns = [
        "pl_name",       # Planet name
        "hostname",      # Host star name
        "disc_year",     # Discovery year
        "disc_facility", # Discovery facility
        "pl_orbper",     # Orbital period (days)
        "pl_rade",       # Planet radius (Earth radii)
        "pl_bmasse",     # Planet mass (Earth masses)
        "pl_eqt",         # Equilibrium temperature (K)
        "tic_id"  # Equilibrium temperature (K)
    ]
    df = df[useful_columns]

    print(f"Columns selected: {useful_columns}")

    # 3. Drop rows with missing values
    before_drop = len(df)
    df = df.dropna()
    after_drop = len(df)
    print(f"Dropped {before_drop - after_drop} rows with NaN values.")
    print(f"Remaining rows: {after_drop}")

    # 4. Create processed directory if not exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 5. Save cleaned dataset
    df.to_csv(PROCESSED_DATA, index=False)
    print(f"✅ Processed dataset saved as {PROCESSED_DATA} with {len(df)} rows.")

if __name__ == "__main__":
    preprocess_data()
