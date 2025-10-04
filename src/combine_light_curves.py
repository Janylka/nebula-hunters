import pandas as pd
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # one level up from src/
PROCESSED_LC_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "light_curves")
COMBINED_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "all_light_curves.csv")


def combine_light_curves():
    print("🚀 Combining processed light curves...")

    # List all processed CSVs
    csv_files = [f for f in os.listdir(PROCESSED_LC_DIR) if f.endswith("_lc.csv")]
    print(f"Found {len(csv_files)} processed CSV files.")

    combined_df = pd.DataFrame()

    for file in csv_files:
        filepath = os.path.join(PROCESSED_LC_DIR, file)
        try:
            df = pd.read_csv(filepath)
            # Extract TIC ID from filename
            tic_id = file.replace("_lc.csv", "")
            df['tic_id'] = tic_id  # add TIC ID column

            combined_df = pd.concat([combined_df, df], ignore_index=True)
        except Exception as e:
            print(f"❌ Failed to process {file}: {e}")

    print(f"✅ Combined dataset shape: {combined_df.shape}")

    # Save combined CSV
    combined_df.to_csv(COMBINED_CSV, index=False)
    print(f"✅ Saved combined light curves CSV to {COMBINED_CSV}")


if __name__ == "__main__":
    combine_light_curves()
