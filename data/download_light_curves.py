import os
import pandas as pd
import lightkurve as lk

# Get project root (one level up from data/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Paths
PROCESSED_DATA = os.path.join(PROJECT_ROOT, "data", "processed", "processed_exoplanets.csv")
LIGHT_CURVES_DIR = os.path.join(PROJECT_ROOT, "data", "light_curves")

# Ensure directory exists
os.makedirs(LIGHT_CURVES_DIR, exist_ok=True)

def download_light_curves():
    print("🚀 Starting download of light curves...")

    # Load processed dataset
    df = pd.read_csv(PROCESSED_DATA)

    # Check if 'tic_id' exists
    if 'tic_id' not in df.columns:
        print("⚠️ No 'tic_id' column found in processed dataset!")
        return

    # Get list of TESS IDs, remove NaNs
    tic_ids = df['tic_id'].dropna().astype(str).tolist()

    if len(tic_ids) == 0:
        print("⚠️ No TESS IDs found in dataset.")
        return

    # Loop through IDs and download light curves
    for tic_id in tic_ids:
        # Remove 'TIC ' prefix if exists
        tic_id_clean = tic_id.replace("TIC ", "").strip()
        filename = os.path.join(LIGHT_CURVES_DIR, f"TIC_{tic_id_clean}.fits")

        # Skip if already downloaded
        if os.path.exists(filename):
            print(f"✅ Light curve for TIC {tic_id_clean} already exists, skipping.")
            continue

        try:
            print(f"📥 Downloading light curve for TIC {tic_id_clean}...")
            lc_search = lk.search_lightcurve(f"TIC {tic_id_clean}", mission='TESS')
            if lc_search:
                lc = lc_search.download()
                lc.to_fits(filename, overwrite=True)
                print(f"✅ Saved: {filename}")
            else:
                print(f"⚠️ No light curve found for TIC {tic_id_clean}")
        except Exception as e:
            print(f"❌ Failed for TIC {tic_id_clean}: {e}")

    print("✅ Light curve download complete!")

if __name__ == "__main__":
    download_light_curves()
