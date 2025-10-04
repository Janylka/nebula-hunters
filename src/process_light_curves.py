import lightkurve as lk
import pandas as pd
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
LC_DIR = os.path.join(PROJECT_ROOT, "data", "light_curves")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "light_curves")
os.makedirs(PROCESSED_DIR, exist_ok=True)

fits_files = [f for f in os.listdir(LC_DIR) if f.endswith(".fits")]

print(f"🚀 Processing {len(fits_files)} FITS files...")

for f in fits_files:
    file_path = os.path.join(LC_DIR, f)
    tic_id = f.replace(".fits", "")

    try:
        lc_data = lk.read(file_path)

        # Take first light curve if collection
        if hasattr(lc_data, "lc_collection") or isinstance(lc_data, lk.collections.LightCurveCollection):
            lc = lc_data[0]
        else:
            lc = lc_data

        # Make sure time and flux exist
        if lc.time is None or lc.flux is None:
            raise ValueError("Missing time or flux data")

        df = pd.DataFrame({"time": lc.time.value, "flux": lc.flux})
        df.to_csv(os.path.join(PROCESSED_DIR, f"{tic_id}_lc.csv"), index=False)
        print(f"✅ Saved {tic_id}_lc.csv")

    except Exception as e:
        print(f"❌ Failed to process {f}: {e}")
