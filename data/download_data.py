import os
import requests
import pandas as pd

# NASA Exoplanet Archive API endpoint (returns CSV)
DATA_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+ps&format=csv"

# Directory where we will store the raw dataset
DATA_DIR = os.path.join(os.path.dirname(__file__), "raw")
OUTPUT_FILE = os.path.join(DATA_DIR, "kepler_exoplanets.csv")


def download_data():
    """Download dataset from NASA Exoplanet Archive and save it locally"""
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Downloading dataset from NASA...")

    # Use requests with a User-Agent to avoid HTTP 400 error
    response = requests.get(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code != 200:
        raise Exception(f"Failed to download data. HTTP {response.status_code}")

    # Save the file locally
    with open(OUTPUT_FILE, "wb") as f:
        f.write(response.content)

    print(f"✅ Saved dataset to {OUTPUT_FILE}")

    # Test loading with pandas
    df = pd.read_csv(OUTPUT_FILE)
    print("\n📊 Preview of dataset:")
    print(df.head())


if __name__ == "__main__":
    download_data()
