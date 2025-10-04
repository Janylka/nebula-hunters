"""
Load trained model and make predictions on new exoplanet data.
"""

import pandas as pd
import joblib
import os

# Paths
MODEL_PATH = os.path.join("..", "data", "model.pkl")
PROCESSED_DATA = os.path.join("..", "data", "processed", "processed_exoplanets.csv")


def predict_new_samples():
    # Load model
    print("💾 Loading trained model...")
    clf = joblib.load(MODEL_PATH)

    # Load processed data (for demo purposes)
    df = pd.read_csv(PROCESSED_DATA)

    # Select features
    features = ['pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_eqt']
    X = df[features]

    # Predict
    print("🔮 Making predictions...")
    df['predicted_candidate'] = clf.predict(X)

    # Show sample predictions
    print("\n📊 Sample predictions:")
    print(df[['pl_name', 'predicted_candidate']].head(10))


if __name__ == "__main__":
    predict_new_samples()
