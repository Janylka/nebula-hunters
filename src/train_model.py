"""
Train a simple ML model to predict potential exoplanets
based on processed Kepler data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Paths
PROCESSED_DATA = os.path.join("..", "data", "processed", "processed_exoplanets.csv")
MODEL_PATH = os.path.join("..", "data", "model.pkl")


def create_label(df):
    """
    Create a simple target label 'is_candidate':
    1 if potential habitable candidate (pl_eqt ~ 200-350K and radius < 2 Earth radii)
    0 otherwise
    """
    df['is_candidate'] = ((df['pl_eqt'] >= 200) & (df['pl_eqt'] <= 350) & (df['pl_rade'] < 2)).astype(int)
    return df


def train_model():
    print("🚀 Loading processed dataset...")
    df = pd.read_csv(PROCESSED_DATA)

    print(f"Dataset shape: {df.shape}")

    # Create label
    df = create_label(df)

    # Select features
    features = ['pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_eqt']
    X = df[features]
    y = df['is_candidate']

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    print("💻 Training RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # Save model
    joblib.dump(clf, MODEL_PATH)
    print(f"\n✅ Model saved at {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
