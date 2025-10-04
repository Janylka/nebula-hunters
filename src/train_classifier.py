import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. Config
# -------------------------------
SEQ_LEN = 50       # length of input sequence
BATCH_SIZE = 16
EPOCHS = 10
INPUT_PATH = "../data/processed/light_curves_with_labels.csv"
MODEL_PATH = "../models/lstm_classifier.h5"

# -------------------------------
# 2. Load dataset
# -------------------------------
print("🚀 Loading labeled dataset...")
df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} rows from {INPUT_PATH}")

# -------------------------------
# 3. Prepare sequences
# -------------------------------
X, y = [], []

for tic_id, group in df.groupby("tic_id"):
    group_sorted = group.sort_values("time")
    flux_values = group_sorted["flux_norm"].values
    labels = group_sorted["label"].values

    for i in range(len(flux_values) - SEQ_LEN):
        X.append(flux_values[i:i+SEQ_LEN])
        y.append(labels[i+SEQ_LEN-1])  # predict label for last point in sequence

X = np.array(X).reshape(-1, SEQ_LEN, 1)
y = np.array(y)
print(f"Prepared {len(X)} sequences")

# -------------------------------
# 4. Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train sequences: {len(X_train)}, Test sequences: {len(X_test)}")

# -------------------------------
# 5. Build LSTM classifier
# -------------------------------
model = Sequential([
    LSTM(32, input_shape=(SEQ_LEN, 1)),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# -------------------------------
# 6. Train model
# -------------------------------
model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=EPOCHS, batch_size=BATCH_SIZE)

# -------------------------------
# 7. Save model
# -------------------------------
model.save(MODEL_PATH)
print(f"✅ Trained LSTM classifier saved to {MODEL_PATH}")
