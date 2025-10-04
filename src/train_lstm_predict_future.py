import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------------------
# 1. Load cleaned light curves
# -------------------------------
data_path = "../data/processed/all_light_curves_clean.csv"
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} rows from {data_path}")

# -------------------------------
# 2. Prepare sequences for LSTM
# -------------------------------
SEQ_LEN = 50
sequences = []

for tic_id, group in df.groupby('tic_id'):
    flux = group['flux_norm'].values
    if len(flux) > SEQ_LEN:
        for i in range(len(flux) - SEQ_LEN):
            seq_x = flux[i:i+SEQ_LEN]
            seq_y = flux[i+SEQ_LEN]
            sequences.append((seq_x, seq_y))

X = np.array([s[0] for s in sequences])
y = np.array([s[1] for s in sequences])
X = X.reshape((X.shape[0], SEQ_LEN, 1))

# -------------------------------
# 3. Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Build LSTM model
# -------------------------------
model = Sequential([
    LSTM(64, input_shape=(SEQ_LEN, 1), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# -------------------------------
# 5. Callbacks
# -------------------------------
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint("lstm_best_model.h5", save_best_only=True)
]

# -------------------------------
# 6. Train model
# -------------------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    callbacks=callbacks
)

# -------------------------------
# 7. Save final model
# -------------------------------
model.save("lstm_final_model.h5")
print("✅ LSTM model saved as lstm_final_model.h5")

# -------------------------------
# 8. Rolling prediction (N steps ahead)
# -------------------------------
N_FUTURE = 30  # Number of steps to predict ahead
num_examples = 3  # How many sequences to visualize

example_indices = np.random.choice(len(X_test), num_examples, replace=False)

plt.figure(figsize=(15, 8))

for i, idx in enumerate(example_indices):
    input_seq = X_test[idx].flatten().tolist()  # start sequence
    future_preds = []

    # Rolling prediction
    for _ in range(N_FUTURE):
        seq_array = np.array(input_seq[-SEQ_LEN:]).reshape(1, SEQ_LEN, 1)
        pred = model.predict(seq_array, verbose=0)[0][0]
        future_preds.append(pred)
        input_seq.append(pred)

    # Plot
    plt.subplot(num_examples, 1, i + 1)
    plt.plot(range(SEQ_LEN), X_test[idx].flatten(), label='Input Sequence', color='blue')
    plt.plot(range(SEQ_LEN, SEQ_LEN+N_FUTURE), future_preds, label='Predicted Future', color='red')
    plt.xlabel('Time step')
    plt.ylabel('Normalized flux')
    plt.title(f'Example {i+1} Rolling Prediction')
    plt.legend()

plt.tight_layout()
plt.show()
