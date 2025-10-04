import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
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

print(f"Total sequences: {len(sequences)}")

X = np.array([s[0] for s in sequences])
y = np.array([s[1] for s in sequences])
X = X.reshape((X.shape[0], X.shape[1], 1))

# -------------------------------
# 3. Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

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
model.summary()

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
# 7. Evaluate model
# -------------------------------
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.5f}, Test MAE: {mae:.5f}")

# -------------------------------
# 8. Save final model
# -------------------------------
model.save("lstm_final_model.h5")
print("✅ LSTM model saved as lstm_final_model.h5")

# -------------------------------
# 9. Predict and visualize some sequences
# -------------------------------
num_examples = 5  # How many examples to visualize
example_indices = np.random.choice(len(X_test), num_examples, replace=False)

plt.figure(figsize=(15, 10))
for i, idx in enumerate(example_indices):
    input_seq = X_test[idx].reshape(1, SEQ_LEN, 1)
    true_value = y_test[idx]
    pred_value = model.predict(input_seq)[0][0]

    plt.subplot(num_examples, 1, i + 1)
    plt.plot(range(SEQ_LEN), X_test[idx].flatten(), label='Input Sequence', color='blue')
    plt.scatter(SEQ_LEN, true_value, label='True Next Flux', color='green')
    plt.scatter(SEQ_LEN, pred_value, label='Predicted Next Flux', color='red')
    plt.legend()
    plt.xlabel('Time step')
    plt.ylabel('Normalized flux')
    plt.title(f'Example {i+1} (TIC sample)')

plt.tight_layout()
plt.show()
