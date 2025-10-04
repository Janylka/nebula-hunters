import numpy as np
import matplotlib.pyplot as plt

ML_DIR = "../data/ml_ready"
X = np.load(f"{ML_DIR}/ml_light_curves.npy")
y = np.load(f"{ML_DIR}/ml_labels.npy")
tic_ids = np.load(f"{ML_DIR}/ml_tic_ids.npy")

# Take the first window for demonstration
features = X[0]  # shape: (WINDOW_SIZE, 4)
labels = y[0]    # shape: (WINDOW_SIZE,)
tic = tic_ids[0]

flux_norm = features[:, 0]
flux_minmax = features[:, 1]
flux_smooth = features[:, 2]
flux_deriv = features[:, 3]

plt.figure(figsize=(14, 8))

plt.subplot(4, 1, 1)
plt.plot(flux_norm, label='Normalized Flux')
plt.title(f'TIC {tic} - Normalized Flux')
plt.ylabel('flux_norm')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(flux_minmax, label='Min-Max Flux', color='orange')
plt.title('Min-Max Scaled Flux')
plt.ylabel('flux_minmax')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(flux_smooth, label='Smoothed Flux', color='green')
plt.title('Smoothed Flux')
plt.ylabel('flux_smooth')
plt.legend()

plt.subplot(4, 1, 4)
plt.plot(flux_deriv, label='Flux Derivative', color='red')
plt.fill_between(range(len(labels)), -0.5, 0.5, where=labels==1, color='yellow', alpha=0.3, label='Transit')
plt.title('Flux Derivative + Transit Labels')
plt.ylabel('flux_deriv')
plt.xlabel('Time index')
plt.legend()

plt.tight_layout()
plt.show()
