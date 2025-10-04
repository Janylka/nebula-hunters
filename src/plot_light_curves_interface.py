import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import random

# Load the combined CSV with predictions
CSV_FILE = "../data/processed/all_light_curves_full.csv"
df = pd.read_csv(CSV_FILE)

# Select a few random TICs for demonstration
tic_list = df['tic_id'].unique().tolist()
demo_tics = random.sample(tic_list, min(10, len(tic_list)))

# Initial TIC
current_tic = demo_tics[0]

def plot_tic(tic_id):
    """Plot historical and predicted flux for a given TIC"""
    df_tic = df[df['tic_id'] == tic_id]
    plt.clf()  # Clear previous plot
    plt.plot(df_tic['time'], df_tic['flux'], label='Historical flux', color='blue')
    if 'flux_pred' in df_tic.columns:
        plt.plot(df_tic['time'], df_tic['flux_pred'], label='Predicted flux', color='red', alpha=0.7)
    plt.title(f"TIC {tic_id}")
    plt.xlabel("Time [days]")
    plt.ylabel("Flux [normalized]")
    plt.legend()
    plt.draw()

# Create interactive figure
fig, ax = plt.subplots(figsize=(10, 5))
plt.subplots_adjust(bottom=0.25)

# Slider axis
ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
tic_slider = Slider(
    ax=ax_slider,
    label='TIC index',
    valmin=0,
    valmax=len(demo_tics)-1,
    valinit=0,
    valstep=1
)

def update(val):
    idx = int(tic_slider.val)
    plot_tic(demo_tics[idx])

tic_slider.on_changed(update)

# Initial plot
plot_tic(current_tic)

plt.show()
