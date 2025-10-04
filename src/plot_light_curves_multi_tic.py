import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------
# 1. Config
# -------------------------------
INPUT_PATH = "../data/processed/all_light_curves_full.csv"

# -------------------------------
# 2. Load dataset
# -------------------------------
print("🚀 Loading full light curves dataset...")
df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} rows from {INPUT_PATH}")

# -------------------------------
# 3. Function to plot multiple TICs
# -------------------------------
def plot_multiple_tics(tic_ids, df, max_points=2000):
    """
    Plot light curves (history + predictions) for multiple TICs on one graph.
    Args:
        tic_ids: list of TIC IDs to visualize
        df: DataFrame with 'time', 'flux_norm', 'predicted_flux_norm'
        max_points: max points per TIC to display (downsampling for speed)
    """
    fig = go.Figure()

    for tic_id in tic_ids:
        df_tic = df[df["tic_id"] == tic_id].sort_values("time")

        # Downsample if too many points
        if len(df_tic) > max_points:
            df_tic = df_tic.iloc[:: len(df_tic) // max_points]

        # History curve
        fig.add_trace(go.Scatter(
            x=df_tic["time"],
            y=df_tic["flux_norm"],
            mode="lines+markers",
            name=f"TIC {tic_id} - History",
            line=dict(color="blue"),
            marker=dict(size=4, color=df_tic["flux_norm"], colorscale="Blues"),
            hovertemplate=(
                f"TIC {tic_id}<br>"
                "Time: %{x:.2f}<br>"
                "Flux (norm): %{y:.4f}<extra></extra>"
            )
        ))

        # Prediction curve
        if "predicted_flux_norm" in df_tic.columns:
            df_pred = df_tic.dropna(subset=["predicted_flux_norm"])
            if not df_pred.empty:
                fig.add_trace(go.Scatter(
                    x=df_pred["time"],
                    y=df_pred["predicted_flux_norm"],
                    mode="lines+markers",
                    name=f"TIC {tic_id} - Prediction",
                    line=dict(color="red", dash="dot"),
                    marker=dict(size=4, color=df_pred["predicted_flux_norm"], colorscale="Reds"),
                    hovertemplate=(
                        f"TIC {tic_id} (Pred)<br>"
                        "Time: %{x:.2f}<br>"
                        "Predicted Flux: %{y:.4f}<extra></extra>"
                    )
                ))

    # Layout
    fig.update_layout(
        title="Light Curves for Multiple TICs (History + Predictions)",
        xaxis_title="Time",
        yaxis_title="Normalized Flux",
        template="plotly_white",
        legend=dict(title="Legend", orientation="h", y=-0.2),
        hovermode="closest"
    )

    fig.show()

# -------------------------------
# 4. Example usage
# -------------------------------
if __name__ == "__main__":
    # Pick random 3 TICs for demo
    example_tics = df["tic_id"].dropna().unique()[:3]
    print(f"✅ Example TICs: {example_tics}")
    plot_multiple_tics(example_tics, df)
