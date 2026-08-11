"""
plot_safety_cost_heatmaps.py
==================================

Usage:
    python plot_safety_cost_heatmaps.py
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_HALF_LENGTH = 0.10
OMEGA0 = np.sqrt(9.81 / 0.90)
SAFETY_WEIGHT_C2 = 0.5
SAFETY_WEIGHT_C3 = 0.5
COM_X_RANGE = (-0.3, 0.3)
COM_XDOT_RANGE = (-0.3, 0.3)
GRID_N = 150


def make_grid():
    x = np.linspace(*COM_X_RANGE, GRID_N)
    xdot = np.linspace(*COM_XDOT_RANGE, GRID_N)
    return np.meshgrid(x, xdot)


def candidate1_cost(X, XDOT):
    return np.zeros_like(X)


def candidate2_cost(X, XDOT):
    xcom = X + XDOT / OMEGA0
    return SAFETY_WEIGHT_C2 * np.abs(xcom)


def candidate3_cost(X, XDOT):
    capture_point = X + XDOT / OMEGA0
    excess = np.clip(np.abs(capture_point) - BASE_HALF_LENGTH, 0.0, None) / BASE_HALF_LENGTH
    return SAFETY_WEIGHT_C3 * (excess ** 2)


if __name__ == "__main__":
    X, XDOT = make_grid()
    costs = {
        "Candidate 1 (no safety term)": candidate1_cost(X, XDOT),
        "Candidate 2 (XCoM penalty)": candidate2_cost(X, XDOT),
        "Candidate 3 (capture-point penalty)": candidate3_cost(X, XDOT),
    }

    titles = [f"{name}<br><sup>max value: {cost.max():.4f}</sup>"
              for name, cost in costs.items()]
    fig = make_subplots(rows=1, cols=3, subplot_titles=titles, horizontal_spacing=0.10)

    for i, (name, cost) in enumerate(costs.items()):
        # INDEPENDENT color scale per panel -- this is the fix. zmin/zmax
        # are each panel's OWN min/max, not shared.
        fig.add_trace(go.Heatmap(
            z=cost, x=np.linspace(*COM_X_RANGE, GRID_N), y=np.linspace(*COM_XDOT_RANGE, GRID_N),
            colorscale="Viridis", zmin=cost.min(), zmax=max(cost.max(), 1e-6),
            colorbar=dict(title="Penalty", x=1.0 - i * 0.005, len=0.9,
                          xanchor="left") if i == 2 else dict(
                          title="Penalty", x=[0.30, 0.635, 0.97][i], len=0.9),
        ), row=1, col=i + 1)

        for boundary in (-BASE_HALF_LENGTH, BASE_HALF_LENGTH):
            fig.add_shape(type="line", x0=boundary, x1=boundary,
                          y0=COM_XDOT_RANGE[0], y1=COM_XDOT_RANGE[1],
                          line=dict(color="white", dash="dash", width=1.5),
                          row=1, col=i + 1)

    fig.update_xaxes(title_text="CoM position x [m]")
    fig.update_yaxes(title_text="CoM velocity ẋ [m/s]", row=1, col=1)

    fig.update_layout(
        title="Safety-Term Cost Structure Across Candidates (Independent Color Scales)<br>"
              "<sup>Dashed lines mark the base-of-support boundary (±base_half_length). "
              "Each panel's colorbar uses its OWN scale, compare using the printed max "
              "values in each title, not color alone.</sup>",
        template="plotly_white", font=dict(size=13),
        width=1500, height=550,
        margin=dict(t=140),
    )
    fig.write_html("safety_cost_heatmaps.html")
    try:
        fig.write_image("safety_cost_heatmaps.png", scale=2)
    except Exception as e:
        print(f"(PNG export skipped, run `pip install -U kaleido`: {e})")
    print("Saved safety_cost_heatmaps.html (+ .png if kaleido available)")
    print("\nEach panel's max penalty value (for numeric comparison):")
    for name, cost in costs.items():
        print(f"  {name}: max={cost.max():.4f}")
