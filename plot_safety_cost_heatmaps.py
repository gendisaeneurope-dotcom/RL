"""
plot_safety_cost_heatmaps.py
==================================
FIXED vs. previous version:
  - BASE_HALF_LENGTH was 0.10, does not match the actual L_AP = 0.15
    (foot_geom size[0]). Corrected.
  - SAFETY_WEIGHT_C2/C3 were both 0.5, stale placeholders. Corrected to
    the seed-confirmed values: C2 = 0.90 (post A_SCALE test),
    C3 = 0.50 (original, unchanged).
  - Colorscale: "Viridis" was already colorblind-safe; kept.

Usage:
    python plot_safety_cost_heatmaps.py
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_HALF_LENGTH = 0.15   # was 0.10 -- actual L_AP from foot_geom
OMEGA0 = np.sqrt(9.81 / 0.90)
SAFETY_WEIGHT_C2 = 0.90    # was 0.5 -- seed-confirmed post A_SCALE test
SAFETY_WEIGHT_C3 = 0.50    # was 0.5 -- matches, coincidentally correct
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
        f"Candidate 2 (XCoM penalty, w={SAFETY_WEIGHT_C2})": candidate2_cost(X, XDOT),
        f"Candidate 3 (capture-point penalty, w={SAFETY_WEIGHT_C3})": candidate3_cost(X, XDOT),
    }

    titles = [f"{name}<br><sup>max value: {cost.max():.4f}</sup>"
              for name, cost in costs.items()]
    fig = make_subplots(rows=1, cols=3, subplot_titles=titles, horizontal_spacing=0.10)

    for i, (name, cost) in enumerate(costs.items()):
        fig.add_trace(go.Heatmap(
            z=cost, x=np.linspace(*COM_X_RANGE, GRID_N), y=np.linspace(*COM_XDOT_RANGE, GRID_N),
            colorscale="Viridis", zmin=cost.min(), zmax=max(cost.max(), 1e-6),
            colorbar=dict(title="Penalty", x=[0.30, 0.635, 0.97][i], len=0.9),
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
              "<sup>Dashed lines mark the base-of-support boundary (±0.15m). "
              "Each panel's colorbar uses its OWN scale, compare using the printed max "
              "values in each title, not color alone. Candidate 2 shown at its "
              "post-A_SCALE-test weight (0.90); Candidate 3 at original weight (0.50).</sup>",
        template="plotly_white", font=dict(size=13),
        width=1500, height=550,
        margin=dict(t=140),
    )
    fig.write_html("safety_cost_heatmaps.html")
    try:
        fig.write_image("safety_cost_heatmaps.png", scale=2)
    except Exception as e:
        print(f"(PNG export skipped, run pip install -U kaleido: {e})")
    print("Saved safety_cost_heatmaps.html (+ .png if kaleido available)")
    print("\nEach panel's max penalty value (for numeric comparison):")
    for name, cost in costs.items():
        print(f"  {name}: max={cost.max():.4f}")
