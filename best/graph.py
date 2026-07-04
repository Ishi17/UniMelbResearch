import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def calculate_r_squared(y_actual, y_predicted):
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)

    if ss_tot == 0:
        return 0

    return 1 - (ss_res / ss_tot)


def get_best_fit(x, y):
    fit_results = []

    # Linear fit: y = ax + b
    linear_coeffs = np.polyfit(x, y, 1)
    linear_model = np.poly1d(linear_coeffs)
    linear_pred = linear_model(x)
    linear_r2 = calculate_r_squared(y, linear_pred)

    fit_results.append({
        "name": "Linear",
        "r2": linear_r2,
        "model": linear_model,
        "equation": f"y = {linear_coeffs[0]:.4f}x + {linear_coeffs[1]:.4f}"
    })

    # Quadratic fit: y = ax² + bx + c
    quadratic_coeffs = np.polyfit(x, y, 2)
    quadratic_model = np.poly1d(quadratic_coeffs)
    quadratic_pred = quadratic_model(x)
    quadratic_r2 = calculate_r_squared(y, quadratic_pred)

    fit_results.append({
        "name": "Quadratic",
        "r2": quadratic_r2,
        "model": quadratic_model,
        "equation": (
            f"y = {quadratic_coeffs[0]:.4f}x² "
            f"+ {quadratic_coeffs[1]:.4f}x "
            f"+ {quadratic_coeffs[2]:.4f}"
        )
    })

    # Exponential fit: y = ae^(bx)
    # Only valid if all y-values are positive.
    if np.all(y > 0):
        log_y = np.log(y)
        exp_coeffs = np.polyfit(x, log_y, 1)

        b = exp_coeffs[0]
        a = np.exp(exp_coeffs[1])

        exp_pred = a * np.exp(b * x)
        exp_r2 = calculate_r_squared(y, exp_pred)

        fit_results.append({
            "name": "Exponential",
            "r2": exp_r2,
            "model": lambda x_fit: a * np.exp(b * x_fit),
            "equation": f"y = {a:.4f}e^({b:.4f}x)"
        })

    best_fit = max(fit_results, key=lambda item: item["r2"])

    print("\nFit comparison:")
    for fit in fit_results:
        print(f"{fit['name']}: R² = {fit['r2']:.4f}")

    print(f"\nBest fit: {best_fit['name']} with R² = {best_fit['r2']:.4f}")

    return best_fit
def plot_layer_summary(
    summary_df,
    metric="mean_accuracy",
    error_col="sd_accuracy",
    save_file=None
):
    plt.figure(figsize=(9, 5))

    x = summary_df["layers"].to_numpy()
    y = summary_df[metric].to_numpy()
    yerr = summary_df[error_col].to_numpy()

    # Plot actual mean values with standard deviation error bars.
    # linestyle="none" prevents connecting every dot.
    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=5,
        linestyle="none",
        label="Mean result ± SD"
    )

    # Find best fitting curve.
    best_fit = get_best_fit(x, y)

    # Set fixed x-axis domain.
    # The curve will be shown from 0 to 10.5,
    # but the x-axis itself will not keep auto-expanding.
    x_axis_min = 0
    x_axis_max = 10.5

    # Smooth x-values for fitted curve across the full displayed domain.
    x_fit = np.linspace(x_axis_min, x_axis_max, 500)
    y_fit = best_fit["model"](x_fit)

    # Plot best-fit curve.
    plt.plot(
        x_fit,
        y_fit,
        linestyle="-",
        label=f"{best_fit['name']} best fit"
    )

    # Find maximum point on fitted curve within the displayed domain.
    max_index = np.argmax(y_fit)
    max_x = x_fit[max_index]
    max_y = y_fit[max_index]

    # Add dotted vertical line at maximum fitted value.
    plt.axvline(
        x=max_x,
        linestyle=":",
        linewidth=2,
        label=f"Max fit at {max_x:.2f} layers"
    )

    # Add marker at maximum fitted value.
    plt.scatter(
        [max_x],
        [max_y],
        marker="x",
        s=80,
        label=f"Max {metric}: {max_y:.4f}"
    )

    # Equation text
    equation_text = (
        f"{best_fit['name']} fit\n"
        f"{best_fit['equation']}\n"
        f"R² = {best_fit['r2']:.4f}\n"
        f"Max at x = {max_x:.2f}, y = {max_y:.4f}"
    )

    # Place equation box manually: lower and starting around x = 3.
    text_x = 3.0

    # Use full y-values including the fitted curve to place the box safely.
    combined_y = np.concatenate([y, y_fit])
    y_range = combined_y.max() - combined_y.min()
    text_y = combined_y.min() + 0.18 * y_range

    plt.text(
        text_x,
        text_y,
        equation_text,
        horizontalalignment="left",
        verticalalignment="center",
        bbox=dict(
            facecolor="white",
            edgecolor="black",
            boxstyle="round,pad=0.4",
            alpha=0.85
        )
    )

    plt.title(f"Number of GRU Layers vs {metric}")
    plt.xlabel("Number of GRU Layers")
    plt.ylabel(metric)
    plt.grid(True)

    # Fix the x-axis so it shows the full intended domain.
    plt.xlim(x_axis_min, x_axis_max)

    # Legend/key in top-right corner.
    plt.legend(loc="upper right")

    plt.tight_layout()

    if save_file is not None:
        plt.savefig(save_file, dpi=300)

    plt.show()
summary_df = pd.read_csv("natural_layer_sweep_summary_results.csv")

plot_layer_summary(
    summary_df,
    metric="mean_accuracy",
    error_col="sd_accuracy",
    save_file="natural_layers_accuracy_best_fit.png"
)

plot_layer_summary(
    summary_df,
    metric="mean_f1",
    error_col="sd_f1",
    save_file="natural_layers_f1_best_fit.png"
)