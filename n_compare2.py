import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

def plot_glass_dispersion(
    glass_names,
    catalog,
    wl_keys=["g", "F", "e", "d", "C"],
    wl_nm=[435.8, 486.1, 546.1, 587.6, 656.3],
    yaxis_mode="shared",
    percent_range=0.01,
    cols=3
):
    """
    Plot refractive index vs wavelength for a list of glasses.

    Parameters:
    - glass_names: list of glass name strings
    - catalog: pandas DataFrame containing refractive index columns like 'ng', 'nF', ...
    - wl_keys: list of wavelength labels (default: Fraunhofer)
    - wl_nm: corresponding list of wavelengths in nm
    - yaxis_mode: 'shared', 'relative', or 'percent'
    - percent_range: ± percentage range for 'percent' mode (default: 1%)
    - cols: number of subplot columns (default: 3)
    """
    
    n_keys = [f"n{key}" for key in wl_keys]
    n = len(glass_names)
    rows = math.ceil(n / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), constrained_layout=True)
    axes = axes.flatten()

    n_matrix = []
    for glass_name in glass_names:
        row = catalog[catalog['glass'] == glass_name]
        if row.empty:
            n_matrix.append(None)
        else:
            try:
                n_values = [float(row[key]) for key in n_keys]
                n_matrix.append(n_values)
            except KeyError:
                n_matrix.append(None)

    # Compute y-limits
    y_limits = []
    if yaxis_mode == "shared":
        all_vals = [n for nlist in n_matrix if nlist for n in nlist]
        ymin, ymax = min(all_vals), max(all_vals)
        y_limits = [(ymin, ymax)] * n

    elif yaxis_mode == "relative":
        spans = [max(nlist) - min(nlist) for nlist in n_matrix if nlist]
        span = max(spans) if spans else 0.01
        for nlist in n_matrix:
            if nlist:
                mid = np.median(nlist)
                y_limits.append((mid - span / 2, mid + span / 2))
            else:
                y_limits.append(None)

    elif yaxis_mode == "percent":
        for nlist in n_matrix:
            if nlist:
                mid = np.median(nlist)
                delta = mid * percent_range
                y_limits.append((mid - delta, mid + delta))
            else:
                y_limits.append(None)
    else:
        raise ValueError("Invalid yaxis_mode. Use 'shared', 'relative', or 'percent'.")

    # Plotting loop
    for i, (glass_name, nlist) in enumerate(zip(glass_names, n_matrix)):
        ax = axes[i]
        if nlist is None:
            ax.set_title(f"{glass_name} (data missing)")
            ax.axis('off')
            continue

        ax.plot(wl_nm, nlist, marker='o')
        ax.set_title(glass_name)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Refractive Index n")
        ax.grid(True)

        if y_limits[i]:
            ax.set_ylim(y_limits[i])

    # Turn off unused subplots
    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Refractive Index vs Wavelength", fontsize=16)
    plt.show()


if __name__ = "__main__":

    plot_glass_dispersion(
    glass_names,
    catalog,
    cols=5)





