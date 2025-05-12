import pandas as pd
import matplotlib.pyplot as plt
import math
from getglass import *

# Load catalog
glassdata = 'glass/Optical Glass Lookup Table(Typecode Reference).csv'
catalog = load_glass_catalog(glassdata)

# List of glass names (can be up to 12)
glass_names = ["N-BK7","Fluorite2", "LaC10", "Q-FKH1S", "J-LASF02", "J-LAFH3HS", "LaC14", "N-FK5", "J-K3", "LF5HTi", "S-TIM5", "J-LASF021HS", "J-KF6", "TaF5"]  # Example; can be any length ≤ 12

glass_names = ['N-PK52A', 'FCD100', 'MP-TAF105', 'E-FL6', 'SF5', 'FCD515', 'TaFD30', 'E-FD80', 'FDS18-W', 'H-ZF11', 'E-FD4L', 'NbFD29', 'TaFD37']
# Wavelengths (nm) and corresponding catalog keys
wl_nm = [435.8, 486.1, 546.1, 587.6, 656.3]
n_keys = ["ng", "nF", "ne", "nd", "nC"]

# Determine subplot grid layout
n = len(glass_names)
cols = 5  # You can choose 2, 3, or 4 based on your preferred layout
rows = math.ceil(n / cols)

# Create figure and axes
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), constrained_layout=True)
axes = axes.flatten()  # Flatten in case of 2D array

lamb_nm = np.linspace(400,770,371)

# Loop over each glass and plot
for i, glass_name in enumerate(glass_names):
    row = catalog.loc[glass_name]
    ax = axes[i]
    
    if row.empty:
        ax.set_title(f"{glass_name} (not found)")
        ax.axis('off')
        continue
    
    try:
        n_values = [float(row[key]) for key in n_keys]
    except KeyError:
        ax.set_title(f"{glass_name} (missing keys)")
        ax.axis('off')
        continue
    
    ax.scatter(wl_nm, n_values, marker='o')
    glass_data = row
    # Formula for refractive index calculation
    formula = glass_data['Formula']
    manufacturer = glass_data['manufacturer']
    # For example, As (a4) might be relevant for other glass properties
    ncoeffs = np.array([glass_data[f'a{i}'] for i in range(1, 7)])  # if you need to use it elsewhere

    n_compute = np.array([ compute_refractive_index(formula, ncoeffs, lamb_mm, manufacturer) for lamb_mm in lamb_nm*1e-6])   

    ax.plot(lamb_nm, n_compute)
    n_mid = np.median(n_compute)
    n_delta = np.max(n_compute) - np.min(n_compute)
    ax.set_title(f"{glass_name} n={n_mid:.3f} d_n={n_delta:.3f}")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Refractive Index n")
    ax.grid(True)

# Turn off unused subplots
for j in range(n, len(axes)):
    axes[j].axis('off')

plt.suptitle("Refractive Index vs Wavelength", fontsize=16)
plt.show()

from lenspy3d import *
n_airlist = np.array([n_air(lamb_mm) for  lamb_mm in lamb_nm*1e-6])

plt.plot(lamb_nm, n_airlist)
plt.show()
