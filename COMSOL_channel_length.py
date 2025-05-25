
# -*- coding: utf-8 -*-
"""
Updated to use plot_style.py for consistent figure styling
"""

import matplotlib.pyplot as plt
import numpy as np
from plot_style import PLOT_STYLE

# Data for each harmonic
fifth_harmonic = [
    [0.1, 3.34e6],[0.5, 3.95e6], [1, 3.36e6], [5, 1.13e6],
    [10, 2.32e5], [15, 4.36e4], [25, 9.3e3], [30, 397], [50, 0.68], [100,8.72e-8],[200, 1.63e-21]
]
third_harmonic = [
    [0.1, 9.52e6], [0.5, 8.42e6], [5, 6.27e5],
    [10, 3.56e4], [15, 1.97e3], [20, 108], [25, 6.09],
    [50, 3.23e-6], [100, 8.82e-19], [200, 7.08e-44]
]
fundi_harmonic = [
    [0.1, 1.08e6], [0.5, 1.8e5], [1, 4.77e4], [2, 3.52e3], [2.5, 954], [3, 259], [4,19],
    [5, 1.4], [8, 5.57e-4], [10, 3.01e-6], [15, 6.48e-12], [20, 1e-17], [30, 1e-29], [50, 1e-51], [100, 3.01e-108], [200, 0]
]
seventh_harmonic = [
    [0.1, 3.41e6], [0.5, 2.84e6], [1, 1.86e6], [20, 3.54e4], [30, 3.35e3], [50, 32.5], [100, 4.67e-4], [200, 6.54e-12]
]

# Convert data to NumPy arrays
fifth_harmonic = np.array(fifth_harmonic)
third_harmonic = np.array(third_harmonic)
fundi_harmonic = np.array(fundi_harmonic)
seventh_harmonic = np.array(seventh_harmonic)

# Initialize plot
fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])

# Plot data
ax.plot(fundi_harmonic[:, 0], fundi_harmonic[:, 1], marker='o', label= u'3.2 $\mu$m (Fundamental)', linewidth=PLOT_STYLE["linewidth"])
ax.plot(third_harmonic[:, 0], third_harmonic[:, 1], marker='o', label= u'1.07 $\mu$m (HH3)', linewidth=PLOT_STYLE["linewidth"])
ax.plot(fifth_harmonic[:, 0], fifth_harmonic[:, 1], marker='o', label='640 nm (HH5)', linewidth=PLOT_STYLE["linewidth"])
ax.plot(seventh_harmonic[:, 0], seventh_harmonic[:, 1], marker='o', label='457 nm (HH7)', linewidth=PLOT_STYLE["linewidth"])

# Apply log scaling
ax.set_xscale('log')
ax.set_yscale('log')

# Axes labels
ax.set_xlabel(u'Length of Channel Sampled [ $\mu$m]', fontsize=PLOT_STYLE["xlabelsize"])
ax.set_ylabel('Electric Field Norm [V/m]', fontsize=PLOT_STYLE["ylabelsize"])
ax.set_ylim(1e-2)

# Styling
ax.tick_params(axis='both', which='both',
               length=PLOT_STYLE["tick_length"],
               width=PLOT_STYLE["tick_width"],
               labelsize=PLOT_STYLE["labelsize"])

for spine in ax.spines.values():
    spine.set_linewidth(PLOT_STYLE["tick_width"])

# Legend and grid
ax.legend(fontsize=PLOT_STYLE["legend_fontsize"])
ax.grid(True, which="both", linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()