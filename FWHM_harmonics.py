# -*- coding: utf-8 -*-
"""
Created on Wed May 14 15:14:18 2025

@author: shmitra
"""

import numpy as np
import matplotlib.pyplot as plt

# --- Fundamental parameters ---
lambda0 = 3000  # nm
fwhm0 = 1000    # nm

# --- Define wavelength axis ---
wl = np.linspace(200, 5000, 5000)

def gaussian(wl, center, fwhm):
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-((wl - center) ** 2) / (2 * sigma ** 2))

# --- Plot fundamental ---
I0 = gaussian(wl, lambda0, fwhm0)

plt.figure(figsize=(10, 6))
plt.plot(wl, I0, '--', color='gray', label='Fundamental (3000 nm)', linewidth=2)

# --- Plot harmonics ---
colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
for i, n in enumerate([2, 3, 4, 5]):
    center = lambda0 / n
    fwhm_n = fwhm0 / (n ** 2)
    I_n = gaussian(wl, center, fwhm_n)
    plt.plot(wl, I_n, label=f"{n}ω ({center:.0f} nm, FWHM={fwhm_n:.1f} nm)", color=colors[i], linewidth=2)

plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Spectral Intensity (a.u.)", fontsize=14)
plt.title("Shrinking FWHM of Harmonics (Plotted in Wavelength Domain)", fontsize=15)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

