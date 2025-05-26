# -*- coding: utf-8 -*-
"""
Created on Mon May 26 14:05:42 2025

@author: shmitra
"""

import numpy as np
import matplotlib.pyplot as plt
from laser_spectrum import load_laser_spectrum, HARDCODED_LASER_PATH
from laser_fwhm_vs_harmonic import compute_voigt_fit

def plot_voigt_fit_of_laser(harmonic_order=1):
    harmonics, _ = load_laser_spectrum(HARDCODED_LASER_PATH)
    if harmonic_order not in harmonics:
        print(f"[ERROR] Harmonic {harmonic_order} not found in laser spectrum.")
        return

    wl, I = harmonics[harmonic_order]
    wl = np.array(wl)
    I = np.array(I)

    result = compute_voigt_fit(wl, I)

    # Compute FWHM
    sigma = result.params["sigma"].value
    gamma = result.params["gamma"].value
    fwhm = 0.5346 * 2 * gamma + np.sqrt(0.2166 * (2 * gamma)**2 + (2.3548 * sigma)**2)
    center = result.params["center"].value

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(wl, I, label="Laser Fundamental", alpha=0.5, color='gray')
    plt.plot(wl, result.best_fit, label=f"Voigt Fit\nCenter={center:.2f} nm, FWHM={fwhm:.2f} nm", color='black', linestyle='--')
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity (a.u.)")
    plt.title(f"Voigt Fit to Harmonic {harmonic_order} of Laser")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_voigt_fit_of_laser(harmonic_order=1)  # Fundamental = 1
