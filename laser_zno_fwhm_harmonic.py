# -*- coding: utf-8 -*-
"""
Created on Mon May 26 18:12:44 2025

@author: shmitra
"""
import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import Model

# --- Load Data ---
harmonics = []
laser_fwhm = []
zno_fwhm = []

with open("ZnO_laser_FWHM_laser.txt", "r") as f:
    for line in f:
        parts = line.strip().split()
        if not parts or parts[0].lower() == "n/a":
            continue
        try:
            n = int(parts[0])
        except ValueError:
            continue

        laser = float(parts[1]) if len(parts) > 1 and parts[1].lower() != "n/a" else np.nan
        zno = float(parts[2]) if len(parts) > 2 and parts[2].lower() != "n/a" else np.nan

        harmonics.append(n)
        laser_fwhm.append(laser)
        zno_fwhm.append(zno)

harmonics = np.array(harmonics)
laser_fwhm = np.array(laser_fwhm)
zno_fwhm = np.array(zno_fwhm)

# --- Filter NaNs for fitting ---
def get_valid_fit_data(x, y):
    x = np.array(x)
    y = np.array(y)
    mask = ~np.isnan(y)
    return x[mask], y[mask]

# --- Inverse-n model ---
def inverse_n(n, A): return A / n
inv_model = Model(inverse_n)

# --- Fit laser ---
n_laser, y_laser = get_valid_fit_data(harmonics, laser_fwhm)
result_laser = inv_model.fit(y_laser, n=n_laser, A=y_laser[0])
fit_laser = inverse_n(np.linspace(min(n_laser), max(n_laser), 300), **result_laser.best_values)

# --- Fit ZnO ---
n_zno, y_zno = get_valid_fit_data(harmonics, zno_fwhm)
result_zno = inv_model.fit(y_zno, n=n_zno, A=y_zno[0])
fit_zno = inverse_n(np.linspace(min(n_zno), max(n_zno), 300), **result_zno.best_values)

# --- Plot ---
plt.figure(figsize=(8, 5))

# Raw data
plt.plot(harmonics, laser_fwhm, 'o', label="Laser FWHM", color="#D55E00")
plt.plot(harmonics, zno_fwhm, 's', label="ZnO FWHM", color="#0072B2")

# Fit lines
plt.plot(np.linspace(min(n_laser), max(n_laser), 300), fit_laser, '-', color="#D55E00",
         label=f"Laser Fit (A/n), A={result_laser.best_values['A']:.2f}")
plt.plot(np.linspace(min(n_zno), max(n_zno), 300), fit_zno, '--', color="#0072B2",
         label=f"ZnO Fit (A/n), A={result_zno.best_values['A']:.2f}")
plt.semilogx()
plt.semilogy()
plt.xlabel("Harmonic Order")
plt.ylabel("FWHM [nm]")
plt.title("FWHM vs Harmonic Order with Inverse-n Fit")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# --- Save FWHM data to separate CSV files ---
np.savetxt("laser_fwhm.csv", 
           np.column_stack((harmonics, laser_fwhm)), 
           delimiter=",", 
           header="Harmonic_Order,Laser_FWHM_nm", 
           comments='', 
           fmt=["%d", "%.6f"])

np.savetxt("zno_fwhm.csv", 
           np.column_stack((harmonics, zno_fwhm)), 
           delimiter=",", 
           header="Harmonic_Order,ZnO_FWHM_nm", 
           comments='', 
           fmt=["%d", "%.6f"])


