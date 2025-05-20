# -*- coding: utf-8 -*-
"""
Created on Mon May 19 09:36:43 2025

@author: shmitra
"""

# --- fitting.py ---

import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel
from spectra_io import read_scope_corrected, discover_files

def fit_peak(model_cls, label, range_min, range_max, no_norm=False):
    file_map = discover_files()
    if label not in file_map:
        print(f"Sample '{label}' not found.")
        return
    wl, sc, it = read_scope_corrected(file_map[label])
    corrected = sc if no_norm else (sc / it)
    mask = (wl >= range_min) & (wl <= range_max)
    x, y = wl[mask], corrected[mask]
    if len(x) == 0:
        print("No valid data in range.")
        return
    model = model_cls()
    params = model.make_params(amplitude=y.max(), center=float(x.iloc[y.argmax()]), sigma=10)
    if 'gamma' in model.param_names:
        params['gamma'].set(value=10)
    result = model.fit(y, x=x, params=params)
    print(result.fit_report())
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b.', label='Data')
    plt.plot(x, result.best_fit, 'r-', label='Fit')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Intensity')
    plt.title(f'{model_cls.__name__} Fit: {label}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plot_fit_residuals(x, y, model, result.params, title=f'{model_cls.__name__} Residuals')

def plot_fit_residuals(x, y, model, params, title='Residuals'):
    residuals = y - model.eval(params, x=x)
    plt.figure(figsize=(10, 4))
    plt.scatter(x, residuals, color='gray', label='Residuals')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Residual')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def gaussian_fit(min_wl, max_wl, label, no_norm=False):
    fit_peak(GaussianModel, label, min_wl, max_wl, no_norm)

def lorentzian_fit(min_wl, max_wl, label, no_norm=False):
    fit_peak(LorentzianModel, label, min_wl, max_wl, no_norm)

def voigt_fit(min_wl, max_wl, label, no_norm=False):
    fit_peak(VoigtModel, label, min_wl, max_wl, no_norm)

def plot_gaussian_overlay(fwhm, amplitude, center, min_wl, max_wl, label, no_norm=False):
    file_map = discover_files()
    if label not in file_map:
        print(f"Sample '{label}' not found.")
        return

    wl, sc, it = read_scope_corrected(file_map[label])
    corrected = sc if no_norm else (sc / it)

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    x = wl
    y = corrected
    gaussian = amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f"Data: {label}", linewidth=2)
    plt.plot(x, gaussian, label=f"Gaussian ($A$={amplitude}, FWHM={fwhm} nm)", color='red', linestyle='--')
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity (normalized)" if not no_norm else "Raw Intensity")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()
