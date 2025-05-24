# -*- coding: utf-8 -*-
"""
Created on Sat May 24 17:15:08 2025

@author: shmitra
"""

# laser_fwhm_vs_harmonic.py

import numpy as np
import math
import matplotlib.pyplot as plt
from lmfit.models import VoigtModel, Model
from laser_spectrum import load_laser_spectrum, HARDCODED_LASER_PATH

def compute_voigt_fwhm(x, y):
    model = VoigtModel()
    center_guess = x[np.argmax(y)]
    amp_guess = np.max(y)
    sigma_guess = (x.max() - x.min()) / 20

    params = model.make_params(amplitude=amp_guess, center=center_guess, sigma=sigma_guess, gamma=1.0)
    result = model.fit(y, x=x, params=params)

    sigma = result.params["sigma"].value
    gamma = result.params["gamma"].value
    fwhm = 0.5346 * 2 * gamma + np.sqrt(0.2166 * (2 * gamma) ** 2 + (2.3548 * sigma) ** 2)
    return fwhm

def plot_laser_fwhm_vs_harmonic(fit_type, laser_path=HARDCODED_LASER_PATH):
    
   # fit_type= input("Enter fit type, exp_decay, inv_sq, inv_sqrt: ")
    harmonics, _ = load_laser_spectrum(laser_path)
    orders = []
    fwhms = []

    for n, (wl, I) in harmonics.items():
        wl = np.array(wl)
        I = np.maximum(I, 1e-6)
        fwhm = compute_voigt_fwhm(wl, I)
        orders.append(n)
        fwhms.append(fwhm)
        print(f"Harmonic {n}: FWHM = {fwhm:.2f} nm")

    orders = np.array(orders)
    fwhms = np.array(fwhms)

    
    def exponential_decay(n, A, B):
        return A * np.exp(-B * n)
    
        # Old model: A / n^2
    def inverse_square(n, A):
        return A / n**2
    
        # New model: A / sqrt(n)
    def inverse_sqrt(n, A):
        return A / np.sqrt(n)
    
    if fit_type == "exp_decay":
        exp_model = Model(exponential_decay)
        result = exp_model.fit(fwhms, n=orders, A=fwhms[0], B=0.1)  # Initial guess
        
        A_fit = result.best_values["A"]
        B_fit = result.best_values["B"]
        fwhm_fit = exponential_decay(orders, A_fit, B_fit)
        label = f'Fit: A·e^(-Bn)\nA={result.best_values["A"]:.2f}, B={result.best_values["B"]:.2f}'

        
    elif fit_type=="inv_sq":
        inv_sq_model = Model(inverse_square)
        result = inv_sq_model.fit(fwhms, n=orders, A=fwhms[0])
        A_fit = result.best_values["A"]
        fwhm_fit = inverse_square(orders, A_fit)
        label = f'Fit: A/n²\nA={result.best_values["A"]:.2f}'
                
    else:
        inv_sqrt_model = Model(inverse_sqrt)
        result = inv_sqrt_model.fit(fwhms, n=orders, A=fwhms[0])
        A_fit = result.best_values["A"]
        fwhm_fit = inverse_sqrt(orders, A_fit)
        label = f'Fit: A/math.sqrt(n)\nA={result.best_values["A"]:.2f}'
    
    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(orders, fwhms, 'o', label='Voigt FWHM (data)', markersize=8)
    plt.plot(orders, fwhm_fit, '--', label=label, linewidth=2)
    plt.xlabel("Harmonic Order (n)")
    plt.ylabel("FWHM [nm]")
    plt.title("Laser Spectrum: FWHM vs Harmonic Order")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        plot_laser_fwhm_vs_harmonic(fit_type=sys.argv[1])
    else:
        print("Usage: python laser_fwhm_vs_harmonic.py [exp_decay | inv_sq | inv_sqrt]")

