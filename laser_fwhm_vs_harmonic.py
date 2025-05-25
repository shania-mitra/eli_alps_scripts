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
from plot_style import PLOT_STYLE


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

def plot_laser_fwhm_vs_harmonic(fit_type, laser_path=HARDCODED_LASER_PATH, smooth_fit=True):

    
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
    
    def inverse_n(n, A):
        return A/n
    
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
        
    elif fit_type== "inv_n":
        inv_n_model=Model(inverse_n)
        result = inv_n_model.fit(fwhms, n=orders, A=fwhms[0])
        A_fit = result.best_values["A"]
        fwhm_fit = inverse_n(orders, A_fit)
        label = f'Fit: A/n\nA={result.best_values["A"]:.2f}'
                
    else:
        inv_sqrt_model = Model(inverse_sqrt)
        result = inv_sqrt_model.fit(fwhms, n=orders, A=fwhms[0])
        A_fit = result.best_values["A"]
        fwhm_fit = inverse_sqrt(orders, A_fit)
        label = f'Fit: A/math.sqrt(n)\nA={result.best_values["A"]:.2f}'
    
    # Plot
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])

    # Apply spine and tick formatting
    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_STYLE["tick_width"])
        
    ax.tick_params(axis='both', which='both',
                   length=PLOT_STYLE["tick_length"],
                   width=PLOT_STYLE["tick_width"],
                   labelsize=PLOT_STYLE["labelsize"])
    
    ax.plot(orders, fwhms, 'o', label='Voigt FWHM (data)', markersize=8)
    if smooth_fit:
        n_smooth = np.linspace(min(orders), max(orders), 300)
        fitted_vals = result.model.eval(params=result.params, n=n_smooth)
        ax.plot(n_smooth, fitted_vals, '--', label=label, linewidth=PLOT_STYLE["linewidth"])
    else:
        ax.plot(orders, fwhm_fit, '--', label=label, linewidth=PLOT_STYLE["linewidth"])
    ax.set_xlabel("Harmonic Order (n)", fontsize=PLOT_STYLE["xlabelsize"])
    ax.set_ylabel("FWHM [nm]", fontsize=PLOT_STYLE["ylabelsize"])
    ax.set_title("Laser Spectrum: FWHM vs Harmonic Order", fontsize=PLOT_STYLE["title_fontsize"])
    ax.grid(True)
    ax.legend(fontsize=PLOT_STYLE["legend_fontsize"])
    plt.tight_layout()
    plt.show()

def plot_sample_fwhm_vs_harmonic(fit_type="exp_decay", csv_path="multi_fit_results.csv", smooth_fit=True):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from lmfit import Model
    from plot_style import PLOT_STYLE

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {csv_path}")
        return

    orders = df["harmonic_order"].values
    fwhms = df["FWHM"].values

    def exponential_decay(n, A, B): return A * np.exp(-B * n)
    def inverse_square(n, A): return A / n**2
    def inverse_sqrt(n, A): return A / np.sqrt(n)      
    def inverse_n(n, A): return A/n
    
    model_map = {
        "exp_decay": (Model(exponential_decay), exponential_decay),
        "inv_sq": (Model(inverse_square), inverse_square),
        "inv_sqrt": (Model(inverse_sqrt), inverse_sqrt),
        "inv_n": (Model(inverse_n), inverse_n),
    }

    model, func = model_map[fit_type]
    result = model.fit(fwhms, n=orders, A=fwhms[0], B=0.1 if fit_type == "exp_decay" else None)
    label = f"Fit: {fit_type}, " + ", ".join(f"{k}={v:.2f}" for k, v in result.best_values.items())

    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_STYLE["tick_width"])
    ax.tick_params(axis='both', which='both',
                   length=PLOT_STYLE["tick_length"],
                   width=PLOT_STYLE["tick_width"],
                   labelsize=PLOT_STYLE["labelsize"])

    ax.plot(orders, fwhms, 'o', label='Sample FWHM', color='tab:orange')
    
    if smooth_fit:
        # Smooth model curve
        n_smooth = np.linspace(min(orders), max(orders), 300)
        fitted_vals = func(n_smooth, **result.best_values)
        ax.plot(n_smooth, fitted_vals, '--', label=label, color='tab:orange', linewidth=PLOT_STYLE["linewidth"])
    else:
        # Discrete model fit only at harmonic orders
        fitted_vals = func(orders, **result.best_values)
        ax.plot(orders, fitted_vals, '--', label=label, color='tab:orange', linewidth=PLOT_STYLE["linewidth"])


    ax.set_xlabel("Harmonic Order", fontsize=PLOT_STYLE["xlabelsize"])
    ax.set_ylabel("FWHM [nm]", fontsize=PLOT_STYLE["ylabelsize"])
    ax.set_title("Sample FWHM vs Harmonic Order", fontsize=PLOT_STYLE["title_fontsize"])
    ax.grid(True)
    ax.legend(fontsize=PLOT_STYLE["legend_fontsize"])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        plot_laser_fwhm_vs_harmonic(fit_type=sys.argv[1])
    else:
        print("Usage: python laser_fwhm_vs_harmonic.py [exp_decay | inv_sq | inv_sqrt | inv_n]")


