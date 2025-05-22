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
from baseline_correction import baseline_als, baseline_airpls
import pandas as pd



def fit_peak(model_cls, label, range_min, range_max, no_norm=False,
             apply_baseline=False, baseline_method="ALS", lam=1e5, p=0.01,
             air_lam=1e5, air_iter=15):

    file_map = discover_files()
    if label not in file_map:
        print(f"Sample '{label}' not found.")
        return

    wl, sc, it = read_scope_corrected(file_map[label])
    corrected = sc if no_norm else (sc / it)

    # ✅ Conditionally apply baseline correction
    if apply_baseline:
        if baseline_method == "ALS":
            baseline = baseline_als(corrected, lam=lam, p=p)
        elif baseline_method == "airPLS":
            baseline = baseline_airpls(corrected, lambda_=air_lam, itermax=air_iter)
        else:
            raise ValueError(f"Unknown baseline method: {baseline_method}")
        corrected = corrected - baseline

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

def gaussian_fit(min_wl, max_wl, label, no_norm=False,
                 apply_baseline=False, baseline_method="ALS", lam=1e5, p=0.01,
                 air_lam=1e5, air_iter=15):
    fit_peak(GaussianModel, label, min_wl, max_wl, no_norm,
             apply_baseline, baseline_method, lam, p, air_lam, air_iter)


def lorentzian_fit(min_wl, max_wl, label, no_norm=False,
                   apply_baseline=False, baseline_method="ALS", lam=1e5, p=0.01,
                   air_lam=1e5, air_iter=15):
    fit_peak(LorentzianModel, label, min_wl, max_wl, no_norm,
             apply_baseline, baseline_method, lam, p, air_lam, air_iter)


def voigt_fit(min_wl, max_wl, label, no_norm=False,
              apply_baseline=False, baseline_method="ALS", lam=1e5, p=0.01,
              air_lam=1e5, air_iter=15):
    fit_peak(VoigtModel, label, min_wl, max_wl, no_norm,
             apply_baseline, baseline_method, lam, p, air_lam, air_iter)

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
    
    
def multi_peak_fit_extract_plot(label, ranges, harmonic_orders, model_type="Gaussian",
                                 no_norm=False, apply_baseline=False, baseline_method="ALS",
                                 lam=1e5, p=0.01, air_lam=1e5, air_iter=15,
                                 save_csv=None):
    from lmfit.models import GaussianModel, LorentzianModel, VoigtModel
    import pandas as pd

    file_map = discover_files()
    if label not in file_map:
        raise ValueError(f"Sample '{label}' not found.")

    wl, sc, it = read_scope_corrected(file_map[label])
    corrected = sc if no_norm else sc / it
    if apply_baseline:
        if baseline_method == "ALS":
            corrected -= baseline_als(corrected, lam=lam, p=p)
        elif baseline_method == "airPLS":
            corrected -= baseline_airpls(corrected, lambda_=air_lam, itermax=air_iter)

    model_cls = {"Gaussian": GaussianModel, "Lorentzian": LorentzianModel, "Voigt": VoigtModel}.get(model_type)
    if not model_cls:
        raise ValueError(f"Unknown model: {model_type}")

    results = []
    for (rmin, rmax), order in zip(ranges, harmonic_orders):
        mask = (wl >= rmin) & (wl <= rmax)
        x, y = wl[mask], corrected[mask]
        if len(x) == 0:
            print(f"[Warning] No data in range {rmin}-{rmax}")
            continue

        model = model_cls()
        sigma_guess = (x.max() - x.min()) / 5
        params = model.make_params(amplitude=y.max(), center=float(x.iloc[y.argmax()]), sigma=sigma_guess)
        if "gamma" in params:
            params["gamma"].set(value=10)

        result = model.fit(y, x=x, params=params)
        sigma = result.params["sigma"].value
        fwhm = 2.3548 * sigma if model_type == "Gaussian" else (
            2 * result.params["gamma"].value if model_type == "Lorentzian" else result.params.get("fwhm", 2.3548 * sigma))

        results.append({
            "harmonic_order": order,
            "center": result.params["center"].value,
            "FWHM": fwhm,
            "amplitude": result.params["amplitude"].value
        })

        # plot fit
        plt.figure(figsize=(6, 4))
        plt.plot(x, y, 'b.', label='Data')
        plt.plot(x, result.best_fit, 'r-', label=f'{model_type} Fit')
        plt.title(f"Harmonic {order}: {label}")
        plt.xlabel("Wavelength [nm]")
        plt.ylabel("Intensity")
        plt.legend()
        plt.tight_layout()
        plt.show()

    if not results:
        print("[INFO] No successful fits.")
        return

    df = pd.DataFrame(results).sort_values("harmonic_order")

    # plot FWHM vs harmonic order
    plt.figure(figsize=(6, 4))
    plt.plot(df["harmonic_order"], df["FWHM"], 'o')
    plt.xlabel("Harmonic Order")
    plt.ylabel("FWHM [nm]")
    plt.title(f"{label} | {model_type} Fit")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"[INFO] Fit results saved to {save_csv}")


