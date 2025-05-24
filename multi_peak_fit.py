import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel
from spectra_io import read_scope_corrected, discover_files
from baseline_correction import baseline_als, baseline_airpls

def multi_peak_fit_extract_plot(label, ranges, harmonic_orders, model_type="Gaussian",
                                 no_norm=False, apply_baseline=False, baseline_method="ALS",
                                 lam=1e5, p=0.01, air_lam=1e5, air_iter=15,
                                 save_csv=True):

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
        print(f"\n--- Starting fit for harmonic {order} in range {rmin}-{rmax} ---")
        mask = (wl >= rmin) & (wl <= rmax)
        x, y = wl[mask], corrected[mask]

        if len(x) == 0:
            print(f"[Warning] No data in range {rmin}-{rmax} for harmonic {order}")
            continue

        try:
            model = model_cls()
            sigma_guess = (x.max() - x.min()) / 5
            params = model.make_params(amplitude=y.max(), center=float(x.iloc[y.argmax()]), sigma=sigma_guess)
            if "gamma" in params:
                params["gamma"].set(value=10)

            result = model.fit(y, x=x, params=params)
            print("[INFO] Fit successful")
        except Exception as e:
            print(f"[ERROR] Fit failed for harmonic {order}: {e}")
            continue

        try:
            sigma = result.params["sigma"].value
            gamma = result.params["gamma"].value if "gamma" in result.params else 0
            fwhm = (
                2.3548 * sigma if model_type == "Gaussian"
                else 2 * gamma if model_type == "Lorentzian"
                else 0.5346 * 2 * gamma + np.sqrt(0.2166 * (2 * gamma) ** 2 + (2.3548 * sigma) ** 2)
            )

            ss_res = np.sum((y - result.best_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else float('nan')
            reduced_chi2 = result.redchi

            print(f"[INFO] Harmonic {order}: Center = {result.params['center'].value:.2f} nm, FWHM = {fwhm:.2f} nm")
            print(f"        Amplitude = {result.params['amplitude'].value:.2e}, R² = {r_squared:.3f}, χ²_red = {reduced_chi2:.3e}")
        except Exception as e:
            print(f"[ERROR] Failed to calculate metrics for harmonic {order}: {e}")
            r_squared = float('nan')
            reduced_chi2 = float('nan')
            fwhm = float('nan')

        results.append({
            "harmonic_order": order,
            "center": result.params["center"].value,
            "FWHM": fwhm,
            "amplitude": result.params["amplitude"].value,
            "R_squared": r_squared
        })

        # --- Plot with R² and χ² on it ---
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
    print("\n[SUMMARY]")
    print(df)

    if save_csv:
        try:
            df.to_csv(save_csv, index=False)
            print(f"[INFO] Fit results saved to {save_csv}")
        except PermissionError:
            print(f"[ERROR] Permission denied while saving to {save_csv}. Is it open in Excel?")


       
def assign_peak_to_harmonic(peak_center, lambda0=3100, max_order=10):
    distances = {
        n: (peak_center - (lambda0 / n)) ** 2
        for n in range(2, max_order + 1)
    }
    return min(distances, key=distances.get)

