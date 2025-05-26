import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel
from spectra_io import read_scope_corrected, discover_files
from baseline_correction import baseline_als, baseline_airpls
from colors import assign_colors_for_plot, get_sample_description, ROLE_COLORS
from plot_style import PLOT_STYLE

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
            "label": label,  # ← Add this line!
            "harmonic_order": order,
            "center": result.params["center"].value,
            "FWHM": fwhm,
            "amplitude": result.params["amplitude"].value,
            "R_squared": r_squared
        })


        # --- Plot with R² and χ² on it ---
        # Assign sample-specific color and label
        colors = assign_colors_for_plot([label])
        desc = get_sample_description(label)
        
        plt.figure(figsize=PLOT_STYLE["figsize"])
        plt.plot(x, y, '.', label='Data', color=colors[label])
        plt.plot(x, result.best_fit, '-', label=f'{model_type} Fit', color=ROLE_COLORS["fit"], linewidth=PLOT_STYLE["linewidth"])
        
        plt.title(f"Harmonic {order}: {desc}", fontsize=PLOT_STYLE["title_fontsize"])
        plt.xlabel("Wavelength [nm]", fontsize=PLOT_STYLE["xlabelsize"])
        plt.ylabel("Intensity", fontsize=PLOT_STYLE["ylabelsize"])
        
        plt.tick_params(axis='both', which='both',
                        length=PLOT_STYLE["tick_length"],
                        width=PLOT_STYLE["tick_width"],
                        labelsize=PLOT_STYLE["labelsize"])
        
        plt.legend(fontsize=PLOT_STYLE["legend_fontsize"])
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

