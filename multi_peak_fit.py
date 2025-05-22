import matplotlib.pyplot as plt
import pandas as pd
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel
from spectra_io import read_scope_corrected, discover_files
from baseline_correction import baseline_als, baseline_airpls

def multi_peak_fit_extract_plot(label, ranges, harmonic_orders, model_type="Gaussian",
                                 no_norm=False, apply_baseline=False, baseline_method="ALS",
                                 lam=1e5, p=0.01, air_lam=1e5, air_iter=15,
                                 save_csv=None):

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
        fwhm = (
            2.3548 * sigma if model_type == "Gaussian"
            else 2 * result.params["gamma"].value if model_type == "Lorentzian"
            else result.params.get("fwhm", 2.3548 * sigma)
        )

        results.append({
            "harmonic_order": order,
            "center": result.params["center"].value,
            "FWHM": fwhm,
            "amplitude": result.params["amplitude"].value
        })

        # plot fit
        plt.figure(figsize=(6, 4))
        plt.scatter(x, y, 'b.', label='Data')
        plt.scatter(x, result.best_fit, color='red', label=f'{model_type} Fit', s=10)
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
    plt.scatter(x, y, color='blue', label='Data', s=10)
    plt.scatter(x, result.best_fit, color='red', label=f'{model_type} Fit', s=10)
    plt.title(f"Harmonic {order}: {label}")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity")
    plt.legend()
    plt.tight_layout()
    plt.show()



    if save_csv:
        df.to_csv_
        
def assign_peak_to_harmonic(peak_center, lambda0=3100, max_order=10):
    distances = {
        n: (peak_center - (lambda0 / n)) ** 2
        for n in range(2, max_order + 1)
    }
    return min(distances, key=distances.get)

