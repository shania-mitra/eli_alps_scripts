# laser_spectrum.py

import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import VoigtModel

def load_laser_spectrum(path):
    wavelength = []
    intensity = []

    with open(path, 'r') as f:
        for row in f:
            row = row.strip().split(',')
            if len(row) < 2:
                continue
            try:
                wavelength.append(float(row[0]))
                intensity.append(float(row[1]))
            except ValueError:
                continue  # Skip non-numeric rows

    wavelength = np.array(wavelength)
    intensity = np.array(intensity)

    harmonics = {}
    for n in range(1, 7):
        wl_n = wavelength / n
        harmonics[n] = (wl_n, intensity.copy())

    return harmonics, intensity


def fit_and_plot_laser_harmonic_fwhm(path, plot_fit_each=False):
    harmonics, _ = load_laser_spectrum(path)

    fwhms = []
    orders = []

    for n, (wl, intensity) in harmonics.items():
        x = wl
        y = intensity

        # Basic filtering (optional)
        y = np.maximum(y, 1e-6)

        model = VoigtModel()
        center_guess = x[np.argmax(y)]
        amp_guess = np.max(y)
        sigma_guess = (x.max() - x.min()) / 20

        params = model.make_params(amplitude=amp_guess, center=center_guess, sigma=sigma_guess, gamma=1.0)
        result = model.fit(y, x=x, params=params)

        sigma = result.params["sigma"].value
        gamma = result.params["gamma"].value
        fwhm = 0.5346 * 2 * gamma + np.sqrt(0.2166 * (2 * gamma) ** 2 + (2.3548 * sigma) ** 2)
        
        fwhms.append(fwhm)
        orders.append(n)

        if plot_fit_each:
            plt.figure()
            plt.plot(x, y, 'b.', label=f'Harmonic {n} data')
            plt.plot(x, result.best_fit, 'r-', label='Voigt Fit')
            plt.xlabel("Wavelength [nm]")
            plt.ylabel("Intensity")
            plt.title(f"Harmonic {n} Fit")
            plt.legend()
            plt.tight_layout()
            plt.show()

    # Final FWHM vs Harmonic Order Plot
    plt.figure()
    plt.plot(orders, fwhms, 'o-', label="FWHM (Voigt)")
    plt.xlabel("Harmonic Order")
    plt.ylabel("FWHM [nm]")
    plt.title("Laser Spectrum: FWHM vs Harmonic Order")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return list(zip(orders, fwhms))


# Default path used in GUI:
HARDCODED_LASER_PATH = "20240212_Spect_BaF2_520_Si_650_-18.8k_-110k_5.8W_4CM_afternoon.txt"
