# laser_spectrum.py

import numpy as np

def load_laser_spectrum(path):
    """
    Load laser spectrum and compute harmonic wavelength axes and intensities.

    Args:
        path (str): CSV/TXT file with two columns: wavelength[nm], intensity

    Returns:
        tuple:
            dict: harmonics[n] = (wavelength[nm], intensity)
            np.ndarray: original intensity array
    """
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

# Hardcoded path used by GUI:
HARDCODED_LASER_PATH = "20240212_Spect_BaF2_520_Si_650_-18.8k_-110k_5.8W_4CM_afternoon.txt"
