# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 16:06:40 2025

@author: shmitra
"""
import pandas as pd
import matplotlib.pyplot as plt
import io
import sys
import os
import datetime
import numpy as np

# List of your 5 file paths
file_paths = [
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m1.1\NOBP\Int_3000mW_m11_aq_30000ms_CEP_0_NOBP_ID_621.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m1.2\NOBP\Int_3000mW_m12_aq_15000ms_CEP_0_ID_804.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m1.3\NOBP\Int_3000mW_m13_aq_15000ms_CEP_0_ID_817.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m1.4\NOBP\Int_3000mW_m14_aq_15000ms_CEP_0_ID_859.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m1.5\NOBP\Int_3000mW_m15_aq_60000ms_CEP_0_NOBP_ID_639.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\r1.1\NOBP\Int_3000mW_r11_aq_10ms_CEP_0_ID_902.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\ZnOref\NOBP\Int_1500mW_ZnOref_aq_0.2ms_CEP_0_ID_989.TXT"
    
]

# Short titles for each curve
file_labels = [
    "m1.1",
    "m1.2",
    "m1.3",
    "m1.4",
    "m1.5",
    "r1.1",
    "ZnO"
]


def print_help():
    help_message = """
    m1_analysis_singleWindowFFT.py - Spectra Analysis Script

    Description:
    This script loads 5 spectrometer measurement files,
    extracts wavelength and scope corrected intensity,
    corrects intensity by integration time from the header,
    replaces negative or zero values with 1e-8,
    and plots all corrected spectra together (logarithmic y-axis).

    It can also perform FFT on a selected sample and a specified wavelength range.

    Usage:
      python m1_analysis_singleWindowFFT.py
          --> Plot all spectra in one window

      python m1_analysis_singleWindowFFT.py -help
          --> Show this help message

      python m1_analysis_singleWindowFFT.py -fft <sample_name> <range_min> <range_max>
          --> Perform FFT (Power Spectrum) on the selected sample between given wavelength limits
          --> Example:
             python m1_analysis_singleWindowFFT.py -fft m1.1 800 1000

    Notes:
    - Provide absolute paths to your data files inside the script.
    - Y-axis is normalized to acquisition (integration) time [Scope Corrected / ms].
    - Negative or zero values are replaced with 1e-8 before plotting.
    - FFT results are plotted separately with log-log scaling (Power Spectrum vs Spatial Frequency).
    """
    print(help_message)


def read_scope_corrected(filepath):
    """Reads a file, extracts integration time, wavelength, and scope corrected columns."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # First extract integration time from header
    integration_time = None
    for line in lines:
        if line.startswith("Integration time [ms]:"):
            integration_time = float(line.split(":")[1].strip())
            break

    if integration_time is None:
        raise ValueError(f"Integration time not found in file {filepath}")

    # Find the start of the actual numeric data
    data_lines = []
    for line in lines:
        if ';' in line and any(char.isdigit() for char in line):
            data_lines.append(line.strip())

    if not data_lines:
        raise ValueError(f"No valid data found in file {filepath}")

    # Read the data into a DataFrame
    data_text = "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(data_text), sep=';', header=None)

    # Extract:
    #   - first column (wavelength)
    #   - last column (scope corrected)
    wavelengths = df.iloc[:, 0]
    scope_corrected = df.iloc[:, -1]

    return wavelengths, scope_corrected, integration_time




def perform_fft(wavelengths, corrected_scope, sample_label, range_min, range_max):
    """Performs and plots the Power Spectrum (FFT squared) for a cropped wavelength range."""
    mask = (wavelengths >= range_min) & (wavelengths <= range_max)
    
    if not mask.any():
        print(f"Error: No data points found between {range_min} and {range_max} nm.")
        return

    wavelengths_cropped = wavelengths[mask].values
    signal_cropped = corrected_scope[mask].values

    n_points = len(wavelengths_cropped)
    wavelengths_uniform = np.linspace(wavelengths_cropped.min(), wavelengths_cropped.max(), n_points)
    signal_uniform = np.interp(wavelengths_uniform, wavelengths_cropped, signal_cropped)

    fft_result = np.fft.fft(signal_uniform)
    fft_freq = np.fft.fftfreq(len(signal_uniform), d=(wavelengths_uniform[1] - wavelengths_uniform[0]))

    pos_mask = fft_freq > 0
    fft_freq = fft_freq[pos_mask]
    power_spectrum = np.abs(fft_result[pos_mask])**2

    # --- Plot Power Spectrum ---
    plt.figure(figsize=(8,6))
    plt.loglog(fft_freq, power_spectrum)
    plt.xlabel("Spatial Frequency [1/nm]")
    plt.ylabel("Power Spectrum [a.u.]")
    plt.title(f"Power Spectrum of {sample_label}\nWavelength Range: {range_min}-{range_max} nm")
    plt.grid(True)
    plt.tight_layout()

def plot_all_spectra(all_data):
    """Plots all spectra together on one plot."""
    plt.figure(figsize=(10, 6))

    for idx, (wavelengths, scope_corrected, integration_time) in enumerate(all_data):
        label = file_labels[idx]

        corrected_scope = scope_corrected / integration_time
        corrected_scope = corrected_scope.clip(lower=1e-8)

        plt.plot(wavelengths, corrected_scope, label=label, linestyle='-', alpha=0.5)

    plt.xlabel("Wavelength [nm]")
    #plt.ylim(1e-4,5e2)
    plt.ylabel("Scope Corrected / Integration Time [counts/ms]")
    plt.title("Spectra Comparison (Corrected by Integration Time, Log Scale Y)")
    plt.yscale('log')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()

def main():
    all_data = [read_scope_corrected(path) for path in file_paths]

    if len(sys.argv) > 1:
        if sys.argv[1] in ["-help", "--help"]:
            print_help()
            sys.exit(0)
        elif sys.argv[1] == "-fft":
            if len(sys.argv) != 5:
                print("Error: -fft needs arguments: sample_name, range_min, range_max")
                sys.exit(1)
            sample_name = sys.argv[2]
            range_min = float(sys.argv[3])
            range_max = float(sys.argv[4])

            if sample_name not in file_labels:
                print(f"Error: Sample name {sample_name} not found.")
                print(f"Available samples: {', '.join(file_labels)}")
                sys.exit(1)

            idx = file_labels.index(sample_name)
            wavelengths, scope_corrected, integration_time = read_scope_corrected(file_paths[idx])

            corrected_scope = scope_corrected / integration_time
            corrected_scope = corrected_scope.clip(lower=1e-8)

            # Plot combined spectra
            plot_all_spectra(all_data)
            plt.show()

            # Then plot power spectrum
            perform_fft(wavelengths, corrected_scope, sample_name, range_min, range_max)
            plt.show()

            sys.exit(0)

    # --- Default behavior: plot all spectra ---
    plot_all_spectra(all_data)
    plt.show()
    perform_fft(wavelengths, corrected_scope, sample_name, range_min, range_max)
    plt.show()

if __name__ == "__main__":
    main()
