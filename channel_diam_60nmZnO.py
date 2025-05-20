# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 17:13:25 2025

@author: shmitra
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 16:40:23 2025

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
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m1.5\NOBP\Int_3000mW_m15_aq_60000ms_CEP_0_NOBP_ID_639.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m3.5\NOBP\Int_3000mW_m35_aq_30000ms_CEP_0_NOBP_ID_595.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m4.5\NOBP\Int_3000mW_m45_aq_1000ms_CEP_0_ID_698.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m5.5\NOBP\Int_3000mW_m55_aq_500ms_CEP_0_NOBP_ID_571.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m6.5\NOBP\Int_3000mW_m65_aq_500ms_CEP_0_ID_1016.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m7.5\NOBP\Int_3000mW_m75_aq_2000ms_CEP_0_ID_1047.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m8.5\NOBP\Int_3000mW_aq_2000ms_CEP_0_NOBP_ID_527.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\r1.5\NOBP\Int_3000mW_r15_aq_10ms_CEP_0_ID_974.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\ZnOref\NOBP\Int_1500mW_ZnOref_aq_0.2ms_CEP_0_ID_989.TXT"
    
]

# Short titles for each curve
file_labels = [
    "m1.1",
    "m3.5",
    "m4.5",
    "m5.5",
    "m6.5",
    "m7.5",
    "m8.1",
    "r1.1",
    "ZnO"
]


def print_help():
    help_message = """
    channel_diam_60nmZnO.py - Spectra Analysis Script

    Description:
    This script loads spectrometer measurements for samples with different
    channel diameters coated with 60 nm ZnO. It compares their spectra
    alongside a silicon reference (r1.5). An additional ZnO reference is available
    but currently commented out.

    The data are processed by:
    - Extracting wavelength and scope corrected intensity
    - Normalizing by acquisition (integration) time from the header
    - Replacing negative or zero values with 1e-8 to enable log plotting
    - Plotting all corrected spectra together on a logarithmic y-axis
    - Optionally performing FFT (power spectrum) over a selected wavelength range

    Usage:
      python channel_diam_60nmZnO.py
          --> Plot all spectra in one window

      python channel_diam_60nmZnO.py -help
          --> Show this help message

      python m1_analysis_singleWindow_refSiZnO.py -compare sample1 sample2 ...
          --> Plot only selected samples normalized by integration time
          --> Example:
              python m1_analysis_singleWindow_refSiZnO.py -compare m1.1 m1.3 r1.1


    Notes:
    - Provide absolute paths to your data files inside the script.
    - Y-axis is normalized to integration time [Scope Corrected / ms].
    - Negative or zero values are replaced with 1e-8 before plotting.
    - FFT results are plotted separately with log-log scaling (Power Spectrum vs Spatial Frequency).

    Current Loaded Samples:
    """
    print(help_message)

    # --- Also print file list, sample names and integration times dynamically ---
    print(f"Date: {datetime.date.today()}\n")
    
    for path, label in zip(file_paths, file_labels):
        # Try to extract integration time from the file
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            integration_time = None
            for line in lines:
                if line.startswith("Integration time [ms]:"):
                    integration_time = float(line.split(":")[1].strip())
                    break
            if integration_time is None:
                integration_time = "Unknown"
        except Exception as e:
            integration_time = "Error"

        print(f"Sample: {label}")
        print(f"  File Path: {path}")
        print(f"  Integration Time: {integration_time} ms\n")


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


def plot_selected_samples(selected_labels):
    """Plots selected samples together, normalized by integration time."""
    plt.figure(figsize=(10, 6))

    for label in selected_labels:
        if label not in file_labels:
            print(f"Warning: Sample '{label}' not found in loaded samples. Skipping.")
            continue

        idx = file_labels.index(label)
        wavelengths, scope_corrected, integration_time = read_scope_corrected(file_paths[idx])

        corrected_scope = scope_corrected / integration_time
        corrected_scope = corrected_scope.clip(lower=1e-8)

        plt.plot(wavelengths, corrected_scope, label=label, linestyle='-')

    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Scope Corrected / Integration Time [counts/ms]")
    plt.title("Selected Spectra Comparison (Log Scale Y)")
    plt.yscale('log')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_all_spectra(all_data):
    """Plots all spectra together on one plot."""
    plt.figure(figsize=(10, 6))

    for idx, (wavelengths, scope_corrected, integration_time) in enumerate(all_data):
        label = file_labels[idx]

        corrected_scope = scope_corrected / integration_time
        corrected_scope = corrected_scope.clip(lower=1e-8)

        plt.plot(wavelengths, corrected_scope, label=label, linestyle='-')

    plt.xlabel("Wavelength [nm]")
    #plt.ylim(1e-4,5e2)
    plt.ylabel("Scope Corrected / Integration Time [counts/ms]")
    plt.title("Spectra Comparison (Corrected by Integration Time, Log Scale Y)")
    plt.yscale('log')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()


def main():
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

            plot_all_spectra([read_scope_corrected(path) for path in file_paths])
            plt.show()

            perform_fft(wavelengths, corrected_scope, sample_name, range_min, range_max)
            plt.show()

            sys.exit(0)

        elif sys.argv[1] == "-compare":
            if len(sys.argv) < 3:
                print("Error: After -compare, specify at least two sample names.")
                sys.exit(1)
            selected_labels = sys.argv[2:]  # Everything after -compare
            plot_selected_samples(selected_labels)
            sys.exit(0)

    # --- Default behavior: plot all spectra ---
    plot_all_spectra([read_scope_corrected(path) for path in file_paths])
    plt.show()


if __name__ == "__main__":
    main()