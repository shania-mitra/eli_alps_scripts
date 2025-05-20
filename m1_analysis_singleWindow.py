import pandas as pd
import matplotlib.pyplot as plt
import io
import sys
import os
import datetime

# List of your 5 file paths
file_paths = [
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m1.1\NOBP\Int_3000mW_m11_aq_30000ms_CEP_0_NOBP_ID_621.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m1.2\NOBP\Int_3000mW_m12_aq_15000ms_CEP_0_ID_804.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m1.3\NOBP\Int_3000mW_m13_aq_15000ms_CEP_0_ID_817.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m1.4\NOBP\Int_3000mW_m14_aq_15000ms_CEP_0_ID_859.TXT",
    r"C:\Users\shmitra\Nextcloud\1uanalysis\m1.5\NOBP\Int_3000mW_m15_aq_60000ms_CEP_0_NOBP_ID_639.TXT"
]

# Short titles for each curve
file_labels = [
    "Sample m1.1",
    "Sample m1.2",
    "Sample m1.3",
    "Sample m1.4",
    "Sample m1.5"
]

def print_help():
    help_message = """
    m1_analysis_singleWindow.py - Spectra Analysis Script

    Description:
    This script loads 5 spectrometer measurement files,
    extracts wavelength and scope corrected intensity,
    divides intensity by integration time (from file header),
    replaces negative/zero values with 1e-8,
    and plots all corrected spectra on a single figure
    with a logarithmic y-axis.

    Usage:
      python m1_analysis_singleWindow.py

    Options:
      -help         Show this help message and exit.

    Notes:
    - Provide absolute paths to your data files inside the script.
    - Y-axis is normalized to acquisition (integration) time [Scope Corrected / ms].
    - No further normalization (e.g., maximum scaling) is applied.
    - Negative or zero values are replaced with 1e-8 to allow log plotting.
    - List of acquisition (integration) times and filenames will be displayed before plotting.
    """
    print(help_message)

def read_scope_corrected(filepath):
    """Reads a file, extracts integration time, wavelength and scope corrected columns."""
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

    # Read the data into a DataFrame
    data_text = "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(data_text), sep=';', header=None)

    # Extract first (wavelength) and last (scope corrected) columns
    wavelengths = df.iloc[:, 0]
    scope_corrected = df.iloc[:, -1]

    return wavelengths, scope_corrected, integration_time

def main():
    # Check if user asks for help
    if len(sys.argv) > 1 and sys.argv[1] in ["-help", "--help"]:
        print_help()
        sys.exit(0)

    # Read all data first
    all_data = [read_scope_corrected(path) for path in file_paths]

    # --- Print Information before plotting ---
    print("\n========== Script Execution Info ==========")
    today = datetime.date.today()
    print(f"Date: {today.strftime('%Y-%m-%d')}\n")

    print("List of Absolute Paths:")
    for path in file_paths:
        print(f"  {path}")
    
    print("\nList of File Names:")
    for path in file_paths:
        print(f"  {os.path.basename(path)}")

    print("\nIntegration Times [ms]:")
    for label, (_, _, integration_time) in zip(file_labels, all_data):
        print(f"  {label}: {integration_time:.2f} ms")
    print("===========================================\n")

    # --- Single Plot: All datasets on one axes ---
    plt.figure(figsize=(10, 6))

    for idx, (wavelengths, scope_corrected, integration_time) in enumerate(all_data):
        label = file_labels[idx]

        # Correct for integration time
        corrected_scope = scope_corrected / integration_time

        # Replace negative or zero values with 1e-8
        corrected_scope = corrected_scope.clip(lower=1e-8)

        # Plot
        plt.plot(wavelengths, corrected_scope, label=label, linestyle='-')

    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Scope Corrected / Integration Time [counts/ms]")
    plt.title("Spectra Comparison (Corrected by Integration Time, Log Scale Y)")
    plt.yscale('log')  # Logarithmic y-axis
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
