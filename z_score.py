#!/usr/bin/env python3
"""
z_score.py

Command-line tool to process spectrometer data files by label.

Usage:
    python z_score.py -normalise_int_time label1 label2 ...
    python z_score.py -noise_hist label1 label2 ...
    python z_score.py -z_score min_wavelength max_wavelength label1 label2 ...

Options:
    -normalise_int_time  Normalize "Scope Corrected for Dark" counts by the integration time and plot the spectra.
    -noise_hist          Plot histogram of raw noise (data ≤ 400 nm) showing mean & std dev.
    -z_score             Compute how many sigma the peak in a specified wavelength range is from the noise mean (≤ 400 nm), and display a histogram marking noise distribution and the peak.
"""
import argparse
import sys
import os
import io
import pandas as pd
import matplotlib.pyplot as plt
import re

# Base directory for file discovery
base_path = r"C:\Users\shmitra\Nextcloud\1uanalysis"

def print_help():
    print(r"""
z_score.py – Spectral Noise & Peak Significance Tool

USAGE:
  python z_score.py [COMMAND] [ARGS] label1 [label2 ...]

COMMANDS:
  -normalise_int_time    Normalize by integration time and plot
  -noise_hist            Plot noise histogram (λ ≤ 400 nm)
  -z_score WL_MIN WL_MAX label1 [label2 ...]
                         Z-score of peak in range vs noise

ORDER:
  For -z_score → WL_MIN WL_MAX must come directly after flag

DEFINITIONS:
  Noise: λ ≤ 400 nm      Z = (Peak − μ) / σ      Z > 3 → significant

EXAMPLES:
  python z_score.py -normalise_int_time m1.1 m2.1
  python z_score.py -noise_hist m1.1
  python z_score.py -z_score 600 700 m1.1 m2.3

NOTES:
  ✓ Labels must match data files
  ✓ Use full label: e.g. m1.1_NOBP_1500
""")





def discover_files(base_dir, filter_subdir="NOBP"):
    file_map = {}
    for root, _, files in os.walk(base_dir):
        if filter_subdir and filter_subdir not in root:
            continue
        for file in files:
            if file.startswith("Int_") and file.upper().endswith(".TXT"):
                full_path = os.path.join(root, file)
                parts = full_path.split(os.sep)
                try:
                    idx = parts.index("1uanalysis")
                    sample = parts[idx + 1]
                    subfolder = parts[idx + 2]
                    match = re.search(r"Int_(\d+)mW", file)
                    power = match.group(1) if match else "unknown"
                    label = f"{sample}_{subfolder}_{power}"
                    file_map[label] = full_path
                except (ValueError, IndexError):
                    continue
    return file_map


def read_scope_corrected(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    integration_time = None
    data_lines = []
    for line in lines:
        if line.startswith("Integration time"):
            integration_time = float(line.split(":")[1].strip())
        if ';' in line and any(c.isdigit() for c in line):
            data_lines.append(line.strip())
    if integration_time is None:
        print(f"Error: Integration time not found in {filepath}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=';', header=None)
    wavelengths = df.iloc[:, 0]
    scope_corrected = df.iloc[:, -1]
    return wavelengths, scope_corrected, integration_time


def normalize_and_plot(labels):
    file_map = discover_files(base_path)
    plt.figure()
    for label in labels:
        if label not in file_map:
            print(f"Warning: '{label}' not found.", file=sys.stderr)
            continue
        wl, sc, itime = read_scope_corrected(file_map[label])
        sc = sc.clip(lower=1e-8)
        norm = sc / itime
        plt.plot(wl, norm, label=label)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Normalized Counts')
    plt.title('Normalized Spectra')
    plt.legend()
    plt.grid(True)
    plt.show()


def noise_hist(labels):
    file_map = discover_files(base_path)
    for label in labels:
        if label not in file_map:
            print(f"Warning: '{label}' not found.", file=sys.stderr)
            continue
        wl, sc, _ = read_scope_corrected(file_map[label])
        noise = sc[wl <= 350]
        mean_n = noise.mean()
        std_n = noise.std()
        fig, ax = plt.subplots()
        ax.hist(noise, bins=200)
        ax.axvline(mean_n, linestyle='--', label=f'Mean = {mean_n:.3e}')
        ax.axvline(mean_n+std_n, linestyle='-.', label=f'Mean+STD = {(mean_n+std_n):.3e}')
        ax.axvline(mean_n-std_n, linestyle='-.', label=f'Mean-STD = {(mean_n-std_n):.3e}')
        ax.set_xlabel('Scope Corrected Counts')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Noise Histogram for {label} (≤ 400 nm)')
        ax.legend()
        plt.tight_layout()
        plt.show()


def z_score(labels, wl_min, wl_max):
    file_map = discover_files(base_path)
    for label in labels:
        if label not in file_map:
            print(f"Warning: '{label}' not found.", file=sys.stderr)
            continue
        wl, sc, _ = read_scope_corrected(file_map[label])
        # Noise distribution (≤ 400 nm)
        noise = sc[wl <= 400]
        mean_n = noise.mean()
        std_n = noise.std()
        # Signal region
        region = sc[(wl >= wl_min) & (wl <= wl_max)]
        if region.empty:
            print(f"No data in range {wl_min}-{wl_max} nm for {label}.", file=sys.stderr)
            continue
        peak = region.max()
        z = (peak - mean_n) / std_n if std_n != 0 else float('inf')
        significance = 'significant' if z > 3 else 'not significant'
        # Print results
        print(f"{label}: Peak = {peak:.3e}, Z-score = {z:.2f} ({significance})")
        # Plot histogram with peak marker
        fig, ax = plt.subplots()
        ax.hist(noise, bins=200, alpha=0.7, label='Noise (≤400 nm)')
        ax.axvline(mean_n, color='blue', linestyle='--', label=f'Mean = {mean_n:.3e}')
        ax.axvline(mean_n+std_n, color='green', linestyle='-.', label=f'Mean+STD = {(mean_n+std_n):.3e}')
        ax.axvline(mean_n-std_n, color='green', linestyle='-.', label=f'Mean-STD = {(mean_n-std_n):.3e}')
        ax.axvline(peak, color='red', linestyle='-', linewidth=2, label=f'Peak ({wl_min}-{wl_max} nm) = {peak:.3e}')
        ax.set_xlabel('Scope Corrected Counts')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Noise Histogram & Peak for {label}')
        ax.legend()
        plt.tight_layout()
        plt.show()

def print_help():
    print(r"""
z_score.py – Spectral Noise & Peak Significance Tool

USAGE:
  python z_score.py [COMMAND] [ARGS] label1 [label2 ...]

COMMANDS:
  -normalise_int_time    Normalize by integration time and plot
  -noise_hist            Plot noise histogram (λ ≤ 400 nm)
  -z_score WL_MIN WL_MAX label1 [label2 ...]
                         Z-score of peak in range vs noise

ORDER:
  For -z_score → WL_MIN WL_MAX must come directly after flag

DEFINITIONS:
  Noise: λ ≤ 400 nm      Z = (Peak − μ) / σ      Z > 3 → significant

EXAMPLES:
  python z_score.py -normalise_int_time m1.1 m2.1
  python z_score.py -noise_hist m1.1
  python z_score.py -z_score 600 700 m1.1 m2.3

NOTES:
  ✓ Labels must match data files
  ✓ Use full label: e.g. m1.1_NOBP_1500
""")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-normalise_int_time', action='store_true')
    parser.add_argument('-noise_hist', action='store_true')
    parser.add_argument('-z_score', action='store_true')
    parser.add_argument('labels', nargs='*')  # allow empty for -help
    args = parser.parse_args()

    if args.normalise_int_time:
        normalize_and_plot(args.labels)
    elif args.noise_hist:
        noise_hist(args.labels)
    elif args.z_score:
        if len(args.labels) < 3:
            print("Usage: python z_score.py -z_score min_wavelength max_wavelength label1 [label2 ...]", file=sys.stderr)
            sys.exit(1)
        try:
            wl_min = float(args.labels[0])
            wl_max = float(args.labels[1])
        except ValueError:
            print("Error: Wavelength range must be numbers.", file=sys.stderr)
            sys.exit(1)
        sample_labels = args.labels[2:]
        z_score(sample_labels, wl_min, wl_max)
    else:
        if '-help' in sys.argv:
            print_help()
        else:
            parser.print_help()



if __name__ == '__main__':
    main()



