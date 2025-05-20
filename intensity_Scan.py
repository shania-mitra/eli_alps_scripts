# -*- coding: utf-8 -*-
"""
Flexible intensity scan script with acquisition time normalization
and overlay of any specific reference sample.
"""

import os
import re
import io
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from spectra_io import discover_files, read_scope_corrected, get_sample_description



def print_usage():
    print("\nUsage:")
    print("  python intensity_scan.py [range_min range_max] SAMPLE_NAME [options]\n")
    print("Arguments:")
    print("  range_min range_max     Optional wavelength range in nm to crop the spectrum (e.g. 450 1100)")
    print("  SAMPLE_NAME             Base sample name (e.g. m6.1, r1.2)\n")
    print("Options:")
    print("  -sub SUBFOLDER          Subfolder name under 1uanalysis (default: NOBP)")
    print("  -ref REFERENCE_LABEL    Overlay a reference sample (e.g. m5.1_NOBP_2500)")
    print("  -no_norm                Disable acquisition time normalization\n")
    print("Examples:")
    print("  python intensity_scan.py m6.1")
    print("  python intensity_scan.py m7.1 -ref m5.1_NOBP_2500")
    print("  python intensity_scan.py m7.1 -sub BP -ref m3.1_BP_2000")
    print("  python intensity_scan.py 450 1100 m6.1")
    print("  python intensity_scan.py 450 1100 m6.1 -ref m1.1_NOBP_1000 -sub NOBP -no_norm")
    print("\nNote:")
    print("  The wavelength range (if given) can appear before or after the sample name.")
    print("  All intensity values are automatically normalized by acquisition time unless -no_norm is used.\n")
    print("  -logy                   Plot y-axis in log scale")
    print("  -liny                   Plot y-axis in linear scale (default)")




base_path = r"C:\\Users\\shmitra\\Nextcloud\\1uanalysis"


def plot_intensity_scan(sample_base, subfolder="NOBP", reference_label=None,
                        normalize=True, range_min=None, range_max=None, log_y=False, save_as=None):
    file_map = discover_files(base_path, filter_subdir=subfolder)
    powers = list(range(500, 3001, 500))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Style: spines and ticks
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis='both', which='both', length=8, width=2, labelsize=26)

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(min(powers), max(powers))

    found_any = False
    for power in powers:
        label = f"{sample_base}_{subfolder}_{power}"
        if label in file_map:
            wl, intensity, it = read_scope_corrected(file_map[label], normalize=normalize)
            if range_min and range_max:
                mask = (wl >= range_min) & (wl <= range_max)
                wl, intensity = wl[mask], intensity[mask]
            ax.plot(wl, intensity, label=f"{power} mW", color=cmap(norm(power)), linewidth=2, alpha=0.6)
            found_any = True
        else:
            print(f"[WARNING] Missing: {label}")

    if not found_any:
        print(f"[ERROR] No intensity scan data found for {sample_base} in subfolder '{subfolder}'")
        return

    if reference_label:
        ref_file_map = discover_files(base_path, filter_subdir="")
        if reference_label in ref_file_map:
            wl, ref_intensity, ref_it = read_scope_corrected(ref_file_map[reference_label], normalize=normalize)
            if range_min and range_max:
                mask = (wl >= range_min) & (wl <= range_max)
                wl, ref_intensity = wl[mask], ref_intensity[mask]
            ref_desc = get_sample_description(reference_label)
            ax.plot(wl, ref_intensity, label=f"Reference: {ref_desc}", color='black', linewidth=2, alpha=0.6)
        else:
            print(f"[WARNING] Reference sample '{reference_label}' not found.")

    ax.set_xlabel("Wavelength [nm]", fontsize=26)
    ylabel = "Spectrometer Counts (arb.units)"
    ax.set_ylabel(ylabel, fontsize=26)

    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-4)

    if range_min is not None and range_max is not None:
        ax.set_xlim(range_min, range_max)

    ax.legend(loc='best', fontsize=22)
    plt.tight_layout()

    if save_as:
        export_path = os.path.join(r"C:\\Users\\shmitra\\Nextcloud\\Master_arbeit\\Figures_Thesis\\results_discussion", f"{save_as}.pdf")
        plt.savefig(export_path, bbox_inches='tight')
        print(f"Plot saved to: {export_path}")
    else:
        plt.show()


def print_usage():
    print(r"""
intensity_Scan.py – Intensity Scan Plotter for Power Series

USAGE:
  python intensity_Scan.py [range_min range_max] SAMPLE_NAME [OPTIONS]

REQUIRED ORDER:
  [range_min range_max]   Optional, but if used must appear BEFORE sample name
  SAMPLE_NAME              Base label like m6.1, r2.3, etc.

OPTIONS (order-independent):
  -sub SUBFOLDER           Subfolder under 1uanalysis (default = NOBP)
  -ref REFERENCE_LABEL     Overlay reference trace (full label like m1.1_NOBP_2000)
  -no_norm                 Disable normalization by acquisition time
  -logy                    Use log scale on Y-axis
  -liny                    Use linear Y-axis (default)

EXAMPLES:
  python intensity_Scan.py m6.1
  python intensity_Scan.py m7.1 -ref m5.1_NOBP_2500
  python intensity_Scan.py 450 1100 m6.1
  python intensity_Scan.py 450 1100 m6.1 -ref m1.1_NOBP_1000 -sub NOBP -logy

NOTES:
  • Wavelength range must appear before sample name if used.
  • All spectra are normalized by integration time unless -no_norm is specified.
  • Default Y-axis is linear (use -logy to switch).
""")




def main():
    args = sys.argv[1:]
    if not args or '-h' in args or '--help' in args or '-help' in args:
        print_usage()
        return


    normalize = True
    if '-no_norm' in args:
        normalize = False
        args.remove('-no_norm')

    log_y = False  # Default is linear
    if '-logy' in args:
        log_y = True
        args.remove('-logy')
    elif '-liny' in args:
        log_y = False
        args.remove('-liny')

    range_min = range_max = None
    try:
        float_args = [arg for arg in args if re.match(r"^\d+(\.\d+)?$", arg)]
        if len(float_args) >= 2:
            range_min = float(float_args[0])
            range_max = float(float_args[1])
            args.remove(float_args[0])
            args.remove(float_args[1])
    except Exception as e:
        print(f"[DEBUG] Skipping wavelength range parse: {e}")

    sample = args[0]
    subfolder = "NOBP"
    reference_label = None

    if '-sub' in args:
        i = args.index('-sub')
        if i + 1 < len(args):
            subfolder = args[i + 1]

    if '-ref' in args:
        i = args.index('-ref')
        if i + 1 < len(args):
            reference_label = args[i + 1]

    plot_intensity_scan(sample, subfolder=subfolder, reference_label=reference_label,
                        normalize=normalize, range_min=range_min, range_max=range_max, log_y=log_y)

if __name__ == "__main__":
    main()

