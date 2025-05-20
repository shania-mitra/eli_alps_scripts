# -*- coding: utf-8 -*-
"""
Created on Mon May 19 09:32:44 2025

@author: shmitra
"""

# --- cli_main.py ---

from spectra_io import discover_files
from plotting import plot_selected_samples, plot_selected_samples_norm_max
from fitting import gaussian_fit, lorentzian_fit, voigt_fit, plot_gaussian_overlay
from colors import assign_colors_for_plot
from gui_plot import launch_gui
from baseline_correction import baseline_als
import sys
import datetime

base_path = r"C:\\Users\\shmitra\\Nextcloud\\1uanalysis"

def print_help(show_filelist=False):
    print("""
cli_main.py – Flexible Spectra Comparison Tool

USAGE:
  python cli_main.py -compare [min_wl max_wl] sample1 sample2 ... [-gauss FWHM AMP CENTER MIN MAX LABEL] [-lin -log] [-no_norm -baseline]
  python cli_main.py -compare_norm_max min max sample1 sample2 ... [-r -g -b] [-no_norm -baseline]
  python cli_main.py -gaussian_fit min max sample
  python cli_main.py -lorentzian_fit min max sample
  python cli_main.py -voigt_fit min max sample
  python cli_main.py -plot_gaussian FWHM AMP CENTER MIN MAX LABEL
  python cli_main.py -gui

EXAMPLES:
  python cli_main.py -compare m1.1_NOBP_1500 m1.2_NOBP_1500
  python cli_main.py -compare 450 1100 m1.1_NOBP_1500 -gauss 80 4e4 780 870 m1.1_NOBP_1500
  python cli_main.py -compare_norm_max 600 800 m1.1_NOBP_1500 m1.2_NOBP_1500
  python cli_main.py -compare 200 1100 ZnOref_NOBP_1500 -gauss 100 4e4 550 500 600 ZnOref_NOBP_1500 -lin

NOTES:
  • All spectra are normalized by integration time unless -no_norm is specified
  • Wavelength cropping is optional
  • Gaussian overlays can be added using -gauss FWHM AMP CENTER MIN MAX LABEL
  • You can add multiple -gauss blocks to overlay several Gaussians
  • Use -baseline to apply ALS baseline correction before plotting
""")

    if show_filelist:
        file_map = discover_files(base_path)
        print("\nAvailable Samples:\n")
        for label in file_map:
            print(f"  {label}")

def main():
    file_map = discover_files(base_path)
    args = sys.argv[1:]

    if '-help' in args or '--help' in args or not args:
        print_help('-filelist' in args)
        return

    no_norm = '-no_norm' in args
    if no_norm:
        args.remove('-no_norm')

    apply_baseline = '-baseline' in args
    if apply_baseline:
        args.remove('-baseline')

    cmd = args[0]

    if cmd == '-compare':
        rest = args[1:]
        color_tags = []
        samples = []
        gaussians = []
        mn = mx = None

        i = 0
        while i < len(rest):
            if rest[i] == '-gauss':
                fwhm = float(rest[i+1])
                amp = float(rest[i+2])
                center = float(rest[i+3])
                mn = float(rest[i+4])
                mx = float(rest[i+5])
                label = rest[i+6]
                gaussians.append({"fwhm": fwhm, "amplitude": amp, "center": center, "min_wl": mn, "max_wl": mx, "label": label})
                i += 7
            elif rest[i].startswith('-'):
                color_tags.append(rest[i])
                i += 1
            else:
                samples.append(rest[i])
                i += 1

        plot_selected_samples(samples, color_tags, range_min=mn, range_max=mx, no_norm=no_norm, gaussian_overlays=gaussians, apply_baseline=apply_baseline)

    elif cmd == '-compare_norm_max':
        mn, mx = float(args[1]), float(args[2])
        samples = [a for a in args[3:] if not a.startswith('-')]
        color_tags = [a for a in args[3:] if a.startswith('-')]
        plot_selected_samples_norm_max(samples, mn, mx, color_tags=color_tags, no_norm=no_norm, apply_baseline=apply_baseline)

    elif cmd == '-gaussian_fit':
        gaussian_fit(float(args[1]), float(args[2]), args[3], no_norm)

    elif cmd == '-lorentzian_fit':
        lorentzian_fit(float(args[1]), float(args[2]), args[3], no_norm)

    elif cmd == '-voigt_fit':
        voigt_fit(float(args[1]), float(args[2]), args[3], no_norm)

    elif cmd == '-plot_gaussian':
        fwhm, amp, center, mn, mx = map(float, args[1:6])
        label = args[6]
        plot_gaussian_overlay(fwhm, amp, center, mn, mx, label, no_norm)

    elif cmd == '-gui':
        launch_gui()

    else:
        print("Invalid command. Use -help for options.")

if __name__ == '__main__':
    main()


