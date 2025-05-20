# -*- coding: utf-8 -*-
"""
Merged and Enhanced compare_spectra.py
Created: May 2025
@author: shmitra

Features:
- Integration-time normalization
- Max-in-range normalization
- Color customization
- Gaussian, Lorentzian, and Voigt fits with residual plots
- High-quality formatting
"""

import os
import sys
import re
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel
import seaborn as sns
# Add after matplotlib import
from matplotlib import cm
import matplotlib.colors as mcolors
import random
import hashlib
import colorsys
from collections import Counter


# Color tags map
tag_to_color = {
    "-r": "#d11141",  # Red
    "-g": "#228B22",  # Green
    "-b": "#3a75c4",  # Blue
    "-y": "#ffc425",  # Yellow
    "-p": "#9b19f5",  # Purple
    "-o": "#f37735",  # Orange
    "-k": "#000000",  # Black
    "-m": "#dc0ab4",  # Magenta
    "-c": "#00bfa0",  # Cyan
    "-si": "#b3d4ff"  # Sky Blue
}

base_path = r"C:\\Users\\shmitra\\Nextcloud\\1uanalysis"
file_map = {}


# --- Brand colors for each sample type ---
sample_base_colors = {
    "m1": "#000000",   # black
    "m2": "#ff7f0e",   # orange
    "m3": "#2ca02c",   # green
    "m4": "#d62728",   # red
    "m5": "#9467bd",   # purple
    "m6": "#8c564b",   # brown
    "m7": "#e377c2",   # magenta
    "m8": "#7f7f7f",   # gray
    "r1": "#bcbd22",   # yellow-green
    "r2": "#17becf",   # teal
    "ZnOref": "#444444"  # dark gray
}


def get_consistent_color(label):
    """
    Returns the fixed brand color based on sample type (e.g. m1, m2, r1, etc.).
    Multiple samples from the same type (m1.1, m1ref, m1_1500mW) share the same color.
    """
    base = label.split("_")[0]  # e.g., 'm1.2' or 'm1ref'
    
    if "ref" in base:
        key = base.replace("ref", "")
    elif "." in base:
        key = base.split(".")[0]
    else:
        key = base

    return sample_base_colors.get(key, "#777777")  # fallback gray for unknown types


# --- Color Preview Utility ---
def preview_palette_assignments():
    fig, axs = plt.subplots(3, 1, figsize=(10, 6), constrained_layout=True)

    # Different sample types
    labels1 = [f"{k}_NOBP_1500" for k in list(sample_base_colors.keys())[:6]]
    colors1 = assign_colors_for_plot(labels1)
    for i, label in enumerate(labels1):
        axs[0].bar(i, 1, color=colors1[label])
        axs[0].text(i, 1.05, label.split("_")[0], rotation=45, ha='right')
    axs[0].set_title("Different Sample Types (Base Colors)")
    axs[0].axis('off')

    # Same sample type
    labels2 = [f"m1.{i}_NOBP_1500" for i in range(1, 6)]
    colors2 = assign_colors_for_plot(labels2)
    for i, label in enumerate(labels2):
        axs[1].bar(i, 1, color=colors2[label])
        axs[1].text(i, 1.05, label.split("_")[0], rotation=45, ha='right')
    axs[1].set_title("Same Sample Type (Dark2 Palette)")
    axs[1].axis('off')

    # Intensity scan
    labels3 = [f"m1.1_NOBP_{p}" for p in [500, 1000, 1500, 2000, 2500, 3000]]
    colors3 = assign_colors_for_plot(labels3)
    for i, label in enumerate(labels3):
        axs[2].bar(i, 1, color=colors3[label])
        axs[2].text(i, 1.05, label.split("_")[2], rotation=45, ha='right')
    axs[2].set_title("Intensity Scan (Set1 Palette)")
    axs[2].axis('off')

    plt.suptitle("Color Assignment Previews", fontsize=16)
    plt.show()




def discover_files(base_dir, filter_subdir="NOBP"):
    """Automatically build a label-to-path map from folder structure."""
    file_map = {}
    for root, _, files in os.walk(base_dir):
        if filter_subdir and filter_subdir not in root:
            continue
        for file in files:
            if file.startswith("Int_") and file.endswith(".TXT"):
                full_path = os.path.join(root, file)
                parts = full_path.split(os.sep)
                try:
                    idx = parts.index("1uanalysis")
                    sample = parts[idx + 1]
                    subfolder = parts[idx + 2]

                    # Extract power from filename e.g. Int_2000mW...
                    match = re.search(r"Int_(\d+)mW", file)
                    power = match.group(1) if match else "unknown"
                    label = f"{sample}_{subfolder}_{power}"

                    file_map[label] = full_path
                except (ValueError, IndexError):
                    continue
    return file_map

def print_help(show_filelist=False):
    print("""
compare_spectra.py - Flexible Spectra Comparison Tool

This script allows you to plot and analyze spectrometer data from structured filenames. It supports:
  ✓ Normalization by integration time
  ✓ Optional max-normalization within wavelength range
  ✓ Color-coded comparison plots
  ✓ Gaussian, Lorentzian, and Voigt peak fitting with residuals

---
USAGE:

Compare (with optional wavelength range and normalization):
  python compare_spectra.py -compare [min_wl max_wl] sample1 sample2 ... [-r -g -b] [-no_norm]

Compare with normalization to maximum in range:
  python compare_spectra.py -compare_norm_max min_wl max_wl [-full] sample1 sample2 ... [-r -g -b] [-no_norm]

Fit a peak in one sample:
  python compare_spectra.py -gaussian_fit min_wl max_wl sample [-no_norm]
  python compare_spectra.py -lorentzian_fit min_wl max_wl sample [-no_norm]
  python compare_spectra.py -voigt_fit min_wl max_wl sample [-no_norm]

Other commands:
  python compare_spectra.py -help                   → Show this help
  python compare_spectra.py -filelist -help         → Show help and available samples
  python compare_spectra.py -preview_colors         → Show color palette preview for all membrane samples

---
EXAMPLES:

Compare samples with colors:
  python compare_spectra.py -compare m1.1_NOBP_1500 m1.2_NOBP_3000 -r -b

Compare within a wavelength window (400–900 nm):
  python compare_spectra.py -compare 400 900 m1.1_NOBP_1500 m1.2_NOBP_3000 -r -b

Compare and normalize to max in 600–800 nm:
  python compare_spectra.py -compare_norm_max 600 800 m1.1_NOBP_1500 m1.2_NOBP_3000 -r -b

Fit a Gaussian peak to a single sample:
  python compare_spectra.py -gaussian_fit 600 800 m1.2_NOBP_3000

---
COLOR TAGS:
  -r red   -g green   -b blue   -o orange   -y yellow
  -p purple   -c cyan   -k black   -m magenta   -si skyblue

---
NOTES:
  • Y-axis uses logarithmic scale
  • All spectra are normalized by integration time unless -no_norm is used
  • -compare_norm_max also normalizes to peak max within given wavelength range
  • Sample labels are auto-detected from directory and filenames (e.g., m1.1_NOBP_1500)
""")

    if show_filelist:
        print("\nAvailable Samples:\n")
        print(f"Date: {datetime.date.today()}\n")
        for label, path in file_map.items():
            try:
                with open(path, 'r') as f:
                    lines = f.readlines()
                integration_time = next(
                    float(line.split(":")[1].strip())
                    for line in lines if line.startswith("Integration time [ms]:")
                )
            except Exception:
                integration_time = "Error"
            print(f"  {label}: {path}\n    Integration Time: {integration_time} ms\n")


def read_scope_corrected(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    integration_time = None
    for line in lines:
        if line.startswith("Integration time [ms]:"):
            integration_time = float(line.split(":")[1].strip())
            break

    if integration_time is None:
        raise ValueError(f"Integration time not found in file {filepath}")

    data_lines = [line.strip() for line in lines if ';' in line and any(c.isdigit() for c in line)]
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=';', header=None)

    wavelengths = df.iloc[:, 0]
    scope_corrected = df.iloc[:, -1]

    return wavelengths, scope_corrected, integration_time


def get_sample_description(label):
    if "ZnOref" in label:
        return "Bulk ZnO, L=90$\\mu$m"
    if label.startswith("m"):
        base = label.split("_")[0]
        if "ref" in base:
            sample = base.replace("ref", "")
            thickness_str = "ref (no ZnO)"
        else:
            sample, thickness_idx = base.split(".")
            thickness_map = {"1": "1 nm ZnO", "2": "6 nm ZnO", "3": "11 nm ZnO", "4": "20 nm ZnO", "5": "63 nm ZnO"}
            thickness_str = thickness_map.get(thickness_idx, "? nm ZnO")
        pitch_map = {
            "m1": "P=1.5$\\mu$m, $\\phi$=1$\\mu$m, L=200$\\mu$m",
            "m2": "P=1.5$\\mu$m, $\\phi$=1$\\mu$m, L=50$\\mu$m",
            "m3": "P=4.2$\\mu$m, $\\phi$=2.5$\\mu$m, L=200$\\mu$m",
            "m4": "P=4.2$\\mu$m, $\\phi$=2.5$\\mu$m, L=50$\\mu$m",
            "m5": "P=12$\\mu$m, $\\phi$=5$\\mu$m, L=350$\\mu$m",
            "m6": "P=12$\\mu$m, $\\phi$=5$\\mu$m, L=500$\\mu$m",
            "m7": "P=20$\\mu$m, $\\phi$=17$\\mu$m, L=500$\\mu$m",
            "m8": "P=20$\\mu$m, $\\phi$=9$\\mu$m, L=500$\\mu$m"
        }
        pitch_str = pitch_map.get(sample, sample)
        return f"{pitch_str}, {thickness_str}"
    if label.startswith("r1") or label.startswith("r2"):
        base = label.split("_")[0]
        if "." in base:
            sample, thickness_idx = base.split(".")
            thickness_map = {"1": "1 nm ZnO", "2": "6 nm ZnO", "3": "11 nm ZnO", "4": "20 nm ZnO", "5": "63 nm ZnO"}
            thickness_str = thickness_map.get(thickness_idx, "? nm ZnO")
        else:
            sample = base
            thickness_str = "ref (no ZnO)"
        base_str = "Bulk Si, L=200$\\mu$m" if sample == "r1" else "Bulk Si, L=525$\\mu$m"
        return f"{base_str}, {thickness_str}"
    if label.startswith("a"):
        return f"AZO Sample {label}"
    return label


def assign_colors_for_plot(sample_list):
    def extract_base(label):
        return label.split("_")[0]  # e.g., m1.5

    def extract_sample_type(label):
        base = extract_base(label)
        return base.replace("ref", "").split(".")[0]  # e.g., m1

    sample_types = [extract_sample_type(lbl) for lbl in sample_list]
    type_counts = Counter(sample_types)
    most_common_type, count = type_counts.most_common(1)[0]

    # --- CASE 1: All same type ---
    if len(set(sample_types)) == 1:
        print("[DEBUG] Detected Mode: Single Sample Type (Tab10)")
        tab_colors = plt.get_cmap("tab10").colors
        return {
            label: mcolors.to_hex(tab_colors[i % len(tab_colors)])
            for i, label in enumerate(sample_list)
        }

    # --- CASE 2: One type appears ≥ 3 times (even if mixed) ---
    if count >= 3:
        print(f"[DEBUG] Detected Mode: Dominant Sample Type ({most_common_type}, count={count}) → Tab10")
        tab_colors = plt.get_cmap("tab10").colors
        return {
            label: mcolors.to_hex(tab_colors[i % len(tab_colors)])
            for i, label in enumerate(sample_list)
        }

    # --- CASE 3: All different types → use brand colors ---
    print("[DEBUG] Detected Mode: All Different Sample Types → Fixed Brand Colors")
    return {
        label: sample_base_colors.get(extract_sample_type(label), "#777777")
        for label in sample_list
    }




def plot_selected_samples(sample_list, color_tags=None, save_as=None, range_min=None, range_max=None, no_norm=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    color_map = assign_colors_for_plot(sample_list)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis='both', which='both', length=8, width=2, labelsize=26)

    for i, label in enumerate(sample_list):
        if label not in file_map:
            print(f"Warning: Sample '{label}' not found. Skipping.")
            continue

        path = file_map[label]
        wavelengths, scope_corrected, integration_time = read_scope_corrected(path)
        corrected = scope_corrected.clip(lower=1e-8) if no_norm else (scope_corrected / integration_time).clip(lower=1e-8)
        mask = corrected >= 1e-4
        wl_p, sc_p = wavelengths[mask], corrected[mask]

                # In both plot_selected_samples and plot_selected_samples_norm_max
        if color_tags:
            if i < len(color_tags):
                tag = color_tags[i]
            else:
                print(f"Warning: No color tag for sample {label}. Using last provided tag.")
                tag = color_tags[-1]  # fallback to last tag
            color = tag_to_color.get(tag, color_map.get(label, "#777777"))
        else:
            color = color_map.get(label, "#777777")

        label_str = get_sample_description(label)
        ax.plot(wl_p, sc_p, label=label_str, color=color, linewidth=2, alpha=0.5)


    ax.set_xlabel("Wavelength (nm)", fontsize=26)
    ax.set_ylabel("Spectrometer Counts (arb. units)", fontsize=26)
    ax.set_yscale('log')
    #ax.grid(True, which="both", linestyle='--', linewidth=0.7)

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

def plot_selected_samples_norm_max(sample_list, range_min, range_max, plot_full_range=False, color_tags=None, save_as=None, no_norm=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    color_map = assign_colors_for_plot(sample_list)

    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.tick_params(axis='both', which='both', length=8, width=2, labelsize=26)

    for i, label in enumerate(sample_list):
        if label not in file_map:
            print(f"Warning: Sample '{label}' not found. Skipping.")
            continue

        path = file_map[label]
        wavelengths, scope_corrected, integration_time = read_scope_corrected(path)
        corrected = scope_corrected.clip(lower=1e-8) if no_norm else (scope_corrected / integration_time).clip(lower=1e-8)

        mask = (wavelengths >= range_min) & (wavelengths <= range_max)
        if not np.any(mask):
            print(f"Warning: No data points in specified range for sample '{label}'. Skipping.")
            continue

        max_in_range = corrected[mask].max()
        if max_in_range == 0:
            print(f"Warning: Maximum value is zero for sample '{label}'. Skipping normalization.")
            continue

        normalized = corrected / max_in_range
        if color_tags:
            if i < len(color_tags):
                tag = color_tags[i]
            else:
                print(f"Warning: No color tag for sample {label}. Using last provided tag.")
                tag = color_tags[-1]  # fallback to last tag
            color = tag_to_color.get(tag, color_map.get(label, "#777777"))
        else:
            color = color_map.get(label, "#777777")

        label_str = get_sample_description(label)
        ax.plot(wavelengths, normalized, label=label_str, color=color, linewidth=2, alpha=0.5)

    ax.set_xlabel("Wavelength (nm)", fontsize=26)
    ax.set_ylabel("Normalized Intensity (to max within range)", fontsize=26)
    ax.set_yscale('log')
    #ax.grid(True, which="both", linestyle='--', linewidth=0.7)

    if not plot_full_range:
        ax.set_xlim(range_min, range_max)

    ax.legend(loc='best', fontsize=20)
    plt.tight_layout()

    if save_as:
        export_path = os.path.join(r"C:\\Users\\shmitra\\Nextcloud\\Master_arbeit\\Figures_Thesis\\results_discussion", f"{save_as}.pdf")
        plt.savefig(export_path, bbox_inches='tight')
        print(f"Plot saved to: {export_path}")
    else:
        plt.show()


# --- Fitting Functions ---

def plot_fit_residuals(x, y, model, params, title='Residuals'):
    residuals = y - model.eval(params, x=x)
    plt.figure(figsize=(10, 4))
    plt.scatter(x, residuals, color='gray', label='Residuals')
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Residual')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def fit_peak(model_cls, label, range_min, range_max, no_norm=False):
    if label not in file_map:
        print(f"Sample '{label}' not found.")
        return
    wl, sc, it = read_scope_corrected(file_map[label])
    corrected = sc.clip(lower=1e-8) if no_norm else (sc / it).clip(lower=1e-8)
    mask = (wl >= range_min) & (wl <= range_max) & (corrected >= 1e-4)
    x, y = wl[mask], corrected[mask]
    if len(x) == 0:
        print("No valid data in range.")
        return
    model = model_cls()
    params = model.make_params(amplitude=y.max(), center=float(x.iloc[y.argmax()]), sigma=10)
    if 'gamma' in model.param_names:
        params['gamma'].set(value=10)
    result = model.fit(y, x=x, params=params)
    print(result.fit_report())
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'b.', label='Data')
    plt.plot(x, result.best_fit, 'r-', label='Fit')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Intensity')
    plt.title(f'{model_cls.__name__} Fit: {label}')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plot_fit_residuals(x, y, model, result.params, title=f'{model_cls.__name__} Residuals')

def gaussian_fit(min_wl, max_wl, label, no_norm=False):
    fit_peak(GaussianModel, label, min_wl, max_wl, no_norm)
    
def plot_gaussian_overlay(fwhm, amplitude, min_wl, max_wl, label, no_norm=False):
    if label not in file_map:
        print(f"Sample '{label}' not found.")
        return

    wl, sc, it = read_scope_corrected(file_map[label])
    corrected = sc if no_norm else (sc / it)
    mask = (wl >= min_wl) & (wl <= max_wl)
    x, y = wl[mask], corrected[mask]
    if len(x) == 0:
        print("No data in range.")
        return

    center = (min_wl + max_wl) / 2
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gaussian = amplitude * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f"Data: {label}", linewidth=2)
    plt.plot(x, gaussian, label=f"Gaussian ($A$={amplitude}, FWHM={fwhm} nm)", color='red', linestyle='--')
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity (normalized)" if not no_norm else "Raw Intensity")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.show()




def lorentzian_fit(min_wl, max_wl, label, no_norm=False):
    fit_peak(LorentzianModel, label, min_wl, max_wl, no_norm)

def voigt_fit(min_wl, max_wl, label, no_norm=False):
    fit_peak(VoigtModel, label, min_wl, max_wl, no_norm)
    

# --- Main Command Handler ---

def main():
    global file_map
    file_map = discover_files(base_path)
    args = sys.argv[1:]
    
    if '-preview_colors' in args:
        preview_palette_assignments()
        return

    if not args or '-help' in args or '--help' in args:
        show_filelist = '-filelist' in args
        print_help(show_filelist=show_filelist)
        return

    cmd = args[0]
    no_norm = '-no_norm' in args
    if no_norm:
        args.remove('-no_norm')

    if cmd == '-compare':
        rest = args[1:]
        mn = mx = None
    
        # Try to parse wavelength range (optional)
        try:
            if len(rest) >= 2 and all("." not in x and x.replace(".", "", 1).isdigit() for x in rest[:2]):
                mn, mx = float(rest[0]), float(rest[1])
                rest = rest[2:]
        except Exception as e:
            print(f"[DEBUG] Skipping wavelength range parse: {e}")
    
        # Parse color tags and sample names
        color_tags = [r for r in rest if r.startswith('-')]
        samples = [r for r in rest if not r.startswith('-')]
    
        print(f"[DEBUG] Calling plot_selected_samples()")
        print(f"[DEBUG] samples = {samples}")
        print(f"[DEBUG] color_tags = {color_tags}")
        print(f"[DEBUG] range: {mn}–{mx}")

        plot_selected_samples(samples, color_tags, save_as=None, range_min=mn, range_max=mx, no_norm=no_norm)


    elif cmd == '-compare_norm_max':
        if len(args) < 4:
            print("Error: not enough arguments for -compare_norm_max")
            return
        try:
            mn, mx = float(args[1]), float(args[2])
        except ValueError:
            print("Error: invalid wavelength range")
            return
        idx = 3
        plot_full = False
        if args[idx] == '-full':
            plot_full = True
            idx += 1
        rest = args[idx:]
        save_as = None
        if 'latex' in rest:
            latex_idx = rest.index('latex')
            if latex_idx+1 < len(rest):
                save_as = rest[latex_idx+1]
                rest = rest[:latex_idx]
        color_tags = [r for r in rest if r.startswith('-')]
        samples = [r for r in rest if not r.startswith('-')]
        plot_selected_samples_norm_max(samples, mn, mx, plot_full, color_tags, save_as, no_norm)

    elif cmd == '-gaussian_fit' and len(args) >= 4:
        gaussian_fit(float(args[1]), float(args[2]), args[3], no_norm)
    elif cmd == '-lorentzian_fit' and len(args) >= 4:
        lorentzian_fit(float(args[1]), float(args[2]), args[3], no_norm)
    elif cmd == '-voigt_fit' and len(args) >= 4:
        voigt_fit(float(args[1]), float(args[2]), args[3], no_norm)
    elif cmd == 'plot_gaussian' and len(args) >= 6:
        try:
            fwhm = float(args[1])
            amp = float(args[2])
            wl1 = float(args[3])
            wl2 = float(args[4])
            sample = args[5]
        except ValueError:
            print("Invalid number format for FWHM, amplitude, or wavelength range.")
            return
        plot_gaussian_overlay(fwhm, amp, wl1, wl2, sample, no_norm)
    else:
        print("Invalid command. Use -help for options.")



if __name__ == '__main__':
    main()