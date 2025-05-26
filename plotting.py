# --- plotting.py ---

import matplotlib.pyplot as plt
import numpy as np
import os
from spectra_io import read_scope_corrected, discover_files
from colors import assign_colors_for_plot, get_sample_description
from plot_style import PLOT_STYLE
from baseline_correction import baseline_als, baseline_airpls
from laser_spectrum import load_laser_spectrum
from smoothing import moving_average


def plot_selected_samples(sample_list, color_tags=None, save_as=None, range_min=None, range_max=None,
                          no_norm=False, gaussian_overlays=None, apply_baseline=False, baseline_method="ALS",
                          lam=1e5, p=0.01, log_y=True, show_laser=False, laser_path=None,
                          laser_range=None, smooth_window=5, smooth=False, air_lam=1e5, air_iter=15):


    file_map = discover_files()
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    color_map = assign_colors_for_plot(sample_list)

    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_STYLE["tick_width"])
    ax.tick_params(axis='both', which='both',
                   length=PLOT_STYLE["tick_length"],
                   width=PLOT_STYLE["tick_width"],
                   labelsize=PLOT_STYLE["labelsize"])

    for i, label in enumerate(sample_list):
        print(f"[DEBUG] Processing sample: {label}")
        if label not in file_map:
            print(f"Warning: Sample '{label}' not found. Skipping.")
            continue

        path = file_map[label]
        wavelengths, scope_corrected, integration_time = read_scope_corrected(path)
        print(f"[DEBUG] Read {len(wavelengths)} wavelengths from file {path}")
        corrected = scope_corrected if no_norm else (scope_corrected / integration_time)

        if apply_baseline:
            if baseline_method == "ALS":
                baseline = baseline_als(corrected, lam=lam, p=p)
            elif baseline_method == "airPLS":
                baseline = baseline_airpls(corrected, lambda_=air_lam, itermax=air_iter)
            else:
                raise ValueError(f"Unknown baseline method: {baseline_method}")
            corrected = corrected - baseline
    
        if smooth:
            corrected = moving_average(corrected, window_size=smooth_window)
        
        print(f"[DEBUG] Pre-thresholding stats: min={corrected.min()}, max={corrected.max()}, any NaNs={np.isnan(corrected).any()}")



        corrected = np.maximum(corrected, 0)
        mask = corrected > 1e-5

        print(f"[DEBUG] After intensity thresholding: min={corrected.min()}, max={corrected.max()}, points kept={np.sum(mask)}")
        wl_p, sc_p = wavelengths[mask], corrected[mask]

        print(f"[DEBUG] Raw wavelength range: {wavelengths.min()} – {wavelengths.max()}")
        print(f"[DEBUG] Filter range: {range_min} – {range_max}")

        if range_min is not None and range_max is not None:
            rmask = (wl_p >= range_min) & (wl_p <= range_max)
            wl_p, sc_p = wl_p[rmask], sc_p[rmask]
            if len(wl_p) == 0 or len(sc_p) == 0:
                print(f"[Warning] No valid data to plot for '{label}' after filtering. Skipping.")
                continue

        tag = color_tags[i] if color_tags and i < len(color_tags) else None
        color = color_map.get(label, '#777777')
        label_str = get_sample_description(label)

        ax.plot(wl_p, sc_p, label=label_str, color=color,
                linewidth=PLOT_STYLE["linewidth"], alpha=PLOT_STYLE["alpha"])

    if gaussian_overlays:
        for g in gaussian_overlays:
            sigma = g['fwhm'] / (2 * np.sqrt(2 * np.log(2)))
            x = np.linspace(g['min_wl'], g['max_wl'], 1000)
            y = g['amplitude'] * np.exp(-((x - g['center']) ** 2) / (2 * sigma ** 2))
            ax.plot(x, y, '--', color=ROLE_COLORS["fit"], label=f"Gaussian @ {g['center']} nm")

    if show_laser and laser_path:
        try:
            harmonics, intensity = load_laser_spectrum(laser_path)
            for n, (wl, I) in harmonics.items():
                if laser_range:
                    min_wl = laser_range[0] if laser_range[0] is not None else wl.min()
                    max_wl = laser_range[1] if laser_range[1] is not None else wl.max()
                    mask = (wl >= min_wl) & (wl <= max_wl)
                    wl_filtered = wl[mask]
                else:
                    wl_filtered = wl

                I_interp = np.interp(wl_filtered, wl, I)
                ax.plot(wl_filtered, I_interp, label=f'{n}ω of driving laser',
                        alpha=PLOT_STYLE["alpha"], linewidth=PLOT_STYLE["linewidth"])
        except Exception as e:
            print(f"[Warning] Could not load laser spectrum: {e}")

    ax.set_xlabel("Wavelength [nm]", fontsize=PLOT_STYLE["xlabelsize"])
    ax.set_ylabel("Spectrometer Counts (a.u.)", fontsize=PLOT_STYLE["ylabelsize"])
    ax.set_yscale('log' if log_y else 'linear')

    if range_min is not None or range_max is not None:
        print(f"[DEBUG] range_min: {range_min}, range_max: {range_max}, type: {type(range_min)}, {type(range_max)}")
        ax.set_xlim(left=range_min, right=range_max)

    ax.legend(loc='best', fontsize=PLOT_STYLE["legend_fontsize"])
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as, bbox_inches='tight')
        print(f"Plot saved to: {save_as}")
    else:
        plt.show()

def plot_selected_samples_norm_max(sample_list, range_min, range_max, plot_full_range=False, color_tags=None, save_as=None, no_norm=False, apply_baseline=False, lam=1e5, p=0.01, log_y=True):
    file_map = discover_files()
    fig, ax = plt.subplots(figsize=PLOT_STYLE["figsize"])
    color_map = assign_colors_for_plot(sample_list)

    for spine in ax.spines.values():
        spine.set_linewidth(PLOT_STYLE["tick_width"])
    ax.tick_params(axis='both', which='both', length=PLOT_STYLE["tick_length"], width=PLOT_STYLE["tick_width"], labelsize=PLOT_STYLE["labelsize"])

    for i, label in enumerate(sample_list):
        if label not in file_map:
            print(f"Warning: Sample '{label}' not found. Skipping.")
            continue

        path = file_map[label]
        wavelengths, scope_corrected, integration_time = read_scope_corrected(path)
        corrected = scope_corrected if no_norm else (scope_corrected / integration_time)

        if apply_baseline:
            corrected = corrected - baseline_als(corrected, lam=lam, p=p)

        corrected = corrected.clip(lower=1e-8)

        mask = (wavelengths >= range_min) & (wavelengths <= range_max)
        if not np.any(mask):
            print(f"Warning: No data points in specified range for sample '{label}'. Skipping.")
            continue

        max_in_range = corrected[mask].max()
        if max_in_range == 0:
            print(f"Warning: Maximum value is zero for sample '{label}'. Skipping normalization.")
            continue

        normalized = corrected / max_in_range
        mask_plot = normalized >= 1e-5
        wavelengths_plot = wavelengths[mask_plot]
        normalized_plot = normalized[mask_plot]

        color = color_map.get(label, '#777777')
        label_str = get_sample_description(label)
        ax.plot(wavelengths_plot, normalized_plot, label=label_str, color=color,
                linewidth=PLOT_STYLE["linewidth"], alpha=PLOT_STYLE["alpha"])

    ax.set_xlabel("Wavelength [nm]", fontsize=PLOT_STYLE["xlabelsize"])
    ax.set_ylabel("Normalized Intensity (to max within range)", fontsize=PLOT_STYLE["ylabelsize"])
    ax.set_yscale('log' if log_y else 'linear')

    if not plot_full_range:
        ax.set_xlim(range_min, range_max)

    ax.legend(loc='best', fontsize=PLOT_STYLE["legend_fontsize"])
    plt.tight_layout()

    if save_as:
        export_path = os.path.join(PLOT_STYLE["export_path"], f"{save_as}.pdf")
        plt.savefig(export_path, bbox_inches='tight')
        print(f"Plot saved to: {export_path}")
    else:
        plt.show()
        


