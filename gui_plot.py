# -*- coding: utf-8 -*-
"""
GUI to select normalization, wavelength range, fitting, and overlays.
Integrates with cli_main plotting and fitting functions.
"""

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import sys

from plotting import plot_selected_samples
from fitting import gaussian_fit, lorentzian_fit, voigt_fit
from laser_spectrum import HARDCODED_LASER_PATH

class SpectrumGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spectra Plotting GUI")

        # --- Normalization Options ---
        self.norm_time_var = tk.BooleanVar(value=True)
        self.norm_max_var = tk.BooleanVar(value=False)
        self.full_norm_var = tk.BooleanVar(value=False)
        self.baseline_correct_var = tk.BooleanVar(value=False)
        self.log_y_var = tk.BooleanVar(value=True)
        self.show_laser_var = tk.BooleanVar(value=False)
        self.smooth_var = tk.BooleanVar(value=False)

        tk.Checkbutton(root, text="Normalize by Acquisition Time", variable=self.norm_time_var).grid(row=0, column=0, sticky="w")
        tk.Checkbutton(root, text="Normalize to Max in Range", variable=self.norm_max_var).grid(row=1, column=0, sticky="w")
        tk.Checkbutton(root, text="Show Full Spectrum (if Norm Range)", variable=self.full_norm_var).grid(row=2, column=0, sticky="w")
        tk.Checkbutton(root, text="Apply ALS Baseline Correction", variable=self.baseline_correct_var).grid(row=3, column=0, sticky="w")
        tk.Checkbutton(root, text="Log Scale Y-Axis", variable=self.log_y_var).grid(row=4, column=0, sticky="w")
        tk.Checkbutton(root, text="Overlay Laser Spectrum", variable=self.show_laser_var).grid(row=5, column=0, sticky="w")
        tk.Checkbutton(root, text="Apply Moving Average Smoothing", variable=self.smooth_var).grid(row=6, column=0, sticky="w")

        tk.Label(root, text="Smoothing Window:").grid(row=2, column=1, sticky="e")
        self.smooth_window_entry = tk.Entry(root)
        self.smooth_window_entry.insert(0, "5")
        self.smooth_window_entry.grid(row=2, column=2)

        tk.Label(root, text="ALS lambda:").grid(row=3, column=1, sticky="e")
        self.baseline_lam_entry = tk.Entry(root)
        self.baseline_lam_entry.insert(0, "1e5")
        self.baseline_lam_entry.grid(row=3, column=2)

        tk.Label(root, text="ALS p:").grid(row=4, column=1, sticky="e")
        self.baseline_p_entry = tk.Entry(root)
        self.baseline_p_entry.insert(0, "0.01")
        self.baseline_p_entry.grid(row=4, column=2)

        tk.Label(root, text="Laser Range Min:").grid(row=5, column=1, sticky="e")
        self.laser_min_entry = tk.Entry(root)
        self.laser_min_entry.grid(row=5, column=2)

        tk.Label(root, text="Laser Range Max:").grid(row=6, column=1, sticky="e")
        self.laser_max_entry = tk.Entry(root)
        self.laser_max_entry.grid(row=6, column=2)

        # --- Norm Range ---
        tk.Label(root, text="Norm Range Min (nm):").grid(row=7, column=0)
        self.norm_min_entry = tk.Entry(root)
        self.norm_min_entry.grid(row=7, column=1)

        tk.Label(root, text="Norm Range Max (nm):").grid(row=8, column=0)
        self.norm_max_entry = tk.Entry(root)
        self.norm_max_entry.grid(row=8, column=1)

        # --- Plot Range ---
        tk.Label(root, text="Plot Range Min (nm):").grid(row=9, column=0)
        self.range_min_entry = tk.Entry(root)
        self.range_min_entry.grid(row=9, column=1)

        tk.Label(root, text="Plot Range Max (nm):").grid(row=10, column=0)
        self.range_max_entry = tk.Entry(root)
        self.range_max_entry.grid(row=10, column=1)

        # --- Fit Options ---
        tk.Label(root, text="Fit Type:").grid(row=11, column=0)
        self.fit_type = ttk.Combobox(root, values=["None", "Gaussian", "Lorentzian", "Voigt"])
        self.fit_type.current(0)
        self.fit_type.grid(row=11, column=1)

        tk.Label(root, text="Fit Range Min (nm):").grid(row=12, column=0)
        self.fit_min_entry = tk.Entry(root)
        self.fit_min_entry.grid(row=12, column=1)

        tk.Label(root, text="Fit Range Max (nm):").grid(row=13, column=0)
        self.fit_max_entry = tk.Entry(root)
        self.fit_max_entry.grid(row=13, column=1)

        tk.Label(root, text="Fit Sample:").grid(row=14, column=0)
        self.fit_sample_entry = tk.Entry(root)
        self.fit_sample_entry.grid(row=14, column=1)

        # --- Gaussian Overlay ---
        tk.Label(root, text="Overlay FWHM:").grid(row=15, column=0)
        self.overlay_fwhm = tk.Entry(root)
        self.overlay_fwhm.grid(row=15, column=1)

        tk.Label(root, text="Overlay Amplitude:").grid(row=16, column=0)
        self.overlay_amp = tk.Entry(root)
        self.overlay_amp.grid(row=16, column=1)

        tk.Label(root, text="Overlay Center (nm):").grid(row=17, column=0)
        self.overlay_center = tk.Entry(root)
        self.overlay_center.grid(row=17, column=1)

        tk.Label(root, text="Overlay Range Min-Max:").grid(row=18, column=0)
        self.overlay_range = tk.Entry(root)
        self.overlay_range.grid(row=18, column=1)

        # --- Sample Input ---
        tk.Label(root, text="Samples (comma-separated):").grid(row=19, column=0)
        self.sample_entry = tk.Entry(root, width=40)
        self.sample_entry.grid(row=19, column=1)

        # --- Submit Button ---
        tk.Button(root, text="Plot", command=self.submit).grid(row=20, column=0, columnspan=3, pady=10)

    def submit(self):
        try:
            # --- Samples ---
            samples = [s.strip() for s in self.sample_entry.get().split(",") if s.strip()]
            if not samples:
                raise ValueError("Enter at least one sample.")
    
            # --- Plotting Ranges ---
            try:
                range_min_raw = self.range_min_entry.get().strip()
                range_max_raw = self.range_max_entry.get().strip()
                range_min = float(range_min_raw) if range_min_raw else None
                range_max = float(range_max_raw) if range_max_raw else None
            except ValueError:
                raise ValueError("Plot Range Min and Max must be valid numbers (e.g. 450, 1100).")
    
            # --- Flags ---
            no_norm = not self.norm_time_var.get()
            apply_baseline = self.baseline_correct_var.get()
            log_y = self.log_y_var.get()
            smooth = self.smooth_var.get()
            smooth_window = int(self.smooth_window_entry.get().strip()) if self.smooth_window_entry.get().strip() else 5
    
            # --- ALS Parameters ---
            lam = float(self.baseline_lam_entry.get().strip()) if self.baseline_lam_entry.get().strip() else 1e5
            p = float(self.baseline_p_entry.get().strip()) if self.baseline_p_entry.get().strip() else 0.01
    
            # --- Laser Spectrum Overlay ---
            show_laser = bool(self.show_laser_var.get())
            laser_range = None
            if show_laser:
                laser_min_raw = self.laser_min_entry.get().strip()
                laser_max_raw = self.laser_max_entry.get().strip()
                laser_min = float(laser_min_raw) if laser_min_raw else None
                laser_max = float(laser_max_raw) if laser_max_raw else None
                laser_range = (laser_min, laser_max) if (laser_min or laser_max) else None
    
            # --- Gaussian Overlay ---
            gaussians = []
            if all([self.overlay_fwhm.get(), self.overlay_amp.get(), self.overlay_center.get(), self.overlay_range.get()]):
                fwhm = float(self.overlay_fwhm.get())
                amp = float(self.overlay_amp.get())
                center = float(self.overlay_center.get())
                rmin, rmax = map(float, self.overlay_range.get().split())
                gaussians.append({"fwhm": fwhm, "amplitude": amp, "center": center, "min_wl": rmin, "max_wl": rmax, "label": "Overlay"})
    
            # --- Plot ---
            plot_selected_samples(
                sample_list=samples,
                color_tags=None,
                save_as=None,
                range_min=range_min,
                range_max=range_max,
                no_norm=no_norm,
                gaussian_overlays=gaussians,
                apply_baseline=apply_baseline,
                lam=lam,
                p=p,
                log_y=log_y,
                show_laser=show_laser,
                laser_path=HARDCODED_LASER_PATH,
                laser_range=laser_range,
                smooth=smooth,
                smooth_window=smooth_window
            )
    
            # --- Optional Fit ---
            fit_type = self.fit_type.get()
            fit_min = self.fit_min_entry.get()
            fit_max = self.fit_max_entry.get()
            fit_sample = self.fit_sample_entry.get().strip()
    
            if fit_type != "None" and fit_min and fit_max and fit_sample:
                fit_min = float(fit_min)
                fit_max = float(fit_max)
    
                if fit_sample not in samples:
                    raise ValueError(f"Sample '{fit_sample}' not in plotted sample list.")
    
                if fit_type == "Gaussian":
                    gaussian_fit(fit_min, fit_max, fit_sample, no_norm)
                elif fit_type == "Lorentzian":
                    lorentzian_fit(fit_min, fit_max, fit_sample, no_norm)
                elif fit_type == "Voigt":
                    voigt_fit(fit_min, fit_max, fit_sample, no_norm)
    
        except Exception as e:
            messagebox.showerror("Error", str(e))




def launch_gui():
    root = tk.Tk()

    def on_closing():
        root.quit()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    app = SpectrumGUI(root)
    root.mainloop()
    sys.exit(0)


if __name__ == '__main__':
    launch_gui()
