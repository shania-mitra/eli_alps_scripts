# -*- coding: utf-8 -*-
"""
Created on Sat May 24 18:50:49 2025

@author: shmitra
"""

# z_score_gui.py

import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re
import io

# ----- File loading utilities (same as in z_score.py) -----
BASE_PATH = r"C:\Users\shmitra\Nextcloud\1uanalysis"

def discover_files(base_dir=BASE_PATH, filter_subdir="NOBP"):
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
        raise ValueError(f"Integration time not found in {filepath}")
    df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=';', header=None)
    wavelengths = df.iloc[:, 0]
    scope_corrected = df.iloc[:, -1]
    return wavelengths, scope_corrected, integration_time

def compute_z_score(label, ranges):
    file_map = discover_files()
    if label not in file_map:
        return f"Sample '{label}' not found.\n"

    wl, sc, _ = read_scope_corrected(file_map[label])
    noise = sc[wl <= 400]
    mean_n = noise.mean()
    std_n = noise.std()
    
    results = []
    for rmin, rmax in ranges:
        region = sc[(wl >= rmin) & (wl <= rmax)]
        if region.empty:
            results.append(f"{rmin}-{rmax} nm: No data\n")
            continue
        peak = region.max()
        z = (peak - mean_n) / std_n if std_n != 0 else float('inf')
        status = "significant ✅" if z >= 3 else "not significant ❌"
        results.append(f"{rmin}-{rmax} nm: Peak={peak:.2e}, Z={z:.2f} → {status}\n")
    return "".join(results)

# ----- GUI -----
class ZScoreGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Z-Score Peak Significance Checker")

        tk.Label(root, text="Sample Label:").grid(row=0, column=0, sticky='e')
        self.sample_entry = tk.Entry(root, width=40)
        self.sample_entry.grid(row=0, column=1)

        tk.Label(root, text="Wavelength Ranges (min max, comma-separated):").grid(row=1, column=0, sticky='ne')
        self.ranges_entry = tk.Entry(root, width=50)
        self.ranges_entry.grid(row=1, column=1)

        self.result_box = tk.Text(root, height=12, width=70)
        self.result_box.grid(row=3, column=0, columnspan=2, pady=10)

        tk.Button(root, text="Check Significance", command=self.run_z_score).grid(row=2, column=0, columnspan=2, pady=5)

    def run_z_score(self):
        label = self.sample_entry.get().strip()
        range_raw = self.ranges_entry.get().strip()
        if not label or not range_raw:
            messagebox.showerror("Missing Input", "Please enter both sample label and wavelength ranges.")
            return

        try:
            ranges = []
            chunks = range_raw.split(',')
            for chunk in chunks:
                parts = list(map(float, chunk.strip().split()))
                if len(parts) != 2:
                    raise ValueError("Each range must have min and max values")
                ranges.append((parts[0], parts[1]))
        except Exception as e:
            messagebox.showerror("Invalid Range Format", str(e))
            return

        output = compute_z_score(label, ranges)
        self.result_box.delete('1.0', tk.END)
        self.result_box.insert(tk.END, output)

# ----- Launch -----
if __name__ == "__main__":
    root = tk.Tk()
    app = ZScoreGUI(root)
    root.mainloop()
