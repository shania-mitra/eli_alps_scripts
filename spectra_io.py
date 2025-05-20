# -*- coding: utf-8 -*-
"""
Created on Mon May 19 09:36:26 2025

@author: shmitra
"""

# --- spectra_io.py ---

import os
import re
import io
import pandas as pd

base_path = r"C:\\Users\\shmitra\\Nextcloud\\1uanalysis"

def discover_files(base_dir=base_path, filter_subdir="NOBP"):
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
