# -*- coding: utf-8 -*-
"""
Created on Mon May 19 09:38:04 2025

@author: shmitra
"""

# --- colors.py ---

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
from collections import Counter
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap

intensity_colours = [
    "#020102",  # Night Rider
    "#baafa3",  # Nomad
    "#3037ff",  # Blue Ribbon
    "#6bafff",  # Cornflower Blue
   # "#f44f4f",  # Carnation
    "#890c92",  # Purple
    "#d80ad6",  # Purple Pizzazz
]

# Fixed base sample colors (legacy fallback if needed)
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

# Color palettes
TAB10 = [c for i, c in enumerate(plt.get_cmap("tab10").colors) if i != 3]  # Tab10 minus yellow

# Okabe-Ito: orange, sky blue, bluish green, blue, vermilion, reddish purple
OKABE_ITO = [
    "#E69F00",  # orange-yellow (keep)
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7"   # reddish purple
]

VIRIDIS = [mcolors.to_hex(plt.cm.viridis(i)) for i in np.linspace(0, 1, 10)]
VIRIDIS = [c for c in VIRIDIS if not ("ffcc33" in c or "fdae61" in c or "fee08b" in c)]

# Utility: get base sample (e.g. 'm1' from 'm1.2_NOBP_3000')
def extract_base(label):
    return label.split("_")[0]

def extract_sample_type(label):
    return extract_base(label).replace("ref", "").split(".")[0]

def extract_thickness_index(label):
    base = extract_base(label)
    parts = base.split(".")
    return parts[1] if len(parts) > 1 else None

def extract_intensity(label):
    match = re.search(r"_(\d+)$", label)
    return int(match.group(1)) if match else None

def assign_colors_for_plot(sample_list):
    color_map = {}
    base_set = set(extract_sample_type(lbl) for lbl in sample_list)

    # Ensure ZnOref always gets the same color
    for label in sample_list:
        if "ZnOref" in label:
            color_map[label] = sample_base_colors["ZnOref"]

    # Test Case 2: Same base, different ZnO thicknesses → Okabe-Ito
    valid_labels = [lbl for lbl in sample_list if extract_thickness_index(lbl) is not None and "ZnOref" not in lbl]
    thickness_indices = sorted(set(extract_thickness_index(lbl) for lbl in valid_labels), key=int)
    
    if len(thickness_indices) > 1 and len(set(extract_sample_type(lbl) for lbl in valid_labels)) == 1:
        print(f"[DEBUG] Okabe-Ito mode triggered. Base sample: {extract_sample_type(valid_labels[0])}")
        thickness_map = {t: OKABE_ITO[i % len(OKABE_ITO)] for i, t in enumerate(thickness_indices)}
        for label in valid_labels:
            t = extract_thickness_index(label)
            color_map[label] = thickness_map.get(t, "#777777")
        return color_map


    # Test Case 3: Same base and thickness, different intensities → Viridis
    elif len(set(extract_intensity(lbl) for lbl in sample_list if "ZnOref" not in lbl)) > 1:
        print("[DEBUG] Using Viridis color map for intensity variations")
        intensities = [extract_intensity(lbl) for lbl in sample_list if "ZnOref" not in lbl]
        norm = plt.Normalize(min(intensities), max(intensities))
        scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=ListedColormap(intensity_colours))
        for label in sample_list:
            if "ZnOref" in label:
                continue
            intensity = extract_intensity(label)
            rgba = scalar_map.to_rgba(intensity)
            color_map[label] = mcolors.to_hex(rgba)
        return color_map

    # Default Case 1: Fixed color per sample base
    print("[DEBUG] Using fixed global color map for base samples")
    for label in sample_list:
        if label not in color_map:
            stype = extract_sample_type(label)
            color_map[label] = sample_base_colors.get(stype, "#777777")
    return color_map


# Role-based color scheme (used for fits, overlays, laser etc.)
ROLE_COLORS = {
    "sample": "#f28e2c",      # default for sample if needed
    "fit": "#e15759",         # for fitted peaks
    "laser": "#4e79a7",       # for laser spectrum overlays
    "overlay": "#76b7b2",     # for Gaussian overlays
    "highlight": "#59a14f",   # optional regions, bands
    "background": "#bab0ac",  # for shading, grid, etc.
    "error": "#b07aa1"         # for failed fits, zero lines
}

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


