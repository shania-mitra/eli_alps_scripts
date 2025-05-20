# -*- coding: utf-8 -*-
"""
Created on Mon May 19 09:38:04 2025

@author: shmitra
"""

# --- colors.py ---

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
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import re
    from collections import Counter

    def extract_base(label):
        return label.split("_")[0]

    def extract_sample_type(label):
        return extract_base(label).replace("ref", "").split(".")[0]

    base_samples = [extract_sample_type(lbl) for lbl in sample_list]
    type_counts = Counter(base_samples)
    most_common_type, count = type_counts.most_common(1)[0]

    # Case: dominant sample type
    if count >= len(sample_list) - 1:
        print("[DEBUG] Detected Mode: Dominant Sample Type (Tab10)")
        tab_colors = plt.get_cmap("tab10").colors
        return {
            label: mcolors.to_hex(tab_colors[i % len(tab_colors)])
            for i, label in enumerate(sample_list)
        }

    # Case: different types
    print("[DEBUG] Detected Mode: Mixed Sample Types (Fixed Brand Colors)")
    return {
        label: sample_base_colors.get(extract_sample_type(label), "#777777")
        for label in sample_list
    }
