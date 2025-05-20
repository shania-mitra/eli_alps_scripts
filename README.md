
# 🧪 Spectral Analysis Toolkit for ZnO and Si Samples

This repository provides a comprehensive framework for analyzing spectral data from ZnO-coated silicon membranes and bulk samples for high-harmonic generation (HHG) research. It supports GUI and CLI-based plotting, normalization, smoothing, peak fitting, and laser harmonic overlays.

---

## 📦 Repository Structure

```
.
├── baseline_correction.py     # ALS baseline removal
├── colors.py                  # Sample-based color mapping and label formatting
├── comparespectra.py          # CLI tool for comparing spectra and fitting
├── fitting.py                 # Gaussian/Lorentzian/Voigt fit logic
├── gui_plot.py                # GUI interface for visualization and fit control
├── laser_spectrum.py          # Laser spectrum import and harmonic decomposition
├── plot_style.py              # Central plot style configuration
├── plotting.py                # Plot logic: spectra, laser overlays, smoothing
├── smoothing.py               # Moving average implementation
└── spectra_io.py              # File discovery + scope-corrected data loader
```

---

## 🚀 Getting Started

### 🔧 Installation

```bash
pip install numpy pandas matplotlib scipy lmfit seaborn
```

### 🖼️ Launch the GUI

```bash
python gui_plot.py
```

### 🧵 Use the CLI

```bash
python comparespectra.py -compare 500 1100 ZnOref_NOBP_1500 m1.3_NOBP_1500 -gauss 80 40000 660 600 750 ZnOref_NOBP_1500
```

---

## ⚙️ Key Features

- ✅ **Baseline Correction** using ALS (Asymmetric Least Squares)
- ✅ **Integration-Time Normalization** and **Max-in-Range Normalization**
- ✅ **Gaussian, Lorentzian, and Voigt Fitting** with residual plots
- ✅ **Color Management** based on sample types and power scans
- ✅ **Laser Spectrum Overlay** with harmonic axes (1ω–5ω)
- ✅ **GUI Controls** for smoothing, fitting, and overlay parameters

---

## 📌 To-Do List (Prioritized)

- [ ] Implement moving averages across all spectra
- [ ] Define and apply ALS baseline for ZnOref
- [ ] Fit single Gaussian peak (post-baseline)
- [ ] Fit multiple overlapping harmonics
- [ ] Extract FWHM and plot vs harmonic order
- [ ] Overlay harmonic peaks for ZnOref across power levels
- [ ] Normalize laser spectra using acquisition time
- [ ] Interpret TIPTOE spectra cautiously (no direct HHG mapping)

---

## 🗂️ Example Folder Structure

```
Nextcloud/
└── 1uanalysis/
    └── m1/
        └── NOBP/
            ├── Int_1500mW.TXT
            ├── Int_2000mW.TXT
```

> 🔍 Files must begin with `Int_` and contain `"Integration time [ms]:"` in header.

---

## 📤 Output Directory

By default, figures are saved to:

```
C:/Users/shmitra/Nextcloud/Master_arbeit/Figures_Thesis/results_discussion
```

This can be modified in `plot_style.py`.

---

## 📚 Dependencies

- `numpy`, `pandas`, `matplotlib`, `scipy`
- `lmfit`, `seaborn`, `tkinter`

---

## 👩‍🔬 Author

**Shania Mitra**  
Master’s Thesis – High-Harmonic Generation in Structured Solids  
ELI-ALPS & University of Göttingen  
May 2025

---

## 📄 License

This code is for academic and research use. Please cite or credit appropriately.

