# -*- coding: utf-8 -*-
"""
Created on Wed May 14 17:31:01 2025

@author: shmitra
"""

# import matplotlib.pyplot as plt
# from lmfit.models import GaussianModel
# import numpy as np


# wavelength = []
# intensity = []

# def frange(start, stop, step):
#     while start <= stop:
#         yield start
#         start += step

# with open('20240215_Spect_BaF2_520_Si_650_-18.8k_-110k_5.8W_4CM_afternoon.txt','r') as f:
#     for row in f:
#         row = row.strip().split(',')
#         wavelength.append(float(row[0]))   # convert to float
#         intensity.append(float(row[1]))
        
# wavelength0 = np.array(wavelength)
# intensity = np.array(intensity)
# wavelength2 = np.divide(wavelength0,2)
# wavelength3 = np.divide(wavelength0,3)
# wavelength4 = np.divide(wavelength0,4)
# wavelength5 = np.divide(wavelength0,5)

# print(np.max(wavelength0))
     
# model0 = GaussianModel()
# params0 = model0.guess(intensity,x=wavelength0 )
# result0 = model0.fit(intensity, params0, x=wavelength0)

# model2 = GaussianModel()
# params2 = model2.guess(intensity,x=wavelength2 )
# result2 = model2.fit(intensity, params2, x=wavelength2)

# model3 = GaussianModel()
# params3 = model3.guess(intensity,x=wavelength3 )
# result3 = model3.fit(intensity, params3, x=wavelength3)

# #print(result.fit_report())

# result0.plot_fit()
# result2.plot_fit()
# result3.plot_fit()


# plt.plot(wavelength0, intensity)
# plt.plot(wavelength2, intensity)
# plt.plot(wavelength3, intensity)
# plt.plot(wavelength4, intensity)
# plt.plot(wavelength5, intensity)

# plt.show()

import matplotlib.pyplot as plt
from lmfit.models import GaussianModel
import numpy as np

# --- Load data ---
wavelength = []
intensity = []

with open('20240215_Spect_BaF2_520_Si_650_-18.8k_-110k_5.8W_4CM_afternoon.txt', 'r') as f:
    for row in f:
        row = row.strip().split(',')
        wavelength.append(float(row[0]))
        intensity.append(float(row[1]))

# --- Convert to numpy arrays ---
wavelength = np.array(wavelength)
intensity = np.array(intensity)

# --- Simulate harmonic wavelength axes ---
harmonics = {
    1: wavelength,
    2: wavelength / 2,
    3: wavelength / 3,
    4: wavelength / 4,
    5: wavelength / 5
}

# --- Fit Gaussians to each harmonic ---
results = {}
models = {}
params = {}

for n, wl in harmonics.items():
    model = GaussianModel()
    param = model.guess(intensity, x=wl)
    result = model.fit(intensity, param, x=wl)
    models[n] = model
    params[n] = param
    results[n] = result

fwhm_list = []
print("\n--- Gaussian Fit Parameters for Each Harmonic ---")
for n in [1, 2, 3, 4, 5]:
    res = results[n]
    center = res.params['center'].value
    sigma = res.params['sigma'].value
    amplitude = res.params['amplitude'].value
    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    fwhm_list.append(fwhm)
    print(f"Harmonic {n}ω:")
    print(f"  Center   = {center:.2f} nm")
    print(f"  Sigma    = {sigma:.2f} nm")
    print(f"  FWHM     = {fwhm:.2f} nm")
    print(f"  Amplitude= {amplitude:.2e} a.u.")
    plt.scatter(n, 2 * np.sqrt(2 * np.log(2)) * sigma)

# Only return this if you're inside a function
# return fwhm_list



# --- Plot all curves and their fits ---
plt.figure(figsize=(10, 6))

colors = ['#333333', '#1f77b4', '#2ca02c', '#d62728', '#9467bd']
for i, n in enumerate([1, 2, 3, 4, 5]):
    wl = harmonics[n]
    label = f"{n}ω (center={results[n].params['center'].value:.1f} nm)"
    plt.plot(wl, intensity, color=colors[i], alpha=0.4, label=label)
    plt.plot(wl, results[n].best_fit, color=colors[i], linestyle='--', linewidth=2)
    

plt.xlabel("Wavelength (nm)", fontsize=14)
plt.ylabel("Spectral Intensity (a.u.)", fontsize=14)
plt.title("Original + Harmonic Spectra with Gaussian Fits", fontsize=15)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
