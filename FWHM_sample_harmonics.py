# -*- coding: utf-8 -*-
"""
Created on Wed May 21 16:01:08 2025

@author: shmitra
"""
from plotting import fwhm_vs_harmonic_plot
ranges = [(500, 530), (570, 600), (630, 670), (700, 740)]
harmonics = [6, 5, 4, 3]

fwhm_vs_harmonic_plot(
    label="m1.5_NOBP_3000",
    ranges=ranges,
    harmonic_orders=harmonics,
    model_type="Voigt",  # or "Gaussian", "Lorentzian"
    apply_baseline=True
)
