# -*- coding: utf-8 -*-
"""
Created on Mon May 19 13:20:29 2025

@author: shmitra
"""

import tkinter as tk
from tkinter import messagebox

def submit():
    try:
        amp = float(entry_amp.get())
        center = float(entry_center.get())
        fwhm = float(entry_fwhm.get())
        messagebox.showinfo("Gaussian Parameters", f"Amplitude: {amp}\nCenter: {center}\nFWHM: {fwhm}")
        # You can now call your Gaussian plot function here with (amp, center, fwhm)
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")

# Create GUI window
root = tk.Tk()
root.title("Gaussian Fit Parameters")

# Labels and Entry boxes
tk.Label(root, text="Amplitude:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
entry_amp = tk.Entry(root)
entry_amp.grid(row=0, column=1, padx=10)

tk.Label(root, text="Center Wavelength (nm):").grid(row=1, column=0, padx=10, pady=5, sticky="e")
entry_center = tk.Entry(root)
entry_center.grid(row=1, column=1, padx=10)

tk.Label(root, text="FWHM (nm):").grid(row=2, column=0, padx=10, pady=5, sticky="e")
entry_fwhm = tk.Entry(root)
entry_fwhm.grid(row=2, column=1, padx=10)

# Submit button
tk.Button(root, text="Submit", command=submit).grid(row=3, column=0, columnspan=2, pady=15)

root.mainloop()
