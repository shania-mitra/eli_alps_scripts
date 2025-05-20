# -*- coding: utf-8 -*-
"""
Created on Mon May 12 13:01:37 2025

@author: shmitra
"""
import matplotlib.pyplot as plt
# Read Text Files with Pandas using read_csv()

# importing pandas
import pandas as pd

# read text file into pandas DataFrame
df = pd.read_csv("C:/Users/shmitra/Nextcloud/1uanalysis/Calibration_Avantes.txt", sep="\t", skiprows=1, comment='#', names=['wavelength_nm', 'calibration'])

# display DataFrame
plt.plot(df['wavelength_nm'], df['calibration'])
plt.semilogy()
plt.show()