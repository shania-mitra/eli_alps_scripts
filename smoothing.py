# -*- coding: utf-8 -*-
"""
Created on Mon May 19 15:55:34 2025

@author: shmitra
"""

# smoothing.py

import numpy as np

def moving_average(y, window_size=5):
    """
    Computes moving average of a 1D array.

    Args:
        y (np.ndarray): input signal (e.g., intensity array)
        window_size (int): number of points for the moving average window

    Returns:
        np.ndarray: smoothed array, same length as input
    """
    return np.convolve(y, np.ones(window_size) / window_size, mode='same')
