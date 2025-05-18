#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:33:57 2024

@author: miyamotomanari
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob

# Lorentzian function for fitting peaks and dips
def Lorentz_func(x, a, mu, sigma, c):
    return a * sigma**2 / ((x - mu)**2 + sigma**2) + c

# Inverted Lorentzian function for fitting dips (valleys)
def Inverted_Lorentz_func(x, a, mu, sigma, c):
    return -a * sigma**2 / ((x - mu)**2 + sigma**2) + c

# Generate x-profile by summing along one axis
def MakeXProfile(data):
    np_xprofile = np.sum(data, axis=0)
    return np_xprofile

def MakeYProfile(data):
    np_xprofile = np.sum(data, axis=1)
    return np_xprofile

# Fit x-profile using Lorentzian function
def FittingXprofile(xprofile, inverted=False):
    x = np.arange(0, len(xprofile))
    y = xprofile
    
    init_params = np.array([2700000, 110, 30, 1500000])
    if inverted:
        popt, _ = curve_fit(Inverted_Lorentz_func, x, y, p0=init_params, maxfev=2000)
        _fit = Inverted_Lorentz_func(x, *popt)
    else:
        popt, _ = curve_fit(Lorentz_func, x, y, p0=init_params, maxfev=2000)
        _fit = Lorentz_func(x, *popt)
    
    peak_height = popt[0]  # Get peak or dip height (a)
    width = popt[2]  # Get width (sigma)
    
    return _fit, peak_height, width

# Plot data
def plot_data(ndarr, xprofile, fit, mode):
    fig, ax = plt.subplots()
    
    if mode == 'image':
        ax.imshow(ndarr, aspect='auto')
        ax.set_title('Image')
    elif mode == 'profile':
        x = np.arange(0, len(xprofile))
        ax.plot(x, xprofile)
        ax.set_ylim([np.min(xprofile), np.max(xprofile)])
        ax.set_title('1D Profile')
    elif mode == 'fitting':
        x = np.arange(0, len(xprofile))
        ax.plot(x, xprofile, label='Profile')
        ax.plot(x, fit, label='Fitting', linestyle='--')
        ax.set_ylim([np.min(xprofile), np.max(xprofile)])
        ax.legend()
        ax.set_title('Fitting Result')
    
    plt.show()

# Fit and find peak or dip using Lorentzian function
def FitAndFindLorentzPeak(x, y, inverted=True):
    init_params = [max(y), 65, 10, min(y)]
    
    if inverted:
        popt, _ = curve_fit(Inverted_Lorentz_func, x, y, p0=init_params, maxfev=2000)
        fitted_curve = Inverted_Lorentz_func(x, *popt)
    else:
        popt, _ = curve_fit(Lorentz_func, x, y, p0=init_params, maxfev=2000)
        fitted_curve = Lorentz_func(x, *popt)
    
    peak_position = popt[1]  # The mean (mu) gives the peak x-value
    
    return fitted_curve, peak_position

# Folder names
# folderNames = ["10-0-10-220z(under)", "10-0-10-220z(upper)"]
folderNames = ["10-0125-10-220z(1)", "10-0125-10-220z(2)", "10-0125-10-220z(3)", "10-0125-10-220z(4)", "10-0125-10-220z(5)"]
npy_files_list = [glob.glob("./" + folder + "/*.npy") for folder in folderNames]
npy_files_list = [sorted(files) for files in npy_files_list]

min_len = min(len(files) for files in npy_files_list)

peak_height_ndarr = np.array([])  # Array for storing peak heights
width_ndarr = np.array([])  # Array for storing width changes
mode = 'image'

# Stack and process data
for i in range(min_len):
    stacked_ndarr = np.zeros_like(np.load(npy_files_list[0][i]))
    stacked_ndarr = stacked_ndarr.astype(np.uint64)
    for files in npy_files_list:
        stacked_ndarr_1 = np.load(files[i]).astype(np.uint64)
        
        stacked_ndarr += stacked_ndarr_1
        

    xprofile = MakeXProfile(stacked_ndarr)
    
    
    # Check if you expect a peak or dip (inverted=False for peak, True for dip)
    fit, peak_height, width = FittingXprofile(xprofile, inverted=False)  # Change inverted=True for dips
    
    peak_height_ndarr = np.append(peak_height_ndarr, peak_height)
    width_ndarr = np.append(width_ndarr, width)
    
    plot_data(stacked_ndarr, xprofile, fit, mode)

# Plot and fit peak height or dip variation
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.arange(0, len(peak_height_ndarr))
ax.plot(x, peak_height_ndarr, label="Peak Height Variation")

# Fit peak height or dip using Lorentzian fit and find position
fitted_curve, peak_position = FitAndFindLorentzPeak(x, peak_height_ndarr, inverted=True)  # Change to inverted=True for dips

ax.plot(x, fitted_curve, linestyle='--', label=f'Lorentzian Fit (Peak at {peak_position:.2f})')
ax.legend()
ax.set_title('Peak Height Variation with Lorentzian Fit')
plt.savefig(f'{folderNames[0]}_peak_height_variation_with_lorentz_fit.png')
plt.show()

print(f"Peak position in peak height variation: {peak_position:.2f}")

# Plot and fit width variation
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, width_ndarr, label="Width Variation")

# Fit width using Lorentzian fit and find peak position
fitted_curve, peak_position = FitAndFindLorentzPeak(x, width_ndarr)

ax.plot(x, fitted_curve, linestyle='--', label=f'Lorentzian Fit (Peak at {peak_position:.2f})')
ax.legend()
ax.set_title('Width Variation with Lorentzian Fit')
plt.savefig(f'{folderNames[0]}_width_variation_with_lorentz_fit.png')
plt.show()

print(f"Peak position in width variation: {peak_position:.2f}")
print(stacked_ndarr.dtype)