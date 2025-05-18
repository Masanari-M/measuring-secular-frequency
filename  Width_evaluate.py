#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 13:32:40 2024

@author: miyamotomanari
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import re
from scipy.optimize import curve_fit

plt.rcParams.update({
    'font.size': 12,            # Default font size
    'font.family': 'serif',     # Use serif fonts
    'text.usetex': False,       # Optional: If you want to use LaTeX for rendering text
    'axes.labelsize': 14,       # Label size for x and y axes
    'axes.titlesize': 16,       # Title size
    'legend.fontsize': 12,      # Font size for the legend
    'xtick.labelsize': 12,      # Font size for x-axis ticks
    'ytick.labelsize': 12,      # Font size for y-axis ticks
    'lines.linewidth': 2,       # Line width
    'lines.markersize': 6,      # Marker size for points
    'figure.figsize': [6, 4]    # Default figure size
})

# Lorentzian function for fitting
def Lorentz_func(x, a, mu, sigma, c):
    return a * sigma**2 / ((x - mu)**2 + sigma**2) + c

# Function to fit width change using Lorentzian function
def FitWidthChange(x, y):
    init_params = [max(y), np.mean(x), np.std(x), min(y)]  # Initial guess for Lorentzian fit
    # init_params = [max(y), 1470, np.std(x), min(y)]  # Initial guess for Lorentzian fit
    popt, pcov = curve_fit(Lorentz_func, x, y, p0=init_params, maxfev=5000)
    fit_y = Lorentz_func(x, *popt)
    return fit_y, popt, np.sqrt(np.diag(pcov))  # Return fit and errors

# Function to fit 1D profile and calculate the width and its error
def FitProfile(x, y):
    init_params = [max(y), 100, 20, min(y)]  # Initial guess: height, center, width, baseline
    # popt, pcov = curve_fit(Lorentz_func, x, y, p0=init_params,  maxfev=2000)#(peak, center, width, background)
    popt, pcov = curve_fit(Lorentz_func, x, y, p0=init_params, bounds=((0, 30, 5, 0),(100000, 150, 50, 1000)), maxfev=2000)
    
    fitted_curve = Lorentz_func(x, *popt)
    width = popt[2]  # Extract the width (sigma)
    width_error = np.sqrt(np.diag(pcov))[2]  # Extract the error (standard deviation) of sigma

    return fitted_curve, width, width_error

# Custom sorting function to extract numbers from filenames and sort based on them
def extract_number(filename):
    match = re.search(r'(\d+\.\d+)', filename)  # Extract the numeric part (float) from the filename
    return float(match.group(1)) if match else float('inf')  # Return as float for sorting

# Function to plot and fit vertical (Y-axis) profiles with error bars
def func_v(num):
    path = dir_path + '/' + dir_list[num]
    files = glob.glob(path + "/*_capImg.npy")
    
    # Sort files by the numeric part extracted from the filename
    files = sorted(files, key=extract_number)

    
    
    widths = []
    width_errors = []
    numbers = []

    for f in files:
        nda = np.zeros((200, 100))
        nda_temp = np.load(f)
        nda += nda_temp
        nda /= len(files)
        nda[nda <= 10] = 0

        fig, ax = plt.subplots()
        ax.imshow(nda)
        a = re.findall(r'(\d+\.\d+)', f)
        ax.set_title(a[0])
        # plt.show()
        plt.close()

        # Get vertical profile
        nda = nda.T
        x0 = []
        for i in np.linspace(v_start, v_end, v_end - v_start + 1):
            x0.append(nda[int(i)])
        x0 = np.array(x0).T
        x1 = np.sum(x0, axis=1)

        # Fit vertical profile
        x = np.arange(len(x1))
        fitted_curve, width, width_error = FitProfile(x, x1)

        # Store width, error, and filename number
        widths.append(width)
        width_errors.append(width_error)
        numbers.append(float(a[0]))  # Store the number from filename

        # Plot original profile and fitting result
        plt.plot(x, x1, label="Profile")
        plt.plot(x, fitted_curve, label="Fit", linestyle='--')
        plt.legend()
        plt.title(f"Vertical Profile Fitting for {a[0]}")
        # plt.show()
        plt.close()

    # Plot width changes with error bars
    plt.errorbar(numbers, widths, yerr=width_errors, fmt='o', color = 'black', label="Width Variation with Errors")
    plt.xlabel("File number")
    plt.ylabel("Width")
    plt.title("Width Variation vs File Number (with Error Bars)")
    plt.legend()
    plt.show()
    
    # Perform Lorentzian fitting on the width variation
    numbers = np.array(numbers)
    widths = np.array(widths)
    width_fit_y, popt, perr = FitWidthChange(numbers, widths)

    # Plot the fitted curve
    plt.plot(numbers, width_fit_y, label=f"Lorentzian Fit peak={popt[1]:.2f}", color='red', linestyle='--')
    plt.errorbar(numbers, widths, yerr=width_errors, fmt='o', label="Width Variation with Errors")
    plt.xlabel("File number")
    plt.ylabel("Width")
    plt.title("Width Variation with Lorentzian Fitting")
    plt.legend()
    graphlog = 'resultfig'
    folder_path = os.path.join(graphlog, foldername)
    os.makedirs(folder_path, exist_ok=True)

    # 画像の保存
    fig_name = os.path.join(folder_path, f'{dir_list[num]}.png')
    plt.savefig(fig_name, dpi=300)
    plt.show()  # 画像を表示
    plt.close()
    
    # Display the fit parameters and errors
    print(f"Fit parameters: a={popt[0]:.2f}, mu={popt[1]:.2f}, sigma={popt[2]:.2f}, c={popt[3]:.2f}")
    print(f"Fit errors: a_err={perr[0]:.2f}, mu_err={perr[1]:.2f}, sigma_err={perr[2]:.2f}, c_err={perr[3]:.2f}")

    # Find the file number corresponding to the maximum width (mu from Lorentz fit)
    max_width_number = popt[1]
    print(f"The file number with the largest width (from Lorentz fit) is: {max_width_number}")
    
# Function to plot and fit gorizontal (X-axis) profiles with error bars
def func_h(num):
    path = dir_path + '/' + dir_list[num]
    files = glob.glob(path + "/*_capImg.npy")
    
    # Sort files by the numeric part extracted from the filename
    files = sorted(files, key=extract_number)

    
    
    widths = []
    width_errors = []
    numbers = []

    for f in files:
        nda = np.zeros((100, 200))
        nda_temp = np.load(f)
        nda += nda_temp
        nda /= len(files)
        nda[nda <= 10] = 0

        fig, ax = plt.subplots()
        ax.imshow(nda)
        a = re.findall(r'(\d+\.\d+)', f)
        ax.set_title(a[0])
        # plt.show()
        plt.close()

        # Get vertical profile
       
        x0 = []
        for i in np.linspace(v_start, v_end, v_end - v_start + 1):
            x0.append(nda[int(i)])
        x0 = np.array(x0)
        x1 = np.sum(x0, axis=0)

        # Fit vertical profile
        x = np.arange(len(x1))
        fitted_curve, width, width_error = FitProfile(x, x1)

        # Store width, error, and filename number
        widths.append(width)
        width_errors.append(width_error)
        numbers.append(float(a[0]))  # Store the number from filename

        # Plot original profile and fitting result
        plt.plot(x, x1, label="Profile")
        plt.plot(x, fitted_curve, label="Fit", linestyle='--')
        plt.legend()
        plt.title(f"Vertical Profile Fitting for {a[0]}")
        # plt.show()
        plt.close()

    # Plot width changes with error bars
    plt.errorbar(numbers, widths, yerr=width_errors, fmt='o', color = 'black', label="Width Variation with Errors")
    plt.xlabel("File number")
    plt.ylabel("Width")
    plt.title("Width Variation vs File Number (with Error Bars)")
    plt.legend()
    plt.show()
    
    # Perform Lorentzian fitting on the width variation
    numbers = np.array(numbers)
    widths = np.array(widths)
    width_fit_y, popt, perr = FitWidthChange(numbers, widths)

    # Plot the fitted curve
    plt.plot(numbers, width_fit_y, label=f"Lorentzian Fit peak={popt[1]:.2f}", color='red', linestyle='--')
    plt.errorbar(numbers, widths, yerr=width_errors, fmt='o', label="Width Variation with Errors")
    plt.xlabel("File number")
    plt.ylabel("Width")
    plt.title("Width Variation with Lorentzian Fitting")
    plt.legend()
    graphlog = 'resultfig'
    folder_path = os.path.join(graphlog, foldername)
    os.makedirs(folder_path, exist_ok=True)

    # 画像の保存
    fig_name = os.path.join(folder_path, f'{dir_list[num]}.png')
    plt.savefig(fig_name, dpi=300)
    plt.show()  # 画像を表示
    plt.close()

    # Display the fit parameters and errors
    print(f"Fit parameters: a={popt[0]:.2f}, mu={popt[1]:.2f}, sigma={popt[2]:.2f}, c={popt[3]:.2f}")
    print(f"Fit errors: a_err={perr[0]:.2f}, mu_err={perr[1]:.2f}, sigma_err={perr[2]:.2f}, c_err={perr[3]:.2f}")

    # Find the file number corresponding to the maximum width (mu from Lorentz fit)
    max_width_number = popt[1]
    print(f"The file number with the largest width (from Lorentz fit) is: {max_width_number}")
#%%
# Example usage
foldername = "0-10-0-8-0/wx"
dir_path = "/Users/miyamotomanari/Desktop/2024_Experiment/20241213_aroundhole/20241211/" + foldername
dir_list = os.listdir(dir_path)

print(dir_list)
# path = dir_path + '/' + dir_list[1]
# files = glob.glob(path + "/*_capImg.npy")
# d = np.load(files[0])
# print(d.shape[0])
#%%
# Call the function with sorted files
v_start, v_end = 20, 80

# for i in range(len(dir_list)):
#         func_v (i)
func_v(4)
# for i in dir_list:
#     path = dir_path + '/' + dir_list[1]
#     files = glob.glob(path + "/*_capImg.npy")
#     d = np.load(files[0])
#     d = d.shape[0]
#     if d == 200:
#         func_v(i)
#     else :
#         func_h(i)

