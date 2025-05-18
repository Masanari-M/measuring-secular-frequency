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

# Generate y-profile by summing along another axis
def MakeYProfile(data):
    np_yprofile = np.sum(data, axis=1)
    return np_yprofile

# Fit y-profile using Lorentzian function
def FittingYprofile(yprofile, inverted=False):
    y = np.arange(0, len(yprofile))
    z = yprofile
    
    init_params = np.array([3000000, 50, 20, 1500000])  # 初期パラメータ
    if inverted:
        popt, _ = curve_fit(Inverted_Lorentz_func, y, z, p0=init_params, maxfev=2000)
        _fit = Inverted_Lorentz_func(y, *popt)
    else:
        popt, _ = curve_fit(Lorentz_func, y, z, p0=init_params, maxfev=2000)
        _fit = Lorentz_func(y, *popt)
    
    peak_height = popt[0]  # Get peak or dip height (a)
    width = popt[2]  # Get width (sigma)
    
    return _fit, peak_height, width

# Plot data for Y profiles
def plot_data(ndarr, profile, fit, mode, direction='Y'):
    fig, ax = plt.subplots()
    
    if mode == 'image':
        ax.imshow(ndarr, aspect='auto')
        ax.set_title('Image')
    elif mode == 'profile':
        axis = np.arange(0, len(profile))
        ax.plot(axis, profile)
        ax.set_ylim([np.min(profile), np.max(profile)])
        ax.set_title(f'1D Profile ({direction} direction)')
    elif mode == 'fitting':
        axis = np.arange(0, len(profile))
        ax.plot(axis, profile, label=f'{direction} Profile')
        ax.plot(axis, fit, label=f'{direction} Fitting', linestyle='--')
        ax.set_ylim([np.min(profile), np.max(profile)])
        ax.legend()
        ax.set_title(f'Fitting Result ({direction} direction)')
    
    plt.show()

# Fit and find peak or dip using Lorentzian function
def FitAndFindLorentzPeak(x, y, inverted=True):
    init_params = [max(y), 30, 5, min(y)]
    
    if inverted:
        popt, _ = curve_fit(Inverted_Lorentz_func, x, y, p0=init_params, maxfev=2000)
        fitted_curve = Inverted_Lorentz_func(x, *popt)
    else:
        popt, _ = curve_fit(Lorentz_func, x, y, p0=init_params, maxfev=2000)
        fitted_curve = Lorentz_func(x, *popt)
    
    peak_position = popt[1]  # The mean (mu) gives the peak x-value
    
    return fitted_curve, peak_position

# Folder names
folderNames = ["10-01-10-220x(1)"]
npy_files_list = [glob.glob("./" + folder + "/*.npy") for folder in folderNames]
npy_files_list = [sorted(files) for files in npy_files_list]

min_len = min(len(files) for files in npy_files_list)

peak_height_y_ndarr = np.array([])  # Array for storing Y-direction peak heights
width_y_ndarr = np.array([])  # Array for storing Y-direction width changes
mode = 'image'

# Stack and process data
for i in range(min_len):
    stacked_ndarr = np.zeros_like(np.load(npy_files_list[0][i]))
    stacked_ndarr = stacked_ndarr.astype(np.uint64)
    for files in npy_files_list:
        stacked_ndarr_1 = np.load(files[i]).astype(np.uint64)
        stacked_ndarr += stacked_ndarr_1

    # Y方向のプロファイルを取得してフィッティング
    yprofile = MakeYProfile(stacked_ndarr)
    fit_y, peak_height_y, width_y = FittingYprofile(yprofile, inverted=False)  # Change inverted=True for dips
    peak_height_y_ndarr = np.append(peak_height_y_ndarr, peak_height_y)
    width_y_ndarr = np.append(width_y_ndarr, width_y)
    plot_data(stacked_ndarr, yprofile, fit_y, mode, direction='Y')

# Plot and fit Y-direction peak height variation
fig = plt.figure()
ax = fig.add_subplot(111)
y = np.arange(0, len(peak_height_y_ndarr))
ax.plot(y, peak_height_y_ndarr, label="Y-direction Peak Height Variation")

# Fit Y-direction peak height using Lorentzian fit and find position
fitted_y_curve, peak_y_position = FitAndFindLorentzPeak(y, peak_height_y_ndarr, inverted=True)  # inverted=True にすれば凹みをフィッティング

ax.plot(y, fitted_y_curve, linestyle='--', label=f'Lorentzian Fit (Peak at {peak_y_position:.2f})')
ax.legend()
ax.set_title('Y-direction Peak Height Variation with Lorentzian Fit')
plt.savefig(f'{folderNames[0]}_y_peak_height_variation_with_lorentz_fit.png')
plt.show()

print(f"Y-direction Peak position in height variation: {peak_y_position:.2f}")

# Plot and fit Y-direction width variation
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(y, width_y_ndarr, label="Y-direction Width Variation")

# Fit Y-direction width using Lorentzian fit and find peak position
fitted_y_width_curve, peak_y_width_position = FitAndFindLorentzPeak(y, width_y_ndarr)

ax.plot(y, fitted_y_width_curve, linestyle='--', label=f'Lorentzian Fit (Peak at {peak_y_width_position:.2f})')
ax.legend()
ax.set_title('Y-direction Width Variation with Lorentzian Fit')
plt.savefig(f'{folderNames[0]}_y_width_variation_with_lorentz_fit.png')
plt.show()

