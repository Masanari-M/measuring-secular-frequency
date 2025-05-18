# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:11:50 2024

@author: oshio
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import re
from PIL import Image
from lmfit.models import LorentzianModel
import os
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.cm as cm
import statistics
import math

def func(x, *params):

    #paramsの長さでフィッティングする関数の数を判別。
    num_func = int(len(params)/3)

    #ガウス関数にそれぞれのパラメータを挿入してy_listに追加。
    y_list = []
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(3*i,3*(i+1),1))
        amp = params[int(param_range[0])]
        ctr = params[int(param_range[1])]
        wid = params[int(param_range[2])]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
        y_list.append(y)

    #y_listに入っているすべてのガウス関数を重ね合わせる。
    y_sum = np.zeros_like(x)
    for i in y_list:
        y_sum = y_sum + i

    #最後にバックグラウンドを追加。
    y_sum = y_sum + params[-1]

    return y_sum

def fit_plot(x, *params):
    num_func = int(len(params)/3)
    y_list = []
    for i in range(num_func):
        y = np.zeros_like(x)
        param_range = list(range(3*i,3*(i+1),1))
        amp = params[int(param_range[0])]
        ctr = params[int(param_range[1])]
        wid = params[int(param_range[2])]
        y = y + amp * np.exp( -((x - ctr)/wid)**2) + params[-1]
        y_list.append(y)
    return y_list

def gaussfit(profile):
    #初期値のリストを作成
    #[amp,ctr,wid]
    guess = []
    guess.append([max(profile), 110, 50])
    #バックグラウンドの初期値
    background = statistics.mean(profile)
    #初期値リストの結合
    guess_total = []
    y = profile
    x=np.linspace(0,len(profile)-1,len(profile))
    
    for i in guess:
        guess_total.extend(i)
    guess_total.append(background)
    popt, pcov = curve_fit(func, x, y, p0=guess_total)
    
    fit = func(x, *popt)
    plt.scatter(x, y, s=20)
    plt.plot(x, fit , ls='-', c='black', lw=1)
    
    y_list = fit_plot(x, *popt)
    baseline = np.zeros_like(x) + popt[-1]
    
    plt.xlabel("z coordinate [pix]");plt.ylabel("intensity [a.u]")
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    for n,i in enumerate(y_list):
        plt.fill_between(x, i, baseline, facecolor=cm.rainbow(n/len(y_list)), alpha=0.6)
    
    print(popt)#最適化推定値[a, μ ,sigma]
    print(np.sqrt(np.diag(pcov)))#パラメータの分散共分散行列[Δa,Δμ, Δsigma]
    plt.show()
    FWHM=2*popt[2]*np.sqrt(2*np.log(2))
    return abs(FWHM)


dir_path = ('./20241009/154415')
dir_list = os.listdir(dir_path)
print(dir_list)

v_start = 30
v_end = 60

path = dir_path + '/' + dir_list[0]
files = glob.glob(path + "/*_capImg.npy")

ionwidthlist=[]
freqlist=[]

for f in files:
    print(f)
    nda = np.zeros((100, 200))
    nda = np.load(f)
    nda[nda<=10]
    x0 = []
    for i in np.linspace(v_start,v_end,v_end - v_start + 1):
        x0.append(nda[int(i)])
    x1 = np.sum(x0,axis = 0)
    ionwidth=gaussfit(x1)
    ionwidthlist.append(ionwidth)
    a = re.findall('(.*?)00000_capImg',f)
    b=a[0][len(path)+1:]
    freqlist.append(float(b))

#%%

x=np.array(freqlist)
y=np.array(ionwidthlist)
y[y>=200]=0
model = LorentzianModel()
params = model.guess(y, x=x)

result = model.fit(y, params, x=x)

print(result.fit_report())

fig, ax = plt.subplots(dpi=130)
result.plot_fit(ax=ax)
ax.set(xlabel="frequency [kHz]",ylabel="FWHM")
plt.show()