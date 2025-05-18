import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import re

#files = glob.glob("./20241009/1610/*_capImg.npy")
dir_path = "/Users/miyamotomanari/Desktop/2024_Experiment/20241009_frequency/20241009/V=10-0-10(220)"
dir_list=os.listdir(dir_path)
print(dir_list)
#%%

def func_v(num):
    path = dir_path + '/' + dir_list[num]
    
    files = glob.glob(path + "/*_capImg.npy")
    

    v_start=20
    v_end=70
    for f in files:
        #nda = np.zeros((100,200))
        nda = np.zeros((200, 100))

        nda_temp = np.load(f)
        nda = nda + nda_temp
        
        nda = nda/len(files)
        nda[nda<=10]=0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(nda)
        a = re.findall('(.*?)00000_capImg',f)
        ax.set_title(a[0][len(path)+1:])
        plt.show()
        nda= nda.T
        x0 = []
        for i in np.linspace(v_start,v_end,v_end - v_start + 1):
            x0.append(nda[int(i)])
        x0 = np.array(x0).T
        x1 = np.sum(x0,axis = 1)
        
        
        
        plt.plot(x1, np.linspace(0,199,200))
        plt.show()
        
def func_h(num):
    path = dir_path + '/' + dir_list[num]
    
    files = glob.glob(path + "/*_capImg.npy")
    

    v_start=20
    v_end=70
    for f in files:
        nda = np.zeros((100,200))
     

        nda_temp = np.load(f)
        nda = nda + nda_temp
        
        nda = nda/len(files)
        nda[nda<=10]=0
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(nda)
        a = re.findall('(.*?)00000_capImg',f)
        ax.set_title(a[0][len(path)+1:])
        plt.show()
        x0 = []
        for i in np.linspace(v_start,v_end,v_end - v_start + 1):
            x0.append(nda[int(i)])
        x1 = np.sum(x0,axis = 0)
        
        plt.plot(np.linspace(0,199,200),x1)
        plt.show()
        
func_h(-1)
print(dir_list[-1])
# for i in  :
#     func(i)