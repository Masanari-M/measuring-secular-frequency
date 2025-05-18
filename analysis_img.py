#Put this file at same directri with the folder includes npy file
import numpy as np
import glob
import matplotlib.pyplot as plt

files = glob.glob("./test/*_capImg.npy")
#image size
nda = np.zeros((200, 80))
#nda = np.zeros((100, 200))

for f in files:
    nda_temp = np.load(f)
    nda = nda + nda_temp
    
    nda = nda/len(files)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(nda)
    plt.show()
    
