from pylablib.devices import uc480
import pylablib as pll
import matplotlib.pyplot as plt
import pyvisa
import numpy as np
import time

# create instance
rm = pyvisa.ResourceManager()
cam = uc480.UC480Camera()
lst_cam = uc480.list_cameras()
print(lst_cam)

# open camera
cam.open()

# set trigger as software
cam.set_trigger_mode('software')

# set exposure time
cam.set_exposure(0.05) # in [sec.]
current_exposure = cam.get_exposure()
print('exposure time (ms) : ', current_exposure * 10**3)

# Range of Image
#wx
# v_start = 310 # min 0
# v_end = 510 # max 1024
# h_start = 640 # min 0
# h_end = 740 # max 1280.

# #wz
v_start = 370 # min 0
v_end = 570 # max 1024
h_start = 700 # min 0
h_end = 780 # max 1280
cam.set_roi(hstart=h_start
            , hend=h_end
            , vstart=v_start
            , vend=v_end
            , hbin=1, vbin=1)
current_roi = cam.get_roi()
print(current_roi)


#%%control FG
__fg = rm.open_resource('USB0::0x0699::0x0353::2132381::INSTR')
__fg.write('OUTP1:STAT OFF')

start_freq = 1703
end_freq = 1750
step_freq = 0.1

frequency_list = np.arange(start_freq, end_freq, step_freq)

for i in frequency_list:
    __fg.write('SOURce1:FREQuency:CW %.1fkHz' %i)
    start_time = time.time()
    img = cam.snap() # type: numpy.ndarray
    np.save('./%1f_capImg.npy' %i, img)
    end_time = time.time()
    time_diff = end_time - start_time
    print('frame rate (ms) : ', time_diff * 10**(3))
    time.sleep(1)

print('finish')

__fg.close()
rm.close()
#%%
cam.close()