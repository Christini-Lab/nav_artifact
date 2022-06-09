import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
from cell_models import protocols
from os import listdir
from scipy import fft
from scipy import ifft

import heka_reader
import pyqtgraph as pg



# Capacitance, series resistance, and seal can be found through the browser
# Get metadata from here: bundle.pul[0][0][0][0]

def get_all_ap_data(folder, file, channel):
    bundle = heka_reader.Bundle(f'data/{folder}/{file}')
    fig = None
    # pos 0: Group (Always 1)
    # pos 1: Protocol
    # pos 2: Sweep ... who knows 
    # pos 3: Channel 1 -> 4

    proto_name = 'CCstim1Hz'
    protos = [p.Label for p in bundle.pul[0].children]

    protos.reverse()
    idx = len(protos) - protos.index(proto_name) - 1

    num_sweeps = bundle.pul[0][idx].NumberSweeps

    all_ap = []

    for sweep in range(0, num_sweeps-1):
        all_ap.append(bundle.data[0, idx, sweep, channel])

    return all_ap


def filter_aps(all_aps):
    """ Filters the data using notch filter

    Description:
        Digital filter which returns the filtered signal using 60Hz
        notch filter. Transforms the signal into frequency domain
        using the fft function of the Scipy Module. Then, suppresses
        the 60Hz signal by equating it to zero. Finally, transforms
        the signal back to time domain using the ifft function.
    Input:
        ECGdata -- list of integers (ECG data)
    Output:
        ifft(fftECG) -- inverse fast fourier transformed array of filtered ECG data
    """
    filtered_aps = []

    for ap in all_aps:
        fftAP = fft(ap)

        
        for i in range(len(fftAP)):
            if 5000<i<22000: fftAP[i]=0
            
        filteredAP = ifft(fftAP)

        filtered_aps.append(filteredAP)
        


    return filtered_aps

folder = 'iPSC_test_1-14-22'
file = '220114_001.dat'
channel = 0
all_aps = get_all_ap_data(folder, file, channel)

aps14 = filter_aps(all_aps)

folder = 'iPSC_test_1-18-22'
file = '220118_004.dat'
channel = 0
all_aps = get_all_ap_data(folder, file, channel)
aps18 = filter_aps(all_aps)




fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 8))

t = np.linspace(0, 1000, 25000)

for i, ap in enumerate(aps14):
    c_ratio = 1-(i/len(aps14))
    axs[0].plot(t[0:10000]-50, 1000*np.array(ap[0:10000]), c=(c_ratio, c_ratio, c_ratio))

t = np.linspace(0, 1000, 25050)

for i, ap in enumerate(aps18):
    c_ratio = 1-(i/len(aps18))
    axs[1].plot(t[0:10000]-50, 1000*np.array(ap[0:10000]), c=(c_ratio, c_ratio, c_ratio))

axs[0].set_ylabel('Voltage (mV)', fontsize=18)
axs[0].set_xlabel('Time (ms)', fontsize=18)
axs[1].set_xlabel('Time (ms)', fontsize=18)

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.rcParams['svg.fonttype'] = 'none'
plt.savefig(f'two_cell_aps.svg', format='svg')

plt.show()




