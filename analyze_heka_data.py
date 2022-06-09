import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
from cell_models import protocols
from os import listdir

import heka_reader
import pyqtgraph as pg



# Capacitance, series resistance, and seal can be found through the browser
# Get metadata from here: bundle.pul[0][0][0][0]


def plot_all_traces(file_name, proto_name, is_shown, xlims=None):
    bundle = heka_reader.Bundle(file_name)
    fig = None
    # pos 0: Group (Always 1)
    # pos 1: Protocol
    # pos 2: Sweep ... who knows 
    # pos 3: Channel 1 -> 4

    protos = [p.Label for p in bundle.pul[0].children]

    for proto_num, proto in enumerate(protos):
        if proto != proto_name:
            if not ((proto_name == 'Opt_iPSC') and (proto == 'CTRL1')):
                continue

        fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axs = axs.flatten()

        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.axhline(y=0, color='grey', linestyle='-', lw=.5)

        ex_sweep_dat = bundle.data[0, proto_num, 0, 0]
        time = np.linspace(0, len(ex_sweep_dat)/25000, len(ex_sweep_dat))*1000
        idx_start = xlims[0] * 25
        idx_end = xlims[1] * 25

        num_sweeps = bundle.pul[0][proto_num].NumberSweeps

        for sweep in range(0, num_sweeps-1):
            for ch in range(0, 4):
                if proto == 'CCstim1Hz':
                    c_ratio = 1-(sweep/num_sweeps)
                    axs[ch].plot(time[idx_start:idx_end]-50,
                            bundle.data[0, proto_num, sweep, ch][idx_start:idx_end]*1000,
                            c=(c_ratio, c_ratio, c_ratio))

                if ((proto == 'Opt_iPSC') or (proto == 'CTRL1') or (proto == 'External (Position 2, ')):
                    cm = bundle.pul[0][proto_num][sweep][ch].CSlow
                    axs[ch].plot(time,
                            bundle.data[0, proto_num, sweep, ch]/cm,
                            'k')
                #axs[ch].plot(time, bundle.data[0, proto_num, sweep, ch], 'k')

        for ch in range(0, 4):
            if proto == 'CCstim1Hz':
                axs[ch].set_ylabel(f'Ch: {ch+1}, mV', fontsize=12)
            else:
                axs[ch].set_ylabel(f'Ch: {ch+1}, A/F', fontsize=12)
            axs[ch].set_xlabel('Time (ms)', fontsize=12)
        fig.suptitle(file_name.split('/')[-1].split('.')[0], fontsize=12)
        if is_shown:
            plt.show()

        #axs[ch].set_xlim(49, 65)



        #plt.show()

    return fig


def save_all_ap_data(folder, is_shown=False):
    proto_key = {'dc_aps': 'CCstim1Hz',
                 'dc_upstroke': 'CCstim1Hz'}
    xlim_key = {'dc_aps': [0, 350],
                'dc_upstroke': [49, 65]}

    which_data = 'dc_aps' # dc_aps, dc_upstroke, vc

    path = f'./data/{folder}'
    files = listdir(path)
    files.sort()

    for which_data in ['dc_aps', 'dc_upstroke']:
        fig_objects = []

        for f in files:
            if '.dat' in f:
                file_name = f'{path}/{f}'
                print(file_name)
                bundle = heka_reader.Bundle(file_name)

                new_fig = plot_all_traces(file_name,
                                          proto_name=proto_key[which_data],
                                              xlims=xlim_key[which_data],
                                              is_shown=is_shown)
                if new_fig:
                    fig_objects.append(new_fig)

        import matplotlib.backends.backend_pdf
        pdf = matplotlib.backends.backend_pdf.PdfPages(f"{path}/{which_data}.pdf")

        for fig in fig_objects:
            pdf.savefig( fig )
        pdf.close()


def save_all_vc_data(folder, is_shown=False):
    proto = 'External (Position 2, '
    xlim = [0, 10000]

    path = f'./data/{folder}'
    files = listdir(path)
    files.sort()

    fig_objects = []

    for f in files:
        if '.dat' in f:
            file_name = f'{path}/{f}'
            print(file_name)
            bundle = heka_reader.Bundle(file_name)

            new_fig = plot_all_traces(file_name,
                                      proto_name=proto,
                                      xlims=xlim,
                                      is_shown=is_shown)
            if new_fig:
                fig_objects.append(new_fig)

    import matplotlib.backends.backend_pdf
    pdf = matplotlib.backends.backend_pdf.PdfPages(f"{path}/vc.pdf")

    for fig in fig_objects:
        pdf.savefig( fig )
    pdf.close()


def plot_nav_traces(f_path, ch, sweep_range):
    #For f'220128_006.dat', channel 2, Series 5 -> 14
    # Series
    bundle = heka_reader.Bundle(f_path)
    # pos 0: Group (Always 1)
    # pos 1: Protocol
    # pos 2: Sweep ... who knows 
    # pos 3: Channel 1 -> 4

    protos = [p.Label for p in bundle.pul[0].children]

    num_comps = 0

    fig, axs = plt.subplots(3, 3, figsize=(12, 8), sharex=True, sharey=True)
    axs = axs.flatten()

    comp = 0
    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axhline(y=0, color='grey', linestyle='-', lw=.5)
        ax.set_title(f'{comp*10}% Compensation')
        ax.set_xlim(-1, 4)

        comp += 1


    for proto_num in range(0, len(protos)):
        if proto_num not in list(sweep_range):
            continue

        num_sweeps = bundle.pul[0][proto_num].NumberSweeps

        ch_label = f'Imon-{ch}'
        ch_index = [i for i, trace in
                      enumerate(bundle.pul[0][
                                proto_num][list(sweep_range)[0]].children)
                                                  if ch_label == trace.Label][0]

        cm = bundle.pul[0][proto_num][0][ch_index].CSlow
        g_series = bundle.pul[0][proto_num][0][ch_index].GSeries

        for sweep in range(0, num_sweeps):
            dat = bundle.data[0, proto_num, sweep, ch_index]/1E-9
            t = np.linspace(0, len(dat)/25000, len(dat))
            c_ratio = 1-(sweep/num_sweeps)

            axs[num_comps].plot(t*1000-10, dat, c=(c_ratio, c_ratio, c_ratio))

        num_comps += 1

    for y_idx in range(6, 9):
        axs[y_idx].set_xlabel('Time (ms)', fontsize=16)
    for x_idx in [0, 3, 6]:
        axs[x_idx].set_ylabel('nA', fontsize=16)

    fig.suptitle(f'Cm={round(cm/1E-12, 2)}pF, Rs={round(1/g_series/1E6, 2)}')
    plt.show()


def plot_nav_iv(f_path, ch, sweep_range):
    #For f'220128_006.dat', channel 2, Series 5 -> 14
    # Series
    bundle = heka_reader.Bundle(f_path)
    # pos 0: Group (Always 1)
    # pos 1: Protocol
    # pos 2: Sweep ... who knows 
    # pos 3: Channel 1 -> 4

    protos = [p.Label for p in bundle.pul[0].children]

    num_comps = 0

    currents = []

    for proto_num in range(0, len(protos)):
        if proto_num not in list(sweep_range):
            continue



        ch_label = f'Imon-{ch}'
        ch_index = [i for i, trace in
                      enumerate(bundle.pul[0][
                                proto_num][list(sweep_range)[0]].children)
                                                  if ch_label == trace.Label][0]

        cm = bundle.pul[0][proto_num][0][ch_index].CSlow
        g_series = bundle.pul[0][proto_num][0][ch_index].GSeries
        rs = bundle.pul[0][proto_num][0][ch_index].RsValue
        print(f'Comp: {num_comps}')
        print(f'\tCSlow is: {cm*1E12} pF')
        print(f'\t1/GSeries is: {1/g_series/1E6} M')
        print(f'\tRsValue is: {rs/1E6} M')
        num_sweeps = bundle.pul[0][proto_num].NumberSweeps

        nav_peaks = []
        for sweep in range(0, num_sweeps):
            dat = bundle.data[0, proto_num, sweep, ch_index]/1E-9
            nav_peaks.append(np.min(dat))
        
        currents.append(nav_peaks)
        num_comps += 1

    voltages = np.linspace(-80, 40, 13)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    for i, peak_navs in enumerate(currents):
        c_ratio = .9-(i/(len(currents)+4))

        ax.plot(voltages, peak_navs, label=i*10,
                        c=(c_ratio, c_ratio, c_ratio), marker='o')

    fig.suptitle(f'Cm={round(cm/1E-12, 2)}pF, Rs={round(1/g_series/1E6, 2)}')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axhline(y=0, color='grey', linestyle='-', lw=.5)
    ax.set_xlabel('Voltage (mV)', fontsize=18)
    ax.set_ylabel('Current (nA)', fontsize=18)
    plt.legend()
    plt.show()


    #fig, axs = plt.subplots(3, 3, figsize=(12, 8), sharex=True)
    #axs = axs.flatten()

    #for ax in axs:
    #    ax.spines['right'].set_visible(False)
    #    ax.spines['top'].set_visible(False)
    #    ax.axhline(y=0, color='grey', linestyle='-', lw=.5)

    #axs[num_comps].plot(t*1000-10, dat, c=(c_ratio, c_ratio, c_ratio))

    #for y_idx in range(6, 9):
    #    axs[y_idx].set_xlabel('nA')
    #for x_idx in [0, 3, 6]:
    #    axs[x_idx].set_xlabel('Time (ms)')



    plt.show()



#Good cells:
# 'data/nav_1-28-2022/220128_006.dat', ch=2, sweep_range=range(4,13)


#Good examples of BAD recordings
# f_path='data/nav_1-27-2022/220127_004.dat', ch=2, sweep_range=range(1,10)
# f_path='data/nav_1-27-2022/220127_006.dat', ch=3, sweep_range=range(1,10)

plot_nav_traces(f_path='data/nav_1-27-2022/220127_006.dat', ch=3,
                                                    sweep_range=range(1,10))

plot_nav_iv(f_path='data/nav_1-27-2022/220127_006.dat', ch=3,
                                                    sweep_range=range(1,10))
