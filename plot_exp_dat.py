import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.optimize import curve_fit
from cell_models import protocols
from os import listdir, mkdir
import csv

import heka_reader
import pyqtgraph as pg



# Capacitance, series resistance, and seal can be found through the browser
# Get metadata from here: bundle.pul[0][0][0][0]


class ExpInformation():
    def __init__(self, f_path, channel, trials):
        self.f_path = f_path
        self.channel = channel
        self.trials = trials

    def plot_all_traces(self, comp_setting='NaIVCP_1'):
        ch = self.channel
        bundle = heka_reader.Bundle(self.f_path)

        num_comps = 0

        fig, axs = plt.subplots(3, 3, figsize=(12, 8), sharex=True, sharey=True)
        axs = axs.flatten()

        proto_num = self.trials['NaIV0_1']

        num_sweeps = bundle.pul[0][proto_num].NumberSweeps

        ch_label = f'Imon-{ch}'
        ch_index = [i for i, trace in enumerate( bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

        cm = bundle.pul[0][proto_num][0][ch_index].CSlow
        g_series = bundle.pul[0][proto_num][0][ch_index].GSeries

        for sweep in range(0, num_sweeps):
            dat = bundle.data[0, proto_num, sweep, ch_index]/1E-9
            t = np.linspace(0, len(dat)/25000, len(dat))
            c_ratio = 1-(sweep/num_sweeps)

            axs[num_comps].plot(t*1000-10, dat, c=(c_ratio, c_ratio, c_ratio))

        comp = 0
        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.axhline(y=0, color='grey', linestyle='-', lw=.5)
            ax.set_title(f'{comp*10}% Compensation')
            ax.set_xlim(-1, 4)

            comp += 1

        num_comps = 1
        comp = 1

        for proto_num in range(self.trials[comp_setting][0], self.trials[comp_setting][1]+1):
            num_sweeps = bundle.pul[0][proto_num].NumberSweeps

            ch_label = f'Imon-{ch}'
            ch_index = [i for i, trace in enumerate(
                            bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

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

    def plot_all_IV(self, comp_setting='NaIVCP_1'):
        ch = self.channel
        bundle = heka_reader.Bundle(self.f_path)

        proto_num = self.trials['NaIV0_1']

        num_sweeps = bundle.pul[0][proto_num].NumberSweeps

        ch_label = f'Imon-{ch}'
        ch_index = [i for i, trace in enumerate( bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

        currents = []
        nav_peaks = []

        cm = bundle.pul[0][proto_num][0][ch_index].CSlow
        g_series = bundle.pul[0][proto_num][0][ch_index].GSeries
        seal_res = bundle.pul[0][proto_num][0][ch_index].SealResistance
        print(f'Cm: {cm/1E-12}')
        print(f'Rseries: {1/g_series/1E6}')
        print(f'Seal: {seal_res/1E9}')

        for sweep in range(0, num_sweeps):
            dat = bundle.data[0, proto_num, sweep, ch_index]/1E-9
            nav_peaks.append(np.min(dat))

        currents.append(nav_peaks)

        for proto_num in range(self.trials[comp_setting][0], self.trials[comp_setting][1]+1):
            num_sweeps = bundle.pul[0][proto_num].NumberSweeps

            ch_label = f'Imon-{ch}'
            ch_index = [i for i, trace in enumerate(
                            bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

            nav_peaks = []
            for sweep in range(0, num_sweeps):
                dat = bundle.data[0, proto_num, sweep, ch_index]/1E-9
                nav_peaks.append(np.min(dat))
                
            currents.append(nav_peaks)

        voltages = np.linspace(-80, 40, len(currents[0])+1)[:-1]

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

    def plot_all_traces_temp(self):
        ch = self.channel
        bundle = heka_reader.Bundle(self.f_path)

        num_comps = 0

        fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
        axs = axs.flatten()

        proto_num = self.trials['NaIV0_25C']

        num_sweeps = bundle.pul[0][proto_num].NumberSweeps

        ch_label = f'Imon-{ch}'
        ch_index = [i for i, trace in enumerate( bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

        cm = bundle.pul[0][proto_num][0][ch_index].CSlow
        g_series = bundle.pul[0][proto_num][0][ch_index].GSeries

        for temp_setting in ['NaIV0_25C', 'NaIV0_35C']:
            proto_num = self.trials[temp_setting]
            for sweep in range(0, num_sweeps):
                dat = bundle.data[0, proto_num, sweep, ch_index]/1E-9
                t = np.linspace(0, len(dat)/25000, len(dat))
                c_ratio = 1-(sweep/num_sweeps)

                if '25C' in temp_setting:
                    axs[num_comps].plot(t*1000-10, dat, c=(0, 0, c_ratio))
                if '35C' in temp_setting:
                    axs[num_comps].plot(t*1000-10, dat, c=(c_ratio, 0, 0))

        comp = 0
        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.axhline(y=0, color='grey', linestyle='-', lw=.5)
            ax.set_title(f'{comp*20}% Compensation')
            ax.set_xlim(-1, 4)

            comp += 1


        for temp_setting in ['NaIVCP_25C', 'NaIVCP_35C']:
            comp = 1
            num_comps = 1
            for proto_num in range(self.trials[temp_setting][0], self.trials[temp_setting][1]+1):
                num_sweeps = bundle.pul[0][proto_num].NumberSweeps

                ch_label = f'Imon-{ch}'
                ch_index = [i for i, trace in enumerate(
                                bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

                cm = bundle.pul[0][proto_num][0][ch_index].CSlow
                g_series = bundle.pul[0][proto_num][0][ch_index].GSeries

                for sweep in range(0, num_sweeps):
                    dat = bundle.data[0, proto_num, sweep, ch_index]/1E-9
                    t = np.linspace(0, len(dat)/25000, len(dat))
                    c_ratio = 1-(sweep/num_sweeps)

                    if '25C' in temp_setting:
                        axs[num_comps].plot(t*1000-10, dat, c=(0, 0, c_ratio))
                    else:
                        axs[num_comps].plot(t*1000-10, dat, c=(c_ratio, 0, 0))

                num_comps += 1

        for y_idx in range(3, 6):
            axs[y_idx].set_xlabel('Time (ms)', fontsize=16)
        for x_idx in [0, 3]:
            axs[x_idx].set_ylabel('nA', fontsize=16)

        fig.suptitle(f'Cm={round(cm/1E-12, 2)}pF, Rs={round(1/g_series/1E6, 2)}')
        plt.show()

    def plot_all_IV_temp(self, comp_setting='NaIVCP_1'):
        ch = self.channel
        bundle = heka_reader.Bundle(self.f_path)

        proto_num = self.trials['NaIV0_25C']

        num_sweeps = bundle.pul[0][proto_num].NumberSweeps

        ch_label = f'Imon-{ch}'
        ch_index = [i for i, trace in enumerate( bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

        currents_25C = []
        nav_peaks_25C = []

        currents_35C = []
        nav_peaks_35C = []

        cm = bundle.pul[0][proto_num][0][ch_index].CSlow
        g_series = bundle.pul[0][proto_num][0][ch_index].GSeries
        seal_res = bundle.pul[0][proto_num][0][ch_index].SealResistance
        print(f'Cm: {cm/1E-12}')
        print(f'Rseries: {1/g_series/1E6}')
        print(f'Seal: {seal_res/1E9}')

        for temp_setting in ['NaIV0_25C', 'NaIV0_35C']:
            proto_num = self.trials[temp_setting]
            for sweep in range(0, num_sweeps):
                dat = bundle.data[0, proto_num, sweep, ch_index]/1E-9
                if '25C' in temp_setting:
                    nav_peaks_25C.append(np.min(dat))
                if '35C' in temp_setting:
                    nav_peaks_35C.append(np.min(dat))
        
        currents_25C.append(nav_peaks_25C)
        currents_35C.append(nav_peaks_35C)

        for temp_setting in ['NaIVCP_25C', 'NaIVCP_35C']:
            for proto_num in range(self.trials[temp_setting][0], self.trials[temp_setting][1]+1):
                num_sweeps = bundle.pul[0][proto_num].NumberSweeps

                ch_label = f'Imon-{ch}'
                ch_index = [i for i, trace in enumerate(
                                bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

                nav_peaks_25C = []
                nav_peaks_35C = []
                for sweep in range(0, num_sweeps):
                    dat = bundle.data[0, proto_num, sweep, ch_index]/1E-9
                    if '25C' in temp_setting:
                        nav_peaks_25C.append(np.min(dat))
                    if '35C' in temp_setting:
                        nav_peaks_35C.append(np.min(dat))

                if '25C' in temp_setting:
                    currents_25C.append(nav_peaks_25C)
                if '35C' in temp_setting:
                    currents_35C.append(nav_peaks_35C)

            voltages = np.linspace(-80, 40, len(currents_25C[0])+1)[:-1]

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))


        for i, peak_navs in enumerate(currents_25C):
            c_ratio = .9-(i/(len(currents_25C)+4))
            ax.plot(voltages, peak_navs, label=i*20,
                        c=(0, 0, c_ratio), marker='o')

        for i, peak_navs in enumerate(currents_35C):
            c_ratio = .9-(i/(len(currents_35C)+4))
            ax.plot(voltages, peak_navs, label=i*20,
                        c=(c_ratio, 0, 0), marker='o')


        fig.suptitle(f'Cm={round(cm/1E-12, 2)}pF, Rs={round(1/g_series/1E6, 2)}')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axhline(y=0, color='grey', linestyle='-', lw=.5)
        ax.set_xlabel('Voltage (mV)', fontsize=18)
        ax.set_ylabel('Current (nA)', fontsize=18)
        plt.legend()
        plt.show()

    def write_csv(self, comp_type='NaIVCP_1'):
        ch = self.channel
        bundle = heka_reader.Bundle(self.f_path)

        num_comps = 0

        proto_num = self.trials['NaIV0_1']

        num_sweeps = bundle.pul[0][proto_num].NumberSweeps

        ch_label = f'Imon-{ch}'
        ch_index = [i for i, trace in enumerate(bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

        # FIX THIS
        column_names = ['Sweep', 'SweepTime', 'Time']
        column_names += [f'Compound({i})' for i in range(1, 5)]
        column_names += [f'Age({i})' for i in range(1, 5)]
        column_names += [f'Concentration({i})' for i in range(1, 5)]
        column_names += [f'Min({i})' for i in range(1, 5)]
        column_names += [f'Voltage({i})' for i in range(1, 5)]
        column_names += [f'Cm({i})' for i in range(1, 5)]
        column_names += [f'Rseal({i})' for i in range(1, 5)]
        column_names += [f'Rseries({i})' for i in range(1, 5)]
        column_names += [f'Ignore({i})' for i in range(1, 5)]

        f_name = self.f_path.split('/')[-1] 
        metadata_path = '.' + self.f_path.split('.')[1][0:-4] + '/' + f'{f_name[0:-4]}_1NaIV.xls'

        meta = pd.read_csv(metadata_path, sep=r'\t', skiprows=1)
        meta.columns = column_names


        for sweep in range(0, num_sweeps):
            cm = bundle.pul[0][proto_num][0][ch_index].CSlow
            g_series = bundle.pul[0][proto_num][0][ch_index].GSeries

            dat = bundle.data[0, proto_num, sweep, ch_index]/1E-9
            t = np.linspace(0, len(dat)/25000, len(dat))

            curr_trace = dat



        comp = 0
        for ax in axs:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.axhline(y=0, color='grey', linestyle='-', lw=.5)
            ax.set_title(f'{comp*10}% Compensation')
            ax.set_xlim(-1, 4)

            comp += 1

        num_comps = 1
        comp = 1

        for proto_num in range(self.trials[comp_setting][0], self.trials[comp_setting][1]+1):
            num_sweeps = bundle.pul[0][proto_num].NumberSweeps

            ch_label = f'Imon-{ch}'
            ch_index = [i for i, trace in enumerate(
                            bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

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

    def write_meta(self):
        ch = self.channel
        bundle = heka_reader.Bundle(self.f_path)

        num_comps = 0
        proto_num = 1

        num_sweeps = bundle.pul[0][proto_num].NumberSweeps

        ch_label = f'Imon-{ch}'
        ch_index = [i for i, trace in enumerate(bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

        # FIX THIS
        column_names = ['Sweep', 'SweepTime', 'Time']
        column_names += [f'Compound({i})' for i in range(1, 5)]
        column_names += [f'Age({i})' for i in range(1, 5)]
        column_names += [f'Concentration({i})' for i in range(1, 5)]
        column_names += [f'Min({i})' for i in range(1, 5)]
        column_names += [f'Voltage({i})' for i in range(1, 5)]
        column_names += [f'Cm({i})' for i in range(1, 5)]
        column_names += [f'Rseries({i})' for i in range(1, 5)]
        column_names += [f'Rseal({i})' for i in range(1, 5)]
        column_names += [f'Ignore({i})' for i in range(1, 5)]

        f_name = self.f_path.split('/')[-1] 
        metadata_path = '.' + self.f_path.split('.')[1][0:-4] + '/' + f'{f_name[0:-4]}_1NaIV.xls'

        meta = pd.read_csv(metadata_path, sep='\t', skiprows=1, header=None)
        meta.columns = column_names

        new_columns = {'Sweep': [], 'Cm': [], 'Rseries': [], 'Rseal': []}

        voltages = list(range(-80, 50, 10))

        aq_num = 1

        for trial_name, nums in self.trials.items():
            if '_1' not in trial_name:
                continue

            if type(nums) != list: 
                nums = [nums]
            else:
                nums = list(range(nums[0], nums[1] + 1))

            for comp_num, trial in enumerate(nums):
                if 'NaIV0' in trial_name:
                    comp = 0
                else: 
                    comp = (comp_num + 1) * 10
                for sweep in range(0, num_sweeps):
                    current_dat = meta[meta['Sweep'] == f'1_{trial+1}_{sweep+1}']
                    cm = current_dat[f'Cm({ch})'].values[0]
                    rseries = current_dat[f'Rseries({ch})'].values[0]
                    rseal = current_dat[f'Rseal({ch})'].values[0]

                    if aq_num > 99:
                        trial_number = aq_num
                    elif aq_num > 9:
                        trial_number = f'0{aq_num}'
                    else:
                        trial_number = f'00{aq_num}'

                    new_columns['Sweep'].append(f'{aq_num}_{trial_name}_{comp}%_{voltages[sweep]}mV')
                    new_columns['Cm'].append(cm)
                    new_columns['Rseries'].append(rseries)
                    new_columns['Rseal'].append(rseal)

                    aq_num += 1

        meta = pd.DataFrame(new_columns)

        meta.to_csv(f'{self.f_path[0:-4]}_csv/metadata.csv', index=False)

    def write_csv_temp(self):
        new_directory = f'{self.f_path[:-4]}_ch{self.channel}_csv'
        mkdir(new_directory)
        ch = self.channel
        bundle = heka_reader.Bundle(self.f_path)

        num_comps = 0

        proto_num = self.trials['NaIV0_25C']

        num_sweeps = bundle.pul[0][proto_num].NumberSweeps

        ch_label = f'Imon-{ch}'
        ch_index = [i for i, trace in enumerate(bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

        # FIX THIS
        column_names = ['Sweep', 'SweepTime', 'Time']
        column_names += [f'Compound({i})' for i in range(1, 5)]
        column_names += [f'Age({i})' for i in range(1, 5)]
        column_names += [f'Concentration({i})' for i in range(1, 5)]

        column_names += [f'Min({i})' for i in range(1, 5)]
        column_names += [f'Voltage({i})' for i in range(1, 5)]
        column_names += [f'Cm({i})' for i in range(1, 5)]
        column_names += [f'Rseries({i})' for i in range(1, 5)]
        column_names += [f'Rseal({i})' for i in range(1, 5)]

        meta_file = self.f_path[:-4].split('/')[-1] + '_1NaIV.xls'
        folder_date = meta_file.split('_')[0]
        meta_path = f'{self.f_path[0:-15]}/{folder_date}/{meta_file}'
        meta = pd.read_csv(meta_path, sep='\t', skiprows=1, header=None)
        meta.drop(columns=meta.columns[-2:], axis=1,  inplace=True)
        meta.columns = column_names
        
        proto_key = {}
        for proto_name, nums in self.trials.items():
            if proto_name == 'NaIV0_25C':
                proto_key['NaIV_25C_0CP'] = nums
            elif proto_name == 'NaIVCP_25C':
                for i, proto_num in enumerate(
                                    range(nums[0], nums[1]+1)):
                    proto_key[f'NaIV_25C_{i*20+20}CP'] = proto_num
            elif proto_name == 'NaIV0_35C':
                proto_key['NaIV_35C_0CP'] = nums
            elif proto_name == 'NaIVCP_35C':
                for i, proto_num in enumerate(
                                    range(nums[0], nums[1]+1)):
                    proto_key[f'NaIV_35C_{i*20+20}CP'] = proto_num
            else:
                exit

        for proto_name, proto_num in proto_key.items(): 
            new_dat = {} 
            new_meta = {'voltage_mV': [], 'capacitance_pF': [], 'rseries_Mohm': [], 'rseal_Mohm': []}
            for sweep in range(0, num_sweeps):
                sweep_meta = meta.loc[meta['Sweep'] == f'1_{proto_num+1}_{sweep+1}']
                voltage = sweep_meta[
                        f'Voltage({self.channel})'].values[0]*1000
                cm = sweep_meta[
                        f'Cm({self.channel})'].values[0] *1E12
                rseries = sweep_meta[
                        f'Rseries({self.channel})'].values[0] /1E6
                rseal = sweep_meta[
                        f'Rseal({self.channel})'].values[0] /1E6

                new_dat[voltage] = bundle.data[
                                0, proto_num, sweep, ch_index]/1E-9

                new_meta['voltage_mV'].append(voltage)
                new_meta['capacitance_pF'].append(cm)
                new_meta['rseries_Mohm'].append(rseries)
                new_meta['rseal_Mohm'].append(rseal)

                #t = np.linspace(0, len(dat)/25000, len(dat))

                #curr_trace = dat
            curr_traces = pd.DataFrame(new_dat)
            curr_meta = pd.DataFrame(new_meta)
            curr_traces.to_csv(f'{new_directory}/{proto_name}.csv', index=False)
            curr_meta.to_csv(f'{new_directory}/{proto_name}_meta.csv', index=False)

def main():
    #cell_012822_006_ch2 = ExpInformation('./data/nav_1-28-2022/220128_006.dat', 2, 
    #                                     {'NaIV0_1': 4,
    #                                      'NaIVCP_1': [5, 12]})
    #cell_012822_006_ch2.plot_all_IV(comp_setting='NaIVCP_1')
    #cell_012822_006_ch2.plot_all_traces(comp_setting='NaIVCP_1')


    #cell_020322_002_ch2 = ExpInformation('./data/nav_2-3-2022/220203_002.dat', 2, 
    #                                     {'NaIV0_1': 12,
    #                                      'NaIVCP_1': [13, 20],
    #                                      'NaIVC_1': [21, 28],
    #                                      'NaIVP_1': [29, 36]})
    #cell_020322_002_ch2.plot_all_IV(comp_setting='NaIVCP_1')
    #cell_020322_002_ch2.plot_all_traces(comp_setting='NaIVCP_1')


    #cell_020322_007_ch2 = ExpInformation('./data/nav_2-3-2022/220203_007.dat', 2, 
    #                                     {'NaIV0_1': 5,
    #                                      'NaIVCP_1': [6, 13],
    #                                      'NaIVC_1': [14, 21],
    #                                      'NaIVP_1': [22, 29]})
    #cell_020322_007_ch2.plot_all_IV(comp_setting='NaIVC_1')
    #cell_020322_007_ch2.plot_all_traces(comp_setting='NaIVC_1')


    #cell_020322_007_ch4 = ExpInformation('./data/nav_2-3-2022/220203_007.dat', 4, 
    #                                     {'NaIV0_1': 5,
    #                                      'NaIVCP_1': [6, 13],
    #                                      'NaIVC_1': [14, 21],
    #                                      'NaIVP_1': [22, 29]})
    #cell_020322_007_ch4.plot_all_traces(comp_setting='NaIVCP_1')
    #cell_020322_007_ch4.plot_all_IV(comp_setting='NaIVCP_1')


    #cell_020322_008_ch1 = ExpInformation('./data/nav_2-3-2022/220203_008.dat', 1, 
    #                                     {'NaIV0_1': 1,
    #                                      'NaIVCP_1': [2, 9],
    #                                      'NaIVC_1': [10, 17],
    #                                      'NaIVP_1': [18, 25]})
    #cell_020322_008_ch1.plot_all_traces(comp_setting='NaIVCP_1')
    #cell_020322_008_ch1.plot_all_IV(comp_setting='NaIVCP_1')


    #cell_020322_008_ch2 = ExpInformation('./data/nav_2-3-2022/220203_008.dat', 2, 
    #                                     {'NaIV0_1': 1,
    #                                      'NaIVCP_1': [2, 9],
    #                                      'NaIVC_1': [10, 17],
    #                                      'NaIVP_1': [18, 25]})
    #cell_020322_008_ch2.plot_all_traces(comp_setting='NaIVCP_1')
    #cell_020322_008_ch2.plot_all_IV(comp_setting='NaIVCP_1')


    #cell_021022_003_ch3 = ExpInformation('./data/nav_2-10-2022/220210_003.dat', 3, 
    #                                     {'NaIV0_1': 1,
    #                                      'NaIVCP_1': [2, 9],
    #                                      'NaIVC_1': [10, 17],
    #                                      'NaIVP_1': [18, 25]})
    #cell_021022_003_ch3.plot_all_traces(comp_setting='NaIVCP_1')
    #cell_021022_003_ch3.plot_all_IV(comp_setting='NaIVCP_1')


    ###March 3

    #cell_031022_001_ch2 = ExpInformation('./data/nav_3-10-2022/220310_001.dat',
    #                                     2, 
    #                                     {'NaIV0_1': 1,
    #                                      'NaIVCP_1': [2, 9]})
    ##cell_031022_001_ch2.plot_all_traces(comp_setting='NaIVCP_1')
    #cell_031022_001_ch2.plot_all_IV(comp_setting='NaIVCP_1')

    #cell_031022_001_ch3 = ExpInformation('./data/nav_3-10-2022/220310_001.dat',
    #                                     3, 
    #                                     {'NaIV0_1': 1,
    #                                      'NaIVCP_1': [2, 9]})
    #cell_031022_001_ch3.plot_all_traces(comp_setting='NaIVCP_1')
    #cell_031022_001_ch3.plot_all_IV(comp_setting='NaIVCP_1')


    #PREFORATED
    #cell_031022_002_ch1 = ExpInformation('./data/nav_3-10-2022/220310_002.dat',
    #                                     1, 
    #                                     {'NaIV0_1': 2,
    #                                      'NaIVCP_1': [3, 10]})
    ##cell_031022_001_ch2.plot_all_traces(comp_setting='NaIVCP_1')
    #cell_031022_002_ch1.plot_all_IV(comp_setting='NaIVCP_1')


    #cell_031022_003_ch1 = ExpInformation('./data/nav_3-10-2022/220310_003.dat',
    #                                     2, 
    #                                     {'NaIV0_1': 1,
    #                                      'NaIVCP_1': [2, 9]})
    ##cell_031022_003_ch2.plot_all_traces(comp_setting='NaIVCP_1')
    #cell_031022_003_ch1.plot_all_IV(comp_setting='NaIVCP_1')


    #cell_031022_004_ch1 = ExpInformation('./data/nav_3-10-2022/220310_004.dat',
    #                                     2, 
    #                                     {'NaIV0_1': 1,
    #                                      'NaIVCP_1': [2, 5]})
    ##cell_031022_004_ch2.plot_all_traces(comp_setting='NaIVCP_1')
    #cell_031022_004_ch1.plot_all_IV(comp_setting='NaIVCP_1')


    ### MARCH 14
    #EXCELLENT!!
    #cell_031422_001_ch1 = ExpInformation('./data/nav_3-14-2022/220314_001.dat',
    #                                     1, 
    #                                     {'NaIV0_25C': 1,
    #                                      'NaIVCP_25C': [2, 5],
    #                                      'NaIV0_35C': 6,
    #                                      'NaIVCP_35C': [7, 10]})
    #cell_031422_001_ch1.plot_all_traces_temp()
    #cell_031422_001_ch1.plot_all_IV_temp()
    #cell_031422_001_ch1.write_csv_temp()

    #cell_031422_001_ch4 = ExpInformation('./data/nav_3-14-2022/220314_001.dat',
    #                                     4, 
    #                                     {'NaIV0_25C': 1,
    #                                      'NaIVCP_25C': [2, 5],
    #                                      'NaIV0_35C': 6,
    #                                      'NaIVCP_35C': [7, 10]})
    #cell_031422_001_ch4.plot_all_traces_temp()
    #cell_031422_001_ch4.plot_all_IV_temp()


    #cell_031422_003_ch1 = ExpInformation('./data/nav_3-14-2022/220314_003.dat',
    #                                     1, 
    #                                     {'NaIV0_25C': 1,
    #                                      'NaIVCP_25C': [2, 5],
    #                                      'NaIV0_35C': 6,
    #                                      'NaIVCP_35C': [7, 10]})
    #cell_031422_003_ch1.plot_all_traces_temp()
    #cell_031422_003_ch1.plot_all_IV_temp()

    #EXCELLENT!!!
    cell_031422_003_ch2 = ExpInformation('./data/nav_3-14-2022/220314_003.dat',
                                         2, 
                                         {'NaIV0_25C': 1,
                                          'NaIVCP_25C': [2, 5],
                                          'NaIV0_35C': 6,
                                          'NaIVCP_35C': [7, 10]})
    cell_031422_003_ch2.plot_all_traces_temp()
    cell_031422_003_ch2.plot_all_IV_temp()
    #cell_031422_003_ch2.write_csv_temp()


    #EXCELLENT!!
    cell_031422_004_ch2 = ExpInformation('./data/nav_3-14-2022/220314_004.dat',
                                         2, 
                                         {'NaIV0_25C': 1,
                                          'NaIVCP_25C': [2, 5],
                                          'NaIV0_35C': 6,
                                          'NaIVCP_35C': [7, 10]})
    cell_031422_004_ch2.plot_all_traces_temp()
    cell_031422_004_ch2.plot_all_IV_temp()
    #cell_031422_004_ch2.write_csv_temp()





if __name__ == "__main__":
    main()
