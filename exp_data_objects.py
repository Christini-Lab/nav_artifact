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



class ExpInformation():
    def __init__(self, cell_file, meta_path, channel, ljp, trials):
        self.cell_file = cell_file 
        self.meta_path = meta_path
        self.channel = channel
        self.trials = trials
        self.ljp = ljp


    def plot_all_iv_traces(self, write_to=None):
        ch = self.channel
        bundle = heka_reader.Bundle(self.cell_file)

        proto_num = self.trials['NaIV0']

        num_sweeps = bundle.pul[0][proto_num].NumberSweeps

        ch_label = f'Imon-{ch}'
        ch_index = [i for i, trace in enumerate( bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

        cm = bundle.pul[0][proto_num][0][ch_index].CSlow
        g_series = bundle.pul[0][proto_num][0][ch_index].GSeries
        seal_res = bundle.pul[0][proto_num][0][ch_index].SealResistance
        print(f'Cm: {cm/1E-12}')
        print(f'Rseries: {1/g_series/1E6}')
        print(f'Seal: {seal_res/1E9}')

        fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 8))
        axs = axs.flatten()

        proto_names = ['NaIV0', 'NaIV20', 'NaIV40', 'NaIV60', 'NaIV80']
        

        for i, proto_name in enumerate(proto_names):
            proto_num = self.trials[proto_name]
            if proto_num is None:
                continue
            num_sweeps = bundle.pul[0][proto_num].NumberSweeps

            ch_label = f'Imon-{ch}'
            ch_index = [i for i, trace in enumerate(
                            bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

            for sweep in range(0, num_sweeps):
                dat = bundle.data[0, proto_num, sweep, ch_index]/1E-9
                t = 1000*(np.linspace(0, len(dat)/25000, len(dat)))
                axs[i].plot(t, dat, 'k')
            axs[i].set_title(f'Compensation {i*20}%')

        for i, ax in enumerate(axs):
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if i>2:
                ax.set_xlabel('Time (ms)')
            if (i == 0) or (i == 3):
                ax.set_ylabel('Current (nA)')
            #ax.axhline(y=0, color='grey', linestyle='-', lw=.5)
            #ax.set_xlabel('Voltage (mV)', fontsize=18)
            #ax.set_ylabel('Current (nA)', fontsize=18)

        ax.set_xlim(8, 12)

        if write_to is not None:
            plt.savefig(write_to)
        else:
            plt.show()


    def plot_all_inact_traces(self, write_to=None):
        ch = self.channel
        bundle = heka_reader.Bundle(self.cell_file)

        proto_num = self.trials['NaInact0']

        num_sweeps = bundle.pul[0][proto_num].NumberSweeps

        ch_label = f'Imon-{ch}'
        ch_index = [i for i, trace in enumerate( bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

        cm = bundle.pul[0][proto_num][0][ch_index].CSlow
        g_series = bundle.pul[0][proto_num][0][ch_index].GSeries
        seal_res = bundle.pul[0][proto_num][0][ch_index].SealResistance
        print(f'Cm: {cm/1E-12}')
        print(f'Rseries: {1/g_series/1E6}')
        print(f'Seal: {seal_res/1E9}')

        fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(12, 8))
        axs = axs.flatten()

        proto_names = ['NaInact0', 'NaInact20', 'NaInact40', 'NaInact60', 'NaInact80']
        
        for i, proto_name in enumerate(proto_names):
            proto_num = self.trials[proto_name]
            if proto_num is None:
                continue
            num_sweeps = bundle.pul[0][proto_num].NumberSweeps

            ch_label = f'Imon-{ch}'
            ch_index = [i for i, trace in enumerate(
                            bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

            for sweep in range(0, num_sweeps):
                dat = bundle.data[0, proto_num, sweep, ch_index]/1E-9
                t = 1000*(np.linspace(0, len(dat)/25000, len(dat)))
                axs[i].plot(t, dat, 'k')
            axs[i].set_title(f'Compensation {i*20}%')

        for i, ax in enumerate(axs):
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            if i>2:
                ax.set_xlabel('Time (ms)')
            if (i == 0) or (i == 3):
                ax.set_ylabel('Current (nA)')
            #ax.axhline(y=0, color='grey', linestyle='-', lw=.5)
            #ax.set_xlabel('Voltage (mV)', fontsize=18)
            #ax.set_ylabel('Current (nA)', fontsize=18)

        ax.set_xlim(1008, 1012)

        if write_to is not None:
            plt.savefig(write_to)
        else:
            plt.show()


    def plot_all_IV(self, write_to=None):
        ch = self.channel
        bundle = heka_reader.Bundle(self.cell_file)

        proto_num = self.trials['NaIV0']

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
            rmp_curr = np.average(dat[200:220]) #average current at RMP
            min_curr = np.min(dat[200:400])
            max_curr = np.max(dat[200:400])

            if (max_curr - rmp_curr) > (rmp_curr - min_curr):
                nav_peaks.append(max_curr)
            else:
                nav_peaks.append(min_curr)

        currents.append(nav_peaks)

        proto_names = ['NaIV20', 'NaIV40', 'NaIV60', 'NaIV80']

        for proto_name in proto_names:
            proto_num = self.trials[proto_name]
            if proto_num is None:
                continue
            num_sweeps = bundle.pul[0][proto_num].NumberSweeps

            ch_label = f'Imon-{ch}'
            ch_index = [i for i, trace in enumerate(
                            bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

            nav_peaks = []
            for sweep in range(0, num_sweeps):
                dat = bundle.data[0, proto_num, sweep, ch_index]/1E-9
                rmp_curr = np.average(dat[200:220]) #average current at RMP
                min_curr = np.min(dat[200:400])
                max_curr = np.max(dat[200:400])

                if (max_curr - rmp_curr) > (rmp_curr - min_curr):
                    nav_peaks.append(max_curr)
                else:
                    nav_peaks.append(min_curr)
                
            currents.append(nav_peaks)

        voltages = np.linspace(-80-self.ljp, 90-self.ljp, len(currents[0])+1)[:-1]

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        for i, peak_navs in enumerate(currents):
            c_ratio = .9-(i/(len(currents)))

            ax.plot(voltages, peak_navs, label=i*20,
                            c=(c_ratio, c_ratio, c_ratio), marker='o')

        fig.suptitle(f'Cm={round(cm/1E-12, 2)}pF, Rs={round(1/g_series/1E6, 2)}')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axhline(y=0, color='grey', linestyle='-', lw=.5)
        ax.set_xlabel('Voltage (mV)', fontsize=18)
        ax.set_ylabel('Current (nA)', fontsize=18)
        plt.legend()

        if write_to is not None:
            plt.savefig(write_to)
        else:
            plt.show()


    def plot_all_inact(self, write_to=None):
        ch = self.channel
        bundle = heka_reader.Bundle(self.cell_file)

        proto_num = self.trials['NaInact0']

        num_sweeps = bundle.pul[0][proto_num].NumberSweeps
        all_inact = []

        ch_label = f'Imon-{ch}'
        ch_index = [i for i, trace in enumerate( bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

        cm = bundle.pul[0][proto_num][0][ch_index].CSlow
        g_series = bundle.pul[0][proto_num][0][ch_index].GSeries
        seal_res = bundle.pul[0][proto_num][0][ch_index].SealResistance
        print(f'Cm: {cm/1E-12}')
        print(f'Rseries: {1/g_series/1E6}')
        print(f'Seal: {seal_res/1E9}')

        proto_names = ['NaInact0', 'NaInact20', 'NaInact40', 'NaInact60', 'NaInact80']
        
        for i, proto_name in enumerate(proto_names):
            proto_num = self.trials[proto_name]
            if proto_num is None:
                continue
            num_sweeps = bundle.pul[0][proto_num].NumberSweeps

            ch_label = f'Imon-{ch}'
            ch_index = [i for i, trace in enumerate(
                            bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

            na_inact = []
            for sweep in range(0, num_sweeps):
                dat = bundle.data[0, proto_num, sweep, ch_index]/1E-9
                na_inact.append(np.min(dat[10000:]))

            na_inact = na_inact / np.min(na_inact)

            all_inact.append(na_inact)
                
        #COME BACK HERE
        voltages = np.linspace(-120-self.ljp, -0-self.ljp, len(na_inact)+1)[:-1]

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        for i, na_inact in enumerate(all_inact):
            c_ratio = .9-(i/(len(all_inact)))

            ax.plot(voltages, na_inact, label=i*20,
                            c=(c_ratio, c_ratio, c_ratio), marker='o')

        fig.suptitle(f'Cm={round(cm/1E-12, 2)}pF, Rs={round(1/g_series/1E6, 2)}')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axhline(y=0, color='grey', linestyle='-', lw=.5)
        ax.set_xlabel('Voltage (mV)', fontsize=18)
        ax.set_ylabel('Normalized Current', fontsize=18)
        plt.legend()

        if write_to is not None:
            plt.savefig(write_to)
        else:
            plt.show()


    def write_csv(self, write_path):
        ch = self.channel
        bundle = heka_reader.Bundle(self.cell_file)

        num_comps = 0

        proto_num = self.trials['NaIV0']

        num_sweeps = bundle.pul[0][proto_num].NumberSweeps

        ch_label = f'Imon-{ch}'
        ch_index = [i for i, trace in enumerate(bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

        # FIX THIS
        column_names = ['Sweep', 'SweepTime', 'Time']
        column_names += [f'Compound({i})' for i in range(1, 5)]
        column_names += [f'Age({i})' for i in range(1, 5)]
        column_names += [f'Concentration({i})' for i in range(1, 5)]
        column_names += [f'Ignore({i})' for i in range(1, 5)]
        column_names += [f'Rseries({i})' for i in range(1, 5)]
        column_names += [f'Cm({i})' for i in range(1, 5)]
        column_names += [f'Rseal({i})' for i in range(1, 5)]
        column_names += [f'Ignore({i})' for i in range(1, 5)]
        column_names += [f'Ignore({i})' for i in range(1, 5)]

        metadata_path = self.meta_path

        meta_in = pd.read_csv(f'{metadata_path}_1NaInact.xls',
                                                    sep=r'\t', skiprows=0)
        meta_in.columns = column_names
        meta_iv = pd.read_csv(f'{metadata_path}_1NaIV.xls',
                                            sep=r'\t', skiprows=0)
        meta_iv.columns = column_names
        for val in [0, 20, 40, 60, 80]:
            print(f'Compensation: {val}')
            curr_iv = f'NaIV{val}'
            curr_in = f'NaInact{val}'

            proto_iv = self.trials[curr_iv]
            proto_in = self.trials[curr_in]

            if (proto_iv is None):
                continue
            
            #IV first
            num_sweeps = bundle.pul[0][proto_iv].NumberSweeps

            ch_label = f'Imon-{ch}'
            ch_index = [i for i, trace in enumerate(
                            bundle.pul[0][proto_iv][0].children) if trace.Label == ch_label][0]

            iv_dat = {} 
            new_meta = {'voltage_mV': [],
                       'capacitance_pF': [],
                       'rseries_Mohm': [],
                       'rseal_Mohm': []} 
            voltages = np.linspace(-80, 90, 18)[:-1]

            for sweep in range(0, num_sweeps):
                dat = bundle.data[0, proto_iv, sweep, ch_index]/1E-9
                iv_dat[voltages[sweep]] = dat

                sweep_key = f'1_{proto_iv+1}_{sweep+1}'

                sweep_meta = meta_iv.loc[meta_iv['Sweep'] == sweep_key]
                voltage = voltages[sweep] 
                cm = sweep_meta[f'Cm({self.channel})'].values[0] *1E12
                rseries = sweep_meta[f'Rseries({self.channel})'].values[0] /1E6
                rseal = sweep_meta[f'Rseal({self.channel})'].values[0] /1E6

                new_meta['voltage_mV'].append(voltage)
                new_meta['capacitance_pF'].append(cm)
                new_meta['rseries_Mohm'].append(rseries)
                new_meta['rseal_Mohm'].append(rseal)

            final_iv = pd.DataFrame(iv_dat)
            final_iv_meta = pd.DataFrame(new_meta)

            final_iv.to_csv(f'{write_path}/NaIV_35C_{val}CP.csv', index=False)
            final_iv_meta.to_csv(f'{write_path}/NaIV_35C_{val}CP_meta.csv', index=False)

            #Inact
            if proto_in is None:
                continue

            num_sweeps = bundle.pul[0][proto_in].NumberSweeps

            ch_label = f'Imon-{ch}'
            ch_index = [i for i, trace in enumerate(
                            bundle.pul[0][proto_in][0].children) if trace.Label == ch_label][0]

            in_dat = {} 
            new_meta = {'voltage_mV': [],
                       'capacitance_pF': [],
                       'rseries_Mohm': [],
                       'rseal_Mohm': []} 

            voltages = np.linspace(-120, 0, 13)[:-1]

            for sweep in range(0, num_sweeps):
                dat = bundle.data[0, proto_in, sweep, ch_index]/1E-9

                in_dat[voltages[sweep]] = dat

                sweep_key = f'1_{proto_in+1}_{sweep+1}'

                sweep_meta = meta_in.loc[meta_in['Sweep'] == sweep_key]
                voltage = voltages[sweep] 
                cm = sweep_meta[f'Cm({self.channel})'].values[0] *1E12
                rseries = sweep_meta[f'Rseries({self.channel})'].values[0] /1E6
                rseal = sweep_meta[f'Rseal({self.channel})'].values[0] /1E6

                new_meta['voltage_mV'].append(voltage)
                new_meta['capacitance_pF'].append(cm)
                new_meta['rseries_Mohm'].append(rseries)
                new_meta['rseal_Mohm'].append(rseal)

            final_in = pd.DataFrame(in_dat)
            final_in_meta = pd.DataFrame(new_meta)


            final_in.to_csv(f'{write_path}/NaInact_35C_{val}CP.csv', index=False)
            final_in_meta.to_csv(f'{write_path}/NaInact_35C_{val}CP_meta.csv', index=False)







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


if __name__ == "__main__":
    main()
