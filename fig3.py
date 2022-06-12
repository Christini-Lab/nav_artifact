import matplotlib.pyplot as plt
import numpy as np
#from scipy.stats import qmc 
import pandas as pd
#from multiprocessing import Pool
import seaborn as sns
from os import listdir

#import myokit
#from _utility_functions import *



def plot_all_data():

    fig, axs = plt.subplots(2, 5, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(.1, .1, .99, .99)

    voltages = np.linspace(-80, 60, 15)

    all_currs = [[], [], [], [], []]

    for folder in ['high', 'medium']:
        all_exps = listdir(f'./data/csv/{folder}_res_data/')
        for exp in all_exps:
            if 'ch' not in exp:
                continue

            curr_path = f'./data/csv/{folder}_res_data/{exp}'
        
            all_trials = listdir(curr_path)
            all_trials = [name for name in all_trials if 'NaIV' in name]
            all_trials = [name for name in all_trials if 'meta' not in name]
            all_trials.sort()

            for i, trial in enumerate(all_trials):
                curr_trial = pd.read_csv(f'{curr_path}/{trial}')
                curr_vals = curr_trial.min().values
                axs[0][i].plot(voltages, curr_vals, 'grey', alpha=.5)
                normed_curr = curr_vals / np.min(curr_vals)
                axs[1][i].plot(voltages, -normed_curr, 'grey', alpha=.5)
                all_currs[i].append(curr_vals)

    for i, currs in enumerate(all_currs): 
        avg_currs = np.array(currs).mean(0)
        axs[0][i].plot(voltages, avg_currs, marker='.')
        axs[1][i].plot(voltages, -avg_currs/avg_currs.min(), marker='.')

    for i, row in enumerate(axs):
        for ax in row:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i > 0:
                ax.set_xlabel('Voltage (mV)')
            else:
                ax.set_ylim(-50, 0)

    axs[0][0].set_ylabel('Current')
    axs[1][0].set_ylabel('Normed Current')
                
    plt.show()

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    comp = [0, 20, 40, 60, 80]
    for i, currs in enumerate(all_currs): 
        avg_currs = np.array(currs).mean(0)
        axs[0].plot(voltages, avg_currs, marker='.', label=f'{comp[i]}%')
        #axs[0].errorbar(voltages, np.array(currs).mean(0), np.array(currs).std(0), label=f'{comp[i]}%')
        axs[1].plot(voltages, -avg_currs/avg_currs.min(), marker='.')
    

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[0].set_ylabel('Current (nA)')
    axs[1].set_ylabel('Normalized Current')
    axs[1].set_xlabel('Voltage (mV)')

    axs[0].legend()

    plt.show()





def plot_peak_v_relationships():

    #fig, axs = plt.subplots(2, 5, figsize=(12, 8), sharex=True)
    #fig.subplots_adjust(.1, .1, .99, .99)

    voltages = np.linspace(-80, 60, 15)

    all_meta = [[], [], [], [], []]

    for folder in ['high', 'medium']:
        all_exps = listdir(f'./data/csv/{folder}_res_data/')
        for exp in all_exps:
            if 'ch' not in exp:
                continue

            curr_path = f'./data/csv/{folder}_res_data/{exp}'
        
            all_trials = listdir(curr_path)
            all_trials = [name for name in all_trials if 'NaIV' in name]
            all_trials = [name for name in all_trials if 'meta' not in name]
            all_trials.sort()

            for i, trial in enumerate(all_trials):
                curr_trial = pd.read_csv(f'{curr_path}/{trial}')
                curr_meta_name = f'{trial.split(".")[0]}_meta.csv'
                curr_meta = pd.read_csv(f'{curr_path}/{curr_meta_name}')

                all_meta[i].append([curr_meta.mean()['capacitance_pF'],
                                    curr_meta.mean()['rseries_Mohm'],
                                    curr_trial.min().min(),
                                    int(curr_trial.min().argmin()[:-2])])

    for comp_set in all_meta:
        df = pd.DataFrame(comp_set, columns=['Cm', 'Rs', 'peak_I', 'peak_V'])
        sns.pairplot(df, kind='reg')
        plt.show()

    for i, currs in enumerate(all_currs): 
        avg_currs = np.array(currs).mean(0)
        axs[0][i].plot(voltages, avg_currs, marker='.')
        axs[1][i].plot(voltages, -avg_currs/avg_currs.min(), marker='.')

    for i, row in enumerate(axs):
        for ax in row:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i > 0:
                ax.set_xlabel('Voltage (mV)')
            else:
                ax.set_ylim(-50, 0)

    axs[0][0].set_ylabel('Current')
    axs[1][0].set_ylabel('Normed Current')
                

    plt.show()





plot_all_data()
plot_peak_v_relationships()
