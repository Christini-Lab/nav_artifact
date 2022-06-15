import matplotlib.pyplot as plt
import numpy as np
#from scipy.stats import qmc 
import pandas as pd
#from multiprocessing import Pool
import seaborn as sns
from os import listdir
import matplotlib.backends.backend_pdf

#import myokit
#from _utility_functions import *



def plot_all_exp_iv():
    fig, axs = plt.subplots(2, 5, figsize=(12, 8), sharex=True)
    fig.subplots_adjust(.1, .1, .9, .9)

    voltages = np.linspace(-90, 50, 15)

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
        axs[0][i].set_title(f'Compensation {i*20}%', fontsize=12)
        axs[1][i].plot(voltages, -avg_currs/avg_currs.min(), marker='.')

    for i, row in enumerate(axs):
        for ax in row:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i > 0:
                ax.set_xlabel('Voltage (mV)')
            else:
                ax.set_ylim(-50, 0)

    axs[0][0].set_ylabel('Current (nA)')
    axs[1][0].set_ylabel('Normed Current')
                
    plt.show()

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 8))

    comp = [0, 20, 40, 60, 80]
    cols = [.8, .6, .4, .2, 0]
    
    for i, currs in enumerate(all_currs): 
        avg_currs = np.array(currs).mean(0)
        col = (cols[i], cols[i], cols[i])
        axs[0].plot(voltages, avg_currs, marker='.', label=f'{comp[i]}%', c=col)
        #axs[0].errorbar(voltages, np.array(currs).mean(0), np.array(currs).std(0), label=f'{comp[i]}%')
        axs[1].plot(voltages, -avg_currs/avg_currs.min(), marker='.', c=col)
    

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[0].set_ylabel('Current (nA)')
    axs[1].set_ylabel('Normalized Current')
    axs[1].set_xlabel('Voltage (mV)')

    axs[0].legend()

    plt.show()


def plot_exp_iv_af():
    fig, axs = plt.subplots(1, 5, figsize=(12, 5))
    fig.subplots_adjust(.1, .1, .99, .9)
    voltages = np.linspace(-90, 50, 15)

    all_meta = [[], [], [], [], []]
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
                curr_meta_name = f'{trial.split(".")[0]}_meta.csv'
                curr_meta = pd.read_csv(f'{curr_path}/{curr_meta_name}')

                all_meta[i].append([curr_meta.mean()['capacitance_pF'],
                                    curr_meta.mean()['rseries_Mohm'],
                                    curr_trial.min().min(),
                                    int(curr_trial.min().argmin()[:-2])])

                curr_trial = pd.read_csv(f'{curr_path}/{trial}')
                curr_vals = curr_trial.min().values
                normed_curr = curr_vals/curr_meta.mean()['capacitance_pF']
                axs[i].plot(voltages, normed_curr, 'grey', alpha=.5)
                all_currs[i].append(normed_curr)

    #ax.set_ylim(-50, 0)

    axs[0].set_ylabel('Current (nA/pF)')
    #axs[1][0].set_ylabel('Normed Current')
                

    labs = [0, 20, 40, 60 , 80]
    for i, currs in enumerate(all_currs): 
        avg_currs = np.array(currs).mean(0)
        axs[i].plot(voltages, avg_currs, marker='.')
        axs[i].set_title(f'{labs[i]}% Comp')


    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Voltage (mV)')

    plt.show()

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 4))

    comp = [0, 20, 40, 60, 80]
    cols = [.8, .6, .4, .2, 0]
    
    for i, currs in enumerate(all_currs): 
        avg_currs = np.array(currs).mean(0)
        col = (cols[i], cols[i], cols[i])
        ax.plot(voltages, avg_currs, marker='.', label=f'{comp[i]}%', c=col)
        #ax.errorbar(voltages, np.array(currs).mean(0), np.array(currs).std(0), label=f'{comp[i]}%')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel('Current (nA/pF)')
    ax.set_xlabel('Voltage (mV)')

    ax.legend()

    plt.show()


def plot_peak_v_relationships():
    voltages = np.linspace(-90, 50, 15)

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
                                    (curr_trial.min().min()/
                                        curr_meta.mean()['capacitance_pF']),
                                    int(curr_trial.min().argmin()[:-2])])

    pdf = matplotlib.backends.backend_pdf.PdfPages("figures/exp_pairplots.pdf")

    for i, comp_set in enumerate(all_meta):
        df = pd.DataFrame(comp_set, columns=['Cm', 'Rs', 'peak_I_nA', 'peak_I_nApF', 'peak_V'])
        grid = sns.pairplot(df, kind='reg', corner=True).set()
        grid.fig.suptitle(f'{20*i}% Compensation', fontsize=16)
        grid.fig.subplots_adjust(top=.94)
        pdf.savefig(grid.fig)
        plt.show()

    pdf.close()





#plot_all_exp_iv()
#plot_exp_iv_af()
plot_peak_v_relationships()
