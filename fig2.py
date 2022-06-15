import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc 
import pandas as pd
from multiprocessing import Pool
import seaborn as sns

import myokit
from _utility_functions import *



def lhs_array(min_val, max_val, n, log=False):
    sampler = qmc.LatinHypercube(1)
    sample = sampler.random(n)

    if log:
        dat = 10**qmc.scale(sample, np.log10(min_val), np.log10(max_val))
    else:
        dat = qmc.scale(sample, min_val, max_val)

    return dat.flatten()


def mod_sim(param_vals):
    proto = myokit.Protocol()

    for v in range(-90, 50, 2):
        proto.add_step(-100, 2000)
        proto.add_step(v, 20)

    gna = param_vals[0]
    rs = param_vals[1]
    cm = param_vals[2]
    comp = .8

    mod = myokit.load_model('./mmt-models/ord_na_lei.mmt')
    mod['INa']['g_Na_scale'].set_rhs(gna)

    mod['voltageclamp']['rseries'].set_rhs(rs)
    mod['voltageclamp']['rseries_est'].set_rhs(rs)

    cm_val = 15
    mod['voltageclamp']['cm_est'].set_rhs(cm)
    mod['model_parameters']['Cm'].set_rhs(cm)

    mod['voltageclamp']['alpha_c'].set_rhs(comp)
    mod['voltageclamp']['alpha_p'].set_rhs(comp)

    dat, times = simulate_model(mod, proto)

    iv_dat = get_iv_data(mod, dat, times)

    print('Done')

    return [iv_dat, param_vals]


def generate_dat(num_mods=5):
    proto = myokit.Protocol()

    for v in range(-90, 50, 2):
        proto.add_step(-100, 2000)
        proto.add_step(v, 20)

    gna_vals = lhs_array(.2, 5, n=num_mods, log=True)
    rs_vals = lhs_array(4E-3, 15E-3, n=num_mods)
    cm_vals = lhs_array(8, 22, n=num_mods)
    comp = .8

    with Pool() as p:
        dat = p.map(mod_sim, np.array([gna_vals, rs_vals, cm_vals]).transpose())

    all_currents = []
    all_meta = []

    for curr_mod in dat:
        all_currents.append(curr_mod[0]['Current'])
        all_meta.append(curr_mod[1])

    all_sim_dat = pd.DataFrame(all_currents, columns=dat[0][0]['Voltage'])
    mod_meta = pd.DataFrame(all_meta, columns=['G_Na', 'Rs', 'Cm'])

    all_sim_dat.to_csv('./data/simulations/vary_params/all_sim.csv', index=False)
    mod_meta.to_csv('./data/simulations/vary_params/meta.csv', index=False)
    

def plot_all_models():
    iv_dat = pd.read_csv('./data/simulations/vary_params/all_sim.csv')
    iv_dat.T.plot()
    plt.show()


def plot_param_relationships():
    iv_dat = pd.read_csv('./data/simulations/vary_params/all_sim.csv')
    meta_dat = pd.read_csv('./data/simulations/vary_params/meta.csv')

    fig, axs = plt.subplots(1, 3, figsize=(12, 8))
    #num_currs = len(iv_dat.columns())

    peak_voltages = []

    for i, row in iv_dat.iterrows():
        peak_voltages.append(int(row.idxmin()[:-2]))

    min_voltage = min(peak_voltages)
    max_voltage = max(peak_voltages)

    for i, row in iv_dat.iterrows():
        gna = meta_dat.iloc[i]['G_Na']
        rs = meta_dat.iloc[i]['Rs']
        cm = meta_dat.iloc[i]['Cm']

        peak_voltage = peak_voltages[i]
        col = (peak_voltage - min_voltage) / (max_voltage - min_voltage)
        axs[0].scatter(gna, rs, c=(col, col, col))
        axs[1].scatter(gna, cm, c=(col, col, col))
        axs[2].scatter(cm, rs, c=(col, col, col))


    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[0].set_xlabel('GNa')
    axs[1].set_xlabel('GNa')
    axs[2].set_xlabel('Cm')

    axs[0].set_ylabel('Rs')
    axs[1].set_ylabel('Cm')
    axs[2].set_ylabel('Rs')

    import pdb
    pdb.set_trace()
        
    
def plot_peak_voltage_correlation():
    iv_dat = pd.read_csv('./data/simulations/vary_params/all_sim.csv')
    meta_dat = pd.read_csv('./data/simulations/vary_params/meta.csv')

    peak_voltages = []
    peak_currs = []

    for i, row in iv_dat.iterrows():
        peak_voltages.append(int(row.idxmin()[:-2]))
        peak_currs.append(row.min())

    meta_dat['peak_v'] = peak_voltages
    meta_dat['peak_i_nApF'] = peak_currs
    meta_dat['peak_i_nA'] =  meta_dat['peak_i_nApF'] * meta_dat['Cm']
    meta_dat['G_Na'] = np.log10(meta_dat['G_Na'])

    meta_dat.rename(columns={'G_Na': 'Log10(G_Na)'}, inplace=True)

    sns.pairplot(meta_dat, kind='reg', corner=True, height=2)
    plt.show()


def plot_all_iv():
    iv_dat = pd.read_csv('./data/simulations/vary_params/all_sim.csv')

    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    all_voltages = iv_dat.columns
    all_voltages = [int(v[:-2]) for v in all_voltages]

    peak_voltages = []

    all_currents = []
    all_normed_currents = []

    for i, row in iv_dat.iterrows():
        currents = row.values
        all_currents.append(currents)
        axs[0].plot(all_voltages, currents, c=(.5, .5, .5), alpha=.5)
        normed_row = currents/currents.min()
        axs[1].plot(all_voltages, -normed_row, c=(.5, .5, .5), alpha=.5)

    avg_all_currs = np.array(all_currents).mean(0)
    axs[0].plot(all_voltages, avg_all_currs, marker='.')
    axs[1].plot(all_voltages, -avg_all_currs/avg_all_currs.min(), marker='.')

    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axs[0].set_ylabel('A/F', fontsize=12)
    axs[1].set_ylabel('Normed Current', fontsize=12)
    axs[1].set_xlabel('Voltage (mV)', fontsize=12)
    #plt.xticks(np.linspace(0, 80, 10))
    plt.show()


def plot_correlations_iv():
    iv_dat = pd.read_csv('./data/simulations/vary_params/all_sim.csv')
    meta_dat = pd.read_csv('./data/simulations/vary_params/meta.csv')

    peak_voltages = []

    for i, row in iv_dat.iterrows():
        peak_voltages.append(int(row.idxmin()[:-2]))

    meta_dat['peak_v'] = peak_voltages
    meta_dat['G_Na'] = np.log10(meta_dat['G_Na'])

    fig, axs = plt.subplots(2, 2, figsize=(7, 6))
    plt.subplots_adjust(.1, .1, .99, .99, wspace=.3)
    
    all_voltages = iv_dat.columns
    all_voltages = [int(v[:-2]) for v in all_voltages]

    peak_voltages = []

    all_currents = []
    all_normed_currents = []

    for i, row in iv_dat.iterrows():
        currents = row.values
        all_currents.append(currents)
        normed_row = currents/currents.min()
        axs[0][0].plot(all_voltages, -normed_row, c=(.5, .5, .5), alpha=.5)

    avg_all_currs = np.array(all_currents).mean(0)
    axs[0][0].plot(all_voltages, -avg_all_currs/avg_all_currs.min(), marker='.')

    #axs[0][1].scatter(meta_dat['G_Na'], meta_dat['peak_v'])
    #axs[1][0].scatter(meta_dat['Rs'], meta_dat['peak_v'])
    #axs[1][1].scatter(meta_dat['Cm'], meta_dat['peak_v'])
    sns.regplot(meta_dat['G_Na'], meta_dat['peak_v'], ax=axs[0][1])
    sns.regplot(meta_dat['Rs'], meta_dat['peak_v'], ax=axs[1][0])
    sns.regplot(meta_dat['Cm'], meta_dat['peak_v'], ax=axs[1][1])

    fs = 10

    axs[0][0].set_xlabel('Voltage (mV)', fontsize=fs)
    axs[0][0].set_ylabel('Normalize Current', fontsize=fs)

    axs[0][1].set_xlabel('G_Na', fontsize=fs)
    axs[0][1].set_ylabel('Peak Voltage (mV)', fontsize=fs)

    axs[1][0].set_xlabel('Rs (Mohm)', fontsize=fs)
    axs[1][0].set_ylabel('Peak Voltage (mV)', fontsize=fs)

    axs[1][1].set_xlabel('Cm (pF)', fontsize=fs)
    axs[1][1].set_ylabel('Peak Voltage (mV)', fontsize=fs)

    axs = axs.flatten()
    for ax in axs:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.savefig('./figures/fig2.svg')
    plt.show()


def main():
    #generate_dat(150)
    #plot_all_models()
    #plot_param_relationships()
    plot_peak_voltage_correlation()
    #plot_all_iv()
    #plot_correlations_iv()


if __name__ == '__main__':
    main()
