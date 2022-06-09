import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


all_currs = pd.read_csv('./data/mod_simulations-fig2/all_currs.csv')
all_iv = pd.read_csv('./data/mod_simulations-fig2/all_iv.csv')
all_params = pd.read_csv('./data/mod_simulations-fig2/all_params.csv')
baseline_curr = pd.read_csv('./data/mod_simulations-fig2/baseline_curr.csv')
all_v = list(range(-90, 50, 5))

baseline_dat = np.loadtxt('./data/mod_simulations-fig2/baseline.csv')

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

for i in range(0, len(all_iv.values[0, :])):
    ax.plot(all_v, all_iv.values[:, i], 'grey', alpha=.1)

line1 = ax.plot(all_v, all_iv.mean(axis=1).values, 'grey', marker='o', markersize=10, label='Average')
line2 = ax.plot(all_v, baseline_dat, c=(.8, .1, .1), marker='o', markersize=10, label='Baseline')
ebars = ax.errorbar(all_v, all_iv.mean(axis=1).values, yerr=all_iv.std(axis=1).values, c='grey')

ax.set_xlabel('Voltage (mV)', fontsize=16)
ax.set_ylabel('Current(A/F)', fontsize=16)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend()

plt.show()
