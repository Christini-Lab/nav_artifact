import myokit
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import pandas as pd

import heka_reader


# Utility functions
def simulate_model(mod, proto, with_hold=True, sample_freq=0.00004):
    if mod.time_unit().multiplier() == .001:
        scale = 1000
    else:
        scale = 1

    p = mod.get('engine.pace')
    p.set_binding(None)

    v_cmd = mod.get('voltageclamp.Vc')
    v_cmd.set_rhs(0)
    v_cmd.set_binding('pace') # Bind to the pacing mechanism

    # Run for 20 s before running the VC protocol
    if with_hold:
        holding_proto = myokit.Protocol()
        holding_proto.add_step(-.080*scale, 30*scale)
        sim = myokit.Simulation(mod, holding_proto)
        t_max = holding_proto.characteristic_time()
        sim.run(t_max)
        mod.set_state(sim.state())

    t_max = proto.characteristic_time()
    times = np.arange(0, t_max, sample_freq*scale)
    sim = myokit.Simulation(mod, proto)

    dat = sim.run(t_max, log_times=times)

    return dat, times


def get_current_at_steps(times, dat, cm):
    v_dat = np.array([v for v in dat['voltageclamp.Vc']])
    i_out = np.array([i_out / cm for i_out in dat['voltageclamp.Iout']])

    v_change_idx = find_peaks(np.diff(v_dat))[0]
    v_steps = [v_dat[idx+5] for idx in v_change_idx]

    current_at_steps = {}

    for i, v_step in enumerate(v_steps):
        start_t = times[v_change_idx[i]] - 5
        end_t = times[v_change_idx[i]] + 10 
        start_idx = np.abs(times - start_t).argmin()
        end_idx = np.abs(times - end_t).argmin()

        current_at_steps[v_step] = i_out[start_idx:end_idx]

    return current_at_steps


def get_iv_data(mod, dat, times):
    iv_dat = {}

    cm = mod['voltageclamp']['cm_est'].value()
    i_out = [v/cm for v in dat['voltageclamp.Iout']]
    v = np.array(dat['voltageclamp.Vc'])
    step_idxs = np.where(np.diff(v) > .005)[0]

    v_steps = v[step_idxs + 10]
    iv_dat['Voltage'] = v_steps

    sample_period = times[1]
    if mod.time_unit().multiplier() == .001:
        scale = 1000
    else:
        scale = 1

    currs = []
    #for idx in step_idxs:
    #    currs.append(np.min(i_out[(idx+3):(idx+23)]))

    for idx in step_idxs:
        temp_currs = i_out[(idx+3):(idx+103)]
        x = find_peaks(-np.array(temp_currs), distance=5, width=4)

        if len(x[0]) < 1:
            currs.append(np.min(temp_currs))
        else:
            currs.append(temp_currs[x[0][0]])


    iv_dat['Current'] = currs

    return iv_dat


def get_baseline_iv():
    model_name = 'Ord'
    sample_freq = .0001
    mod = myokit.load_model('./mmt-models/ord_na.mmt')
    mem_name = 'Membrane.V'
    ion_name = 'Membrane.i_ion'
    scale = 1000
    g_na = mod['INa']['GNa'].value()
    #mod['INa']['GNa'].set_rhs(g_na*.301)

    p = mod.get('engine.pace')
    p.set_binding(None)

    v = mod.get(mem_name)
    v.demote()
    v.set_rhs(0)
    v.set_binding('pace') # Bind to the pacing mechanism

    holding_proto = myokit.Protocol()
    holding_proto.add_step(-.080*scale, 30*scale)

    t = holding_proto.characteristic_time()
    sim = myokit.Simulation(mod, holding_proto)
    dat = sim.run(t)

    mod.set_state(sim.state())

    proto = get_mont_sodium_proto(scale)
    proto = myokit.Protocol()

    for v in range(-90, 50, 5):
        proto.add_step(-100, 2000)
        proto.add_step(v, 20)

    proto.add_step(-100, 2000)

    sim = myokit.Simulation(mod, proto)
    t = proto.characteristic_time()
    times = np.arange(0, t, sample_freq*scale)

    dat = sim.run(t, log_times=times)

    voltages = np.array(dat[mem_name])
    current = np.array(dat[ion_name]) # in nA

    iv_dat = get_iv_dat(voltages, current)

    return times, dat, iv_dat


def get_mont_sodium_proto(scale=1000):
    proto = myokit.Protocol()
    proto.add_step(-.1*scale, 2*scale)

    for i in range(-80, 66, 5):
        if i == 0:
            proto.add_step(.1/1000*scale, .050*scale)
        else:
            proto.add_step(i/1000*scale, .050*scale)

        proto.add_step(-.1*scale, 1.95*scale)

    return proto


def get_iv_dat(voltages, current):
    iv_dat = {}

    step_idxs = np.where(np.diff(voltages) > .005)[0]

    v_steps = voltages[step_idxs + 10]
    iv_dat['Voltage'] = v_steps

    currs = []
    for idx in step_idxs:
        #start from +3
        temp_currs = current[(idx+3):(idx+103)]
        x = find_peaks(-np.array(temp_currs), distance=5, width=5)
        #if len(x[0]) > 1:

        if len(x[0]) < 1:
            currs.append(np.min(temp_currs))
        else:
            currs.append(temp_currs[x[0][0]])

    iv_dat['Current'] = currs

    return iv_dat




#####
def plot_pop_models():
    times, dat, ideal_iv = get_baseline_iv()

    proto = myokit.Protocol()

    for v in range(-90, 50, 5):
        proto.add_step(-100, 2000)
        proto.add_step(v, 20)

    proto.add_step(-100, 2000)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    all_iv_dat = []

    for i in range(0, 3):
        mod = myokit.load_model('./mmt-models/ord_na_lei.mmt')
        new_cm = np.random.uniform(5, 20) 
        new_cm_star = np.random.normal(new_cm, new_cm * .05)
        new_rs = np.random.uniform(2E-3, 12E-3) 
        new_rs_star = np.random.normal(new_rs, new_rs * .05) 

        mod['voltageclamp']['rseries'].set_rhs(new_rs)
        mod['voltageclamp']['rseries_est'].set_rhs(new_rs_star)

        mod['voltageclamp']['cm_est'].set_rhs(new_cm_star)
        mod['model_parameters']['Cm'].set_rhs(new_cm)

        #mod['INa']['g_Na_scale'].set_rhs(new_gna)
        
        dat, times = simulate_model(mod, proto)
        i_out = [i_out / new_cm_star for i_out in dat['voltageclamp.Iout']]

        axs[1].plot(times, i_out, 'k')

        all_iv_dat.append(get_iv_data(mod, dat, times))

    axs[0].plot(times, dat['voltageclamp.Vc'])

    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    axs[0].set_ylabel('Voltage (mV)')
    axs[1].set_ylabel('Current (A/F)')
    axs[1].set_xlabel('Time (ms)')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    all_currents = []
    for iv_dat in all_iv_dat:
        ax.plot(iv_dat['Voltage'], iv_dat['Current'], c='k', marker='o')
        voltage = iv_dat['Voltage']
        all_currents.append(iv_dat['Current'])


    ax.set_xlabel('Voltage')
    ax.set_ylabel('Current')
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.plot(voltage, np.mean(all_currents, 0), 'k', marker='o')
    curr_std = np.std(all_currents, 0)
    ax.errorbar(voltage, np.mean(all_currents, 0), yerr=curr_std, c='k')

    ax.set_xlabel('Voltage')
    ax.set_ylabel('Current')
    plt.show()


    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.plot(voltage, np.mean(all_currents, 0), 'k', marker='o')
    curr_std = np.std(all_currents, 0)
    ax.errorbar(voltage, np.mean(all_currents, 0), yerr=curr_std, c='k')

    ax.plot(ideal_iv['Voltage'], np.array(ideal_iv['Current'])*2.5, label='Ideal', marker='o')

    ax.set_xlabel('Voltage')
    ax.set_ylabel('Current')
    plt.show()


def create_pop_cells():
    times, dat, ideal_iv = get_baseline_iv()
    #np.savetxt('data/mod_simulations-fig2/baseline.csv', ideal_iv['Current'])
    #np.savetxt('data/mod_simulations-fig2/baseline_curr.csv', dat['INa.INa'])
    import pdb
    pdb.set_trace()

    np.savetxt('data/mod_simulations-fig2/baseline.csv', ideal_iv['Current'])

    proto = myokit.Protocol()

    for v in range(-90, 50, 5):
        proto.add_step(-100, 2000)
        proto.add_step(v, 20)

    proto.add_step(-100, 2000)

    all_curr_responses = {}
    all_iv_dat = {} 
    all_params = {}

    for i in range(5, 20):
        for j in np.linspace(2E-3, 12E-3, 5):
            mod = myokit.load_model('./mmt-models/ord_na_lei.mmt')
            new_cm = i
            new_cm_star = np.random.normal(new_cm, new_cm * .05)
            new_rs = j 
            new_rs_star = np.random.normal(new_rs, new_rs * .05) 
            #new_cm = np.random.uniform(5, 20) 
            #new_cm_star = np.random.normal(new_cm, new_cm * .05)
            #new_rs = np.random.uniform(2E-3, 12E-3) 
            #new_rs_star = np.random.normal(new_rs, new_rs * .05) 

            mod['voltageclamp']['rseries'].set_rhs(new_rs)
            mod['voltageclamp']['rseries_est'].set_rhs(new_rs_star)

            mod['voltageclamp']['cm_est'].set_rhs(new_cm_star)
            mod['model_parameters']['Cm'].set_rhs(new_cm)

            dat, times = simulate_model(mod, proto)
            i_out = [i_out / new_cm_star for i_out in dat['voltageclamp.Iout']]

            iv_dat = get_iv_data(mod, dat, times)

            all_curr_responses[f'mod_{i}_{j}'] = i_out
            all_iv_dat[f'mod_{i}_{j}'] = iv_dat['Current']
            all_params[f'mod_{i}_{j}'] = [new_cm, new_rs]

            print(f'Rs: {i}, Cm: {j}')

        #all_iv_dat.append()
    
    all_currents = pd.DataFrame(all_curr_responses)
    all_ivs = pd.DataFrame(all_iv_dat)
    all_params = pd.DataFrame(all_params)

    all_currents.to_csv('data/mod_simulations-fig2/all_currs.csv', index=False)
    all_ivs.to_csv('data/mod_simulations-fig2/all_iv.csv', index=False)
    all_params.to_csv('data/mod_simulations-fig2/all_params.csv', index=False)


def main():
    np.linspace(0, 1464000/25000, 1464000)


    #create_pop_cells()
    plot_pop_models()


if __name__ == '__main__':
    main()
