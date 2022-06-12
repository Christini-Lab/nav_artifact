import myokit
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

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


def get_baseline_iv(ext_conc=145):
    model_name = 'Ord'
    sample_freq = .0001
    mod = myokit.load_model('./mmt-models/ord_na.mmt')
    mem_name = 'Membrane.V'
    ion_name = 'Membrane.i_ion'
    scale = 1000
    g_na = mod['INa']['GNa'].value()
    mod['INa']['GNa'].set_rhs(g_na*.301)
    #mod['INa']['GNa'].set_rhs(g_na*10)
    mod['extracellular']['nao'].set_rhs(ext_conc)

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
    sim = myokit.Simulation(mod, proto)
    t = proto.characteristic_time()
    times = np.arange(0, t, sample_freq*scale)

    dat = sim.run(t, log_times=times)

    voltages = np.array(dat[mem_name])
    current = np.array(dat[ion_name]) # in nA

    iv_dat = get_iv_dat(voltages, current, dat)

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


def get_iv_dat(voltages, current, dat):
    iv_dat = {}

    step_idxs = np.where(np.diff(voltages) > .005)[0]

    v_steps = voltages[step_idxs + 10]
    iv_dat['Voltage'] = v_steps

    currs = []
    v_ena = []
    gating = []
    for idx in step_idxs:
        #start from +3
        temp_currs = current[(idx+3):(idx+103)]
        temp_gating = np.array(dat['INa.gating'])[(idx+3):(idx+103)]
        x = find_peaks(-np.array(temp_currs), distance=5, width=5)

        if len(x[0]) < 1:
            currs.append(np.min(temp_currs))
            gating.append(np.max(temp_gating))
        else:
            currs.append(temp_currs[x[0][0]])
            x = find_peaks(np.array(temp_gating), distance=5, width=2)
            gating.append(temp_gating[x[0][0]])

        v_ena.append(np.array(dat['INa.v_ENa'])[(idx+3)])

    iv_dat['Current'] = currs
    iv_dat['V-ENa'] = v_ena
    iv_dat['gating'] = gating

    return iv_dat

