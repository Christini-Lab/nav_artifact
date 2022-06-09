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


class ExpInformation():
    def __init__(self, f_path, channel, trials):
        self.f_path = f_path
        self.channel = channel
        self.trials = trials


    def get_exp_data(self, comp_setting, comp_type='NaIVCP'):
        ch = self.channel
        bundle = heka_reader.Bundle(self.f_path)

        num_comps = 0


        if comp_setting == 0:
            proto_num = self.trials['NaIV0_1']
        else:
            proto_range = self.trials[f'{comp_type}_1']
            proto_num = int(comp_setting / 10 - 1 + proto_range[0])

        num_sweeps = bundle.pul[0][proto_num].NumberSweeps

        ch_label = f'Imon-{ch}'
        ch_index = [i for i, trace in enumerate( bundle.pul[0][proto_num][0].children) if trace.Label == ch_label][0]

        cm = bundle.pul[0][proto_num][0][ch_index].CSlow
        g_series = bundle.pul[0][proto_num][0][ch_index].GSeries


        print(f'Cm: {cm*1E12}')
        print(f'Rseries: {1/1E6/g_series}')
        
        all_traces = []

        for sweep in range(0, num_sweeps):
            dat = bundle.data[0, proto_num, sweep, ch_index]/1E-9
            all_traces.append(dat)

        t = np.linspace(0, len(dat)/25000, len(dat))

        all_traces = dict(zip(list(range(-80, 50, 10)), all_traces))

        return t, all_traces


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


#####
def plot_voltage_current():
    cm = 60

    mod = myokit.load_model('./mmt-models/ord_na_lei.mmt')
    mod['voltageclamp']['rseries'].set_rhs(5E-3)
    mod['voltageclamp']['rseries_est'].set_rhs(5E-3)

    mod['voltageclamp']['cm_est'].set_rhs(cm)
    mod['model_parameters']['Cm'].set_rhs(cm)

    proto = myokit.Protocol()

    for v in range(-90, 50, 10):
        proto.add_step(-100, 2000)
        proto.add_step(v, 20)

    proto.add_step(-100, 2000)

    dat, times = simulate_model(mod, proto)
    i_out = [i_out / cm for i_out in dat['voltageclamp.Iout']]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    axs[0].plot(times, dat['voltageclamp.Vc'])
    axs[1].plot(times, i_out)

    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    axs[0].set_ylabel('Voltage (mV)')
    axs[1].set_ylabel('Current (A/F)')
    axs[1].set_xlabel('Time (ms)')
    plt.show()


def plot_multiple_compensations(v_step=-10):
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    #f_path = '../nav-experiments/data/nav_2-10-2022/220210_003.dat'

    f_path = '../nav-study/data/nav_2-3-2022/220203_007.dat'
    cell = ExpInformation(f_path, 2,
                                     {'NaIV0_1': 5,
                                      'NaIVCP_1': [6, 13],
                                      'NaIVC_1': [14, 21],
                                      'NaIVP_1': [22, 29]})


    for comp_setting in range(0, 90, 10):
        t, traces = cell.get_exp_data(comp_setting)
        axs[2].plot(t*1000-10, traces[v_step])

    proto = myokit.Protocol()

    for v in range(-90, 50, 10):
        proto.add_step(-100, 2000)
        proto.add_step(v, 20)

    proto.add_step(-100, 2000)

    cm = 11 
    r_series = 6E-3

    axs[1].set_xlim(-2, 6)

    for comp_setting in range(0, 90, 10):
        mod = myokit.load_model('./mmt-models/ord_na_lei.mmt')
        mod['voltageclamp']['rseries'].set_rhs(r_series)
        mod['voltageclamp']['rseries_est'].set_rhs(r_series)

        mod['voltageclamp']['cm_est'].set_rhs(cm)
        mod['model_parameters']['Cm'].set_rhs(cm)

        mod['voltageclamp']['alpha_c'].set_rhs(float(comp_setting)/100)
        mod['voltageclamp']['alpha_p'].set_rhs(float(comp_setting)/100)

        mod['INa']['g_Na_scale'].set_rhs(.25)

        dat, times = simulate_model(mod, proto)

        current_at_steps = get_current_at_steps(times, dat, cm)

        time_step = times[1] - times[0]
        times = np.arange(-5, 10, time_step)

        axs[1].plot(times, current_at_steps[v_step]*cm/1000, label=f'Comp={comp_setting}%')

    voltage_pts = np.array([[-5, -100], [0, -100], [0, v_step], [15, v_step]])
    axs[0].plot(voltage_pts[:, 0], voltage_pts[:, 1])

    for ax in axs:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    #axs[1].legend()
    axs[0].set_ylabel('Voltage (mV)')
    axs[1].set_ylabel('Current (nA)')
    axs[2].set_ylabel('Current (nA)')
    axs[1].set_xlabel('Time (ms)')

    plt.show()
    import pdb
    pdb.set_trace()


    i_out = [i_out / 60 for i_out in dat['voltageclamp.Iout']]

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    axs[0].plot(times, dat['voltageclamp.Vc'])
    axs[1].plot(times, i_out)

    plt.show()



#plot_voltage_current()
plot_multiple_compensations()
