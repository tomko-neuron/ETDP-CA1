import json
import pickle, gzip
import numpy as np
import matplotlib.patches as mpatches
import matplotlib as mpl
from matplotlib import pyplot as plt, cm
from neuron import h, gui

plt.style.use('seaborn-colorblind')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['lines.markersize'] = 1.5
EXPERIMENTS = ['2stim_3xTBS', '5stim_3xTBS', '5stim_3xTBS_IClamp', '5stim_3xTBS_VClamp']
COLORS = ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161', '#fbafe4', '#949494', '#ece133', '#56b4e9']


def main():
    # avg_LTP()
    fig_LTP()
    # fig_LTP_distribution()
    # fig_lm_voltage()
    # fig_lm_voltage_na_ca_current()
    # fig_lm_voltage_weights()
    # fig_shape_synapses()
    # fig_voltage()
    # fig_LTP_IClamp()


def load_json_data(file):
    try:
        with open(file, 'r') as f:
            data = json.load(f)
            return data
    except FileNotFoundError as fnf_error:
        raise fnf_error


def avg_LTP():
    AVG_LTP = {}
    for e in EXPERIMENTS:
        AVG_LTP[e] = []
        for i in range(1, 6, 1):
            synapses = load_json_data('../recordings/' + e + '/01_0' + str(i) + '/synapses.json')
            ltp = []
            for sec in synapses:
                if len(synapses[sec]) > 0:
                    for syn in synapses[sec]:
                        if syn['receptor'] == 'AMPA':
                            ltp.append(syn['final_weight'] / syn['init_weight'])
            AVG_LTP[e].append(np.average(ltp))
    try:
        with open('./avg_LTP.json', 'w') as f:
            json.dump(AVG_LTP, f, indent=4)
    except FileNotFoundError as fnf_error:
        raise fnf_error


def fig_LTP():
    avg = []
    std = []
    sem = []
    avg_paper = [1.27507, 1.78743, 1.45153, 1.35464]  # 1 min
    sem_paper = [0.0709625, 0.278596, 0.224964, 0.0816961]  # 1 min
    std_paper = [0.255858932, 0.622959594, 0.67488, 0.2450883]  # 1 min

    data = load_json_data('./avg_LTP.json')
    for e in EXPERIMENTS:
        avg.append(np.average(data[e]))
        std.append(np.std(data[e], axis=0))
        sem.append(std[-1] / np.sqrt(len(data[e])))

    barWidth = 0.3
    r1 = np.arange(start=0, stop=4, step=1)
    r2 = np.arange(start=0.3, stop=4.3, step=1)
    r3 = np.arange(start=0.15, stop=4.15, step=1)

    pparam = dict(ylabel='Potentiation ratio\n (Post TBS / Baseline)')
    fig = plt.figure(figsize=(2.8, 2.24), dpi=300)
    ax = plt.axes()
    ax.bar(r1, avg_paper, yerr=sem_paper, width=barWidth, label='exp. data', color='#949494',
           error_kw=dict(lw=0.5, capsize=2.0, capthick=0.5))
    ax.bar(r2, avg, yerr=sem, width=barWidth, label='simulation', color='#029e73',
           error_kw=dict(lw=0.5, capsize=2.0, capthick=0.5))
    ax.set_xticks(r3, EXPERIMENTS, rotation=45)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False, bottom=False)
    ax.legend(fontsize=4)
    # ax.autoscale(tight=True)
    ax.set(**pparam)
    fig.tight_layout()
    fig.savefig('./LTP_v2.svg', format='svg')
    fig.savefig('./LTP_v2.png', format='png')
    plt.show()
    plt.close(fig)


def fig_LTP_distribution():
    fig = plt.figure(figsize=(2.3, 7), dpi=300)
    for i in range(1, 5, 1):
        ax = fig.add_subplot(4, 1, i)
        synapses = load_json_data('../recordings/' + EXPERIMENTS[i - 1] + '/01_01/synapses.json')
        LTP = []
        dist = []
        for sec in synapses:
            if len(synapses[sec]) > 0:
                for syn in synapses[sec]:
                    if syn['receptor'] == 'AMPA':
                        LTP.append(syn['final_weight'] / syn['init_weight'])
                        dist.append(syn['distance'])
        ax.scatter(x=dist, y=LTP, label='Control', color='#029e73', marker='.')

        synapses = load_json_data('../recordings/' + EXPERIMENTS[i - 1] + '/01_01_TTX/synapses.json')
        LTP = []
        dist = []
        for sec in synapses:
            if len(synapses[sec]) > 0:
                for syn in synapses[sec]:
                    if syn['receptor'] == 'AMPA':
                        LTP.append(syn['final_weight'] / syn['init_weight'])
                        dist.append(syn['distance'])
        ax.scatter(x=dist, y=LTP, label='TTX', color='#de8f05', marker='.')
        ax.set_title(EXPERIMENTS[i - 1])
        ax.set_ylim(0.9, 1.7)
        ax.set_ylabel('Potentiation ratio\n (Post TBS / Baseline)')
        ax.set_xlabel('Distance from the soma ($\mu$m)')
        if i == 1:
            plt.legend(loc='upper left', fontsize=4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='both', top=False, right=False, bottom=False)
    fig.tight_layout()
    fig.savefig('./LTP_distribution_v2.svg', format='svg')
    fig.savefig('./LTP_distribution_v2.png', format='png')
    plt.show()
    plt.close(fig)


def fig_lm_voltage_na_ca_current():
    for e in EXPERIMENTS:
        data_voltage = pickle.load(gzip.GzipFile('../recordings/' + e + '/01_01/voltages.p', 'rb'))
        data_voltage_TTX = pickle.load(gzip.GzipFile('../recordings/' + e + '/01_01_TTX/voltages.p', 'rb'))
        data_current = pickle.load(gzip.GzipFile('../recordings/' + e + '/01_01/currents.p', 'rb'))
        data_current_TTX = pickle.load(gzip.GzipFile('../recordings/' + e + '/01_01_TTX/currents.p', 'rb'))
        fig = plt.figure(figsize=(2.031, 3.385))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(data_voltage['T'], data_voltage['V_dends']['dendA5_0111111111111111111'][1]['vector'],
                 label='Control', color='#029e73')
        ax1.plot(data_voltage_TTX['T'], data_voltage_TTX['V_dends']['dendA5_0111111111111111111'][1]['vector'],
                 label='TTX', color='#de8f05')
        ax1.set_xlim(180, 280)
        ax1.set_ylim(-75, -10)
        ax1.set_ylabel('Voltage (mV)')
        ax1.set_xlabel('Time (ms)')
        ax1.set_title(e)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(axis='both', which='both', top=False, right=False, bottom=False)
        plt.legend(fontsize=4)

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(data_current['T'], data_current['ina']['dendA5_0111111111111111111'][1]['vector'],
                 label='Control', color='#029e73')
        ax2.plot(data_current_TTX['T'], data_current_TTX['ina']['dendA5_0111111111111111111'][1]['vector'],
                 label='TTX', color='#de8f05')
        ax2.set_xlim(180, 280)
        ax2.set_ylim(-0.8, 0.05)
        ax2.set_ylabel(r'Na current (mA/cm$^2$)')
        ax2.set_xlabel('Time (ms)')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.tick_params(axis='both', which='both', top=False, right=False, bottom=False)

        fig.tight_layout()
        plt.savefig('./' + e + '_volt_curr_v2.svg', format='svg')
        plt.savefig('./' + e + '_volt_curr_v2.png', format='png')
        plt.close(fig)


def fig_lm_voltage_weights():
    fig = plt.figure(figsize=(2.031, 3.386), dpi=300)
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    colors = ['#d55e00', '#cc78bc', '#ca9161', '#fbafe4']

    ax1.eventplot(positions=[200, 210, 220, 230, 240], orientation='horizontal', color='#0173b2')
    ax1.set_xlim(190, 290)
    ax1.set_ylabel('Input spikes')
    ax1.set_xlabel('Time (ms)')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='both', which='both', top=False, right=False, bottom=False)

    for i in range(4):
        data_voltage = pickle.load(gzip.GzipFile('../recordings/' + EXPERIMENTS[i] + '/01_01/voltages.p', 'rb'))
        data_weights = pickle.load(gzip.GzipFile('../recordings/' + EXPERIMENTS[i] + '/01_01/synapses.p', 'rb'))
        ax2.plot(data_voltage['T'], data_voltage['V_dends']['dendA5_0111111111111111111'][1]['vector'],
                 label=EXPERIMENTS[i], color=colors[i])

        ax3.plot(data_weights['T'], data_weights['synapses']['dendA5_0111111111111111111'][12]['weight'] * 1000,
                 label=EXPERIMENTS[i], color=colors[i])

    ax2.set_xlim(190, 290)
    ax2.set_ylim(-75, -10)
    ax2.set_ylabel('Voltage (mV)')
    ax2.set_xlabel('Time (ms)')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='both', which='both', top=False, right=False, bottom=False)

    ax3.set_xlim(190, 290)
    # ax2.set_ylim(-0.8, 0.05)
    ax3.set_ylabel(r'Synaptic weight (nS)')
    ax3.set_xlabel('Time (ms)')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.tick_params(axis='both', which='both', top=False, right=False, bottom=False)
    ax3.legend(loc='upper right', fontsize=4)

    fig.tight_layout()
    plt.savefig('./volt_weight.svg', format='svg')
    plt.savefig('./volt_weight.png', format='png')
    plt.close(fig)


def fig_shape_synapses():
    h.load_file('../hoc/morphology_ri06.nrn')
    h.load_file('../hoc/naceaxon.nrn')
    for sec in h.allsec():
        sec.v = -100
    shape = h.Shape()

    ps = h.PlotShape(False)
    ps.variable('v')
    ax = ps.plot(plt, cmap=cm.jet).mark(h.dendA5_011111111111111111(0.35), size=1). \
        mark(h.dendA5_011111111111111111(0.50), size=1).mark(h.dendA5_011111111111111111(0.64), size=1). \
        mark(h.dendA5_011111111111111111(0.78), size=1).mark(h.dendA5_011111111111111111(0.92), size=1). \
        mark(h.dendA5_011111111111111110(0.59), size=1).mark(h.dendA5_011111111111111110(0.97), size=1). \
        mark(h.dendA5_011111111111111110(0.35), size=1).mark(h.dendA5_011111111111111110(0.11), size=1). \
        mark(h.dendA5_011111111111111110(0.78), size=1).mark(h.dendA5_011111111111111110(0.07), size=1). \
        mark(h.dendA5_011111111111111110(0.54), size=1).mark(h.dendA5_011111111111111110(0.40), size=1). \
        mark(h.dendA5_011111111111111110(0.21), size=1).mark(h.dendA5_0111111111111111111(0.73), size=1). \
        mark(h.dendA5_0111111111111111111(0.45), size=1).mark(h.dendA5_0111111111111111111(0.92), size=1). \
        mark(h.dendA5_0111111111111111111(0.40), size=1).mark(h.dendA5_0111111111111111111(0.64), size=1). \
        mark(h.dendA5_0111111111111111111(0.54), size=1).mark(h.dendA5_0111111111111111111(0.35), size=1). \
        mark(h.dendA5_0111111111111111111(0.83), size=1).mark(h.dendA5_0111111111111111111(0.30), size=1). \
        mark(h.dendA5_0111111111111111111(0.07), size=1).mark(h.dendA5_0111111111111111111(0.78), size=1). \
        mark(h.dendA5_0111111111111111111(0.02), size=1).mark(h.dendA5_0111111111111111111(0.11), size=1). \
        mark(h.dendA5_0111111111111111101(0.20), size=1).mark(h.dendA5_0111111111111111101(0.44), size=1). \
        mark(h.dendA5_0111111111111111101(0.73), size=1).mark(h.dendA5_0111111111111111101(0.97), size=1). \
        mark(h.dendA5_0111111111111111101(0.91), size=1).mark(h.dendA5_0111111111111111101(0.38), size=1). \
        mark(h.dendA5_0111111111111111101(0.32), size=1).mark(h.dendA5_0111111111111111101(0.02), size=1). \
        mark(h.dendA5_0111111111111111101(0.14), size=1).mark(h.dendA5_0111111111111111101(0.08), size=1). \
        mark(h.dendA5_0111111111111111100(0.80), size=1).mark(h.dendA5_0111111111111111100(0.33), size=1). \
        mark(h.dendA5_0111111111111111100(0.23), size=1).mark(h.dendA5_0111111111111111100(0.43), size=1). \
        mark(h.dendA5_0111111111111111100(0.78), size=1).mark(h.dendA5_0111111111111111100(0.19), size=1). \
        mark(h.dendA5_0111111111111111100(0.39), size=1).mark(h.dendA5_0111111111111111100(0.90), size=1). \
        mark(h.dendA5_0111111111111111100(0.03), size=1).mark(h.dendA5_0111111111111111100(0.17), size=1). \
        mark(h.dendA5_0111111111111111100(0.11), size=1).mark(h.dendA5_0111111111111111100(0.13), size=1). \
        mark(h.dendA5_01111111111111111111(0.70), size=1).mark(h.dendA5_01111111111111111110(0.03), size=1). \
        mark(h.dendA5_01111111111111111110(0.16), size=1).mark(h.dendA5_01111111111111111110(0.36), size=1). \
        mark(h.dendA5_01111111111111111110(0.90), size=1).mark(h.dendA5_01111111111111111110(0.56), size=1). \
        mark(h.dendA5_01111111111111111110(0.96), size=1).mark(h.dendA5_01111111111111111110(0.83), size=1). \
        mark(h.dendA5_01111111111111111011(0.23), size=1).mark(h.dendA5_01111111111111111011(0.70), size=1). \
        mark(h.dendA5_01111111111111111011(0.43), size=1).mark(h.dendA5_01111111111111111011(0.50), size=1). \
        mark(h.dendA5_01111111111111111010(0.58), size=1).mark(h.dendA5_01111111111111111010(0.78), size=1). \
        mark(h.dendA5_01111111111111111010(0.52), size=1).mark(h.dendA5_01111111111111111010(0.16), size=1). \
        mark(h.dendA5_01111111111111111010(0.83), size=1).mark(h.dendA5_01111111111111111010(0.38), size=1). \
        mark(h.dendA5_01111111111111111010(0.21), size=1).mark(h.dendA5_011111111111111111111(0.34), size=1). \
        mark(h.dendA5_011111111111111111111(0.55), size=1).mark(h.dendA5_011111111111111111111(0.39), size=1). \
        mark(h.dendA5_011111111111111111111(0.02), size=1).mark(h.dendA5_011111111111111111111(0.60), size=1). \
        mark(h.dendA5_011111111111111111111(0.50), size=1).mark(h.dendA5_011111111111111111111(0.86), size=1). \
        mark(h.dendA5_011111111111111111110(0.94), size=1).mark(h.dendA5_011111111111111111101(0.50), size=1). \
        mark(h.dendA5_011111111111111111101(0.90), size=1).mark(h.dendA5_011111111111111111100(0.04), size=1). \
        mark(h.dendA5_011111111111111111100(0.19), size=1).mark(h.dendA5_011111111111111111100(0.92), size=1). \
        mark(h.dendA5_011111111111111111100(0.62), size=1).mark(h.dendA5_011111111111111111100(0.46), size=1). \
        mark(h.dendA5_011111111111111111100(0.10), size=1).mark(h.dendA5_011111111111111111100(0.53), size=1). \
        mark(h.dendA5_011111111111111111100(0.74), size=1).mark(h.dendA5_011111111111111111100(0.01), size=1). \
        mark(h.dendA5_011111111111111110111(0.02), size=1).mark(h.dendA5_011111111111111110111(0.71), size=1). \
        mark(h.dendA5_011111111111111110111(0.84), size=1).mark(h.dendA5_011111111111111110111(0.45), size=1). \
        mark(h.dendA5_011111111111111110111(0.36), size=1).mark(h.dendA5_011111111111111110110(0.64), size=1). \
        mark(h.dendA5_011111111111111110110(0.26), size=1).mark(h.dendA5_011111111111111110110(0.35), size=1). \
        mark(h.dendA5_011111111111111110110(0.83), size=1).mark(h.dendA5_011111111111111110110(0.21), size=1). \
        mark(h.dendA5_0111111111111111111111(0.34), size=1).mark(h.dendA5_0111111111111111111111(0.28), size=1). \
        mark(h.dendA5_0111111111111111111110(0.13), size=1).mark(h.dendA5_0111111111111111111110(0.02), size=1). \
        mark(h.dendA5_0111111111111111111110(0.92), size=1).mark(h.dendA5_0111111111111111111110(0.71), size=1). \
        mark(h.dendA5_0111111111111111111110(0.65), size=1).mark(h.dendA5_0111111111111111111011(0.26), size=1). \
        mark(h.dendA5_0111111111111111111011(0.35), size=1).mark(h.dendA5_0111111111111111111011(0.54), size=1). \
        mark(h.dendA5_0111111111111111111011(0.11), size=1).mark(h.dendA5_0111111111111111111010(0.31), size=1). \
        mark(h.dendA5_0111111111111111111010(0.57), size=1).mark(h.dendA5_0111111111111111111010(0.35), size=1). \
        mark(h.dendA5_0111111111111111111010(0.83), size=1).mark(h.dendA5_0111111111111111111010(0.57), size=1). \
        mark(h.dendA5_0111111111111111111010(0.94), size=1).mark(h.dendA5_0111111111111111111010(0.24), size=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_proj_type('ortho')
    ax.view_init(elev=90, azim=-90, vertical_axis='z')
    ax.set_box_aspect(None, zoom=1.5)
    ax.set_axis_off()
    plt.savefig('./shape_synapses4.png', format='png', dpi=600)
    # plt.savefig('./shape_synapses3.pdf', format='pdf')
    # plt.savefig('./shape_synapses3.svg', format='svg')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
