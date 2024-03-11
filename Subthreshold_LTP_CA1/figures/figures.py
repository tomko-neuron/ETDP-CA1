import json
import pickle
import gzip
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt, cm
from neuron import h, gui2

plt.style.use('seaborn-colorblind')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['lines.markersize'] = 1.5
PATH = '../../../recordings/Kim_2015/TBS_stim/w_0.00072/'
PATH_SAVE = '../../experiments/experiment3/figures/'
REC_LM_SEC = 'dendA5_01111111111111111010'
REC_LM_X = 0.5888888888888889
REC_RAD_SEC = 'radTdist2'
REC_RAD_X = 0.1
REC_SOMA = 'soma'
REC_SOMA_X = 0.5
DENDS = ['dend1_00', 'dend5_00', 'dend2_01001', 'dend2_01110', 'dend2_010100']
EXPERIMENTS = ['2_synapses_distal', '3_synapses_distal', '4_synapses_distal', '8_synapses_distal']
EXPERIMENTS_OTHERS = ['12_synapses_proximal', '14_synapses_proximal', '16_synapses_proximal', '4_11_synapses_proximal',
                      '4_synapses_distal_short', '8_synapses_distal_short']


def main():
    # avg_LTP()
    # avg_LTP_others()
    fig_LTP_distal()
    fig_LTP_others()
    statistics_figure_3()
    statistics_others()
    fig_spine_voltage_weight_zoom()
    fig_spine_voltage_weight()
    fig_shape()


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
        for d in DENDS:
            synapses = load_json_data('../recordings/' + d + '/' + e + '/synapses.json')
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


def avg_LTP_others():
    AVG_LTP = {}
    for e in EXPERIMENTS_OTHERS:
        AVG_LTP[e] = {}
        synapses = load_json_data('../recordings/dend5_00/' + e + '/synapses.json')
        ltp = []
        for sec in synapses:
            if len(synapses[sec]) > 0:
                for syn in synapses[sec]:
                    if syn['receptor'] == 'AMPA':
                        ltp.append(syn['final_weight'] / syn['init_weight'])
        AVG_LTP[e]['mean'] = np.mean(ltp)
        AVG_LTP[e]['std'] = np.std(ltp)
        AVG_LTP[e]['sem'] = np.std(ltp) / np.sqrt(len(ltp))
    try:
        with open('avg_LTP_others.json', 'w') as f:
            json.dump(AVG_LTP, f, indent=4)
    except FileNotFoundError as fnf_error:
        raise fnf_error


def fig_LTP_distal():
    LTP = {}
    avg = []
    std = []
    sem = []
    # avg_paper_distal = [0.84, 1.35, 1.32, 1.52]
    # sem_paper_distal = [0.11, 0.13, 0.11, 0.20]
    # avg_paper_distal = [1.1476101288582556, 1.1476101288582556, 1.18]   # old
    # std_paper_distal = [3.245931617, 4.293963916, 3.54]     # old
    # sem_paper_distal = [0.07, 0.07, 0.10]   # old

    avg_paper_distal = [0.9088785046728969, 1.2523364485981303, 1.0957943925233642, 1.1892428487945217]
    sem_paper_distal = [0.105140187, 0.16588785, 0.053738318, 0.091504713]
    std_paper_distal = [0.297381357, 0.469201696, 0.201070374, 0.274514138]
    n_runs = [8, 8, 14, 9]

    data = load_json_data('./avg_LTP.json')
    for e in EXPERIMENTS:
        avg.append(np.average(data[e]))
        std.append(np.std(data[e], axis=0))
        sem.append(std[-1] / np.sqrt(len(data[e])))

    barWidth = 0.4
    r1 = np.arange(start=0, stop=4, step=1)
    r2 = np.arange(start=0.4, stop=4.4, step=1)
    r3 = np.arange(start=0.2, stop=4.2, step=1)

    ticks = ['2_synapses', '3_synapses', '4_synapses', '8_synapses']
    fig = plt.figure(figsize=(2.8, 2.24), dpi=300)
    ax = plt.axes()
    ax.bar(r1, avg_paper_distal, yerr=sem_paper_distal, width=barWidth, label='exp. data', color='#949494',
           error_kw=dict(lw=0.5, capsize=2.0, capthick=0.5))
    ax.bar(r2, avg, yerr=sem, width=barWidth, label='simulation', color='#029e73',
           error_kw=dict(lw=0.5, capsize=2.0, capthick=0.5))
    ax.set_xticks(r3, ticks, rotation=45)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False, bottom=False)
    # ax.autoscale(tight=True)
    ax.set_ylabel('Potentiation ratio\n (Post TBS / Baseline)')
    ax.set_ylim(0.8, 1.5)
    ax.legend(fontsize=4)
    fig.tight_layout()
    plt.savefig('./LTP_distal_10min_v3.svg', format='svg')
    plt.savefig('./LTP_distal_10min_v3.png', format='png')
    plt.show()
    plt.close(fig)


def fig_LTP_others():
    LTP = {}
    avg = []
    std = []
    sem = []

    # 12-16 proximal, 4+11 proximal, 4 distal short, 8 distal short
    avg_paper = [0.8849, 2.0691, 1.1189, 0.9390]
    sem_paper = [0.079, 0.4858, 0.1433, 0.0671]
    n_runs = [5, 7, 5, 4]

    data = load_json_data('./avg_LTP_others.json')
    for e in EXPERIMENTS_OTHERS:
        avg.append(data[e]['mean'])
        std.append(data[e]['std'])
        sem.append(data[e]['sem'])

    avg = [np.mean(avg[:3])] + avg[3:]
    std = [np.mean(std[:3])] + std[3:]
    sem = [np.mean(sem[:3])] + sem[3:]

    barWidth = 0.4
    r1 = np.arange(start=0, stop=4, step=1)
    r2 = np.arange(start=0.4, stop=4.4, step=1)
    r3 = np.arange(start=0.2, stop=4.2, step=1)

    ticks = ['12-16_synapses_proximal', '4+11_synapses_proximal', '4_synapses_distal_short', '8_synapses_distal_short']
    fig = plt.figure(figsize=(2.8, 2.24), dpi=300)
    ax = plt.axes()
    ax.bar(r1, avg_paper, yerr=sem_paper, width=barWidth, label='exp. data', color='#949494',
           error_kw=dict(lw=0.5, capsize=2.0, capthick=0.5))
    ax.bar(r2, avg, yerr=sem, width=barWidth, label='simulation', color='#029e73',
           error_kw=dict(lw=0.5, capsize=2.0, capthick=0.5))
    ax.set_xticks(r3, ticks, rotation=45)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', top=False, right=False)
    # ax.autoscale(tight=True)
    ax.set_ylabel('Potentiation ratio\n (Post TBS / Baseline)')
    ax.set_ylim(0.6, 2.6)
    ax.legend(fontsize=4)
    fig.tight_layout()
    plt.savefig('./LTP_distal_10min_others.svg', format='svg')
    plt.savefig('./LTP_distal_10min_others.png', format='png')
    plt.show()
    plt.close(fig)


def statistics_figure_3():
    avg_paper_distal = [0.9088785046728969, 1.2523364485981303, 1.0957943925233642, 1.1892428487945217]
    sem_paper_distal = [0.105140187, 0.16588785, 0.053738318, 0.091504713]
    std_paper_distal = [0.297381357, 0.469201696, 0.201070374, 0.274514138]
    n_runs = [8, 8, 14, 9]

    data = load_json_data('./avg_LTP.json')

    from scipy.stats import ttest_ind_from_stats

    avg = []
    std = []
    sem = []
    for e in EXPERIMENTS:
        avg.append(np.average(data[e]))
        std.append(np.std(data[e], axis=0))
        sem.append(std[-1] / np.sqrt(len(data[e])))

    for i in range(4):
        # Perform one-sample t-test
        stat, p_value = ttest_ind_from_stats(mean1=avg[i], mean2=avg_paper_distal[i], std1=std[i],
                                             std2=std_paper_distal[i], nobs1=5, nobs2=n_runs[i], equal_var=False,
                                             alternative='two-sided')

        # Check for statistical significance
        if p_value < 0.05:
            print(EXPERIMENTS[i], p_value,
                  "The mean of the simulated data is significantly different from the overall "
                  "mean of the experimental data.")
        else:
            print(EXPERIMENTS[i], p_value,
                  "There is no significant difference between the mean of the simulated data and the overall mean "
                  "of the experimental data.")


def statistics_others():
    avg_paper = [0.8849, 2.0691, 1.1189, 0.9390]
    sem_paper = [0.079, 0.4858, 0.1433, 0.0671]
    std_paper = [0.1766, 1.2853, 0.3204, 0.1342]
    n_runs = [5, 7, 5, 4]

    data = load_json_data('./avg_LTP_others.json')

    from scipy.stats import ttest_ind_from_stats

    avg = []
    std = []
    sem = []
    for e in EXPERIMENTS_OTHERS:
        avg.append(data[e]['mean'])
        std.append(data[e]['std'])
        sem.append(data[e]['sem'])

    avg = [np.mean(avg[:3])] + avg[3:]
    std = [np.mean(std[:3])] + std[3:]
    sem = [np.mean(sem[:3])] + sem[3:]

    ticks = ['12_16_synapses_proximal', '4_11_synapses_proximal', '4_synapses_distal_short', '8_synapses_distal_short']
    for i in range(4):
        # Perform one-sample t-test
        stat, p_value = ttest_ind_from_stats(mean1=avg[i], mean2=avg_paper[i], std1=std[i],
                                             std2=std_paper[i], nobs1=5, nobs2=n_runs[i], equal_var=False,
                                             alternative='two-sided')

        # Check for statistical significance
        if p_value < 0.05:
            print(ticks[i], p_value,
                  "The mean of the simulated data is significantly different from the overall "
                  "mean of the experimental data.")
        else:
            print(ticks[i], p_value,
                  "There is no significant difference between the mean of the simulated data and the overall mean "
                  "of the experimental data.")


def fig_spine_voltage_weight_zoom():
    labels_distal = ['2_synapses', '3_synapses', '4_synapses', '8_synapses']
    colors = ['#d55e00', '#cc78bc', '#ca9161', '#fbafe4']

    fig1 = plt.figure(figsize=(2.3, 1.75), dpi=300)
    ax1 = plt.axes()
    for i in range(4):
        label = labels_distal[i]
        data = pickle.load(gzip.GzipFile('../recordings/dend5_00/' + label + '_distal/voltages.p', 'rb'))
        ax1.plot(data['T'], data['V_spines'][0].vector, label=label, color=colors[i])
    ax1.hlines(y=-37, xmin=11995, xmax=12045, colors='black', linestyles='dotted')
    ax1.set_xlim(11995, 12045)
    ax1.set_ylabel('Voltage (mV)')
    ax1.set_xlabel('Time (ms)')
    ax1.tick_params(axis='both', which='both', top=False, right=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    fig1.tight_layout()
    fig1.savefig('./spine_volt_zoom.svg', format='svg')
    fig1.savefig('./spine_volt_zoom.png', format='png')
    plt.show()
    plt.close(fig1)

    fig2 = plt.figure(figsize=(2.3, 1.75), dpi=300)
    ax2 = plt.axes()
    for i in range(4):
        label = labels_distal[i]
        data = pickle.load(gzip.GzipFile('../recordings/dend5_00/' + label + '_distal/synapses.p', 'rb'))
        ax2.plot(data['T'], data['synapses']['dend5_00'][0]['weight'] * 1000, label=label, color=colors[i])
    ax2.set_xlim(11995, 12045)
    ax2.set_ylim(0.597, 0.606)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel(r'Synaptic weight (nS)')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='both', which='both', top=False, right=False)
    ax2.legend(loc='upper right', fontsize=4)
    fig2.tight_layout()
    fig2.savefig('./spine_weight_zoom.svg', format='svg')
    fig2.savefig('./spine_weight_zoom.png', format='png')
    plt.show()
    plt.close(fig2)


def fig_spine_voltage_weight():
    labels_distal = ['8_synapses', '4_synapses', '3_synapses', '2_synapses']
    colors = ['#fbafe4', '#ca9161', '#cc78bc', '#d55e00']

    fig = plt.figure(figsize=[4.17, 1.75], dpi=300)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    for i in range(4):
        label = labels_distal[i]
        data = pickle.load(gzip.GzipFile('../recordings/dend5_00/' + label + '_distal/voltages.p', 'rb'))
        ax1.plot(data['T'], data['V_spines'][0].vector, label=label, color=colors[i])
        data = pickle.load(gzip.GzipFile('../recordings/dend5_00/' + label + '_distal/synapses.p', 'rb'))
        ax2.plot(data['T'], data['synapses']['dend5_00'][0]['weight'] * 1000, label=label, color=colors[i])
    ax1.hlines(y=-37, xmin=11500, xmax=29000, colors='black', linestyles='dotted')
    ax1.set_xlim(11500, 29000)
    ax1.set_ylabel('Voltage (mV)')
    ax1.tick_params(axis='both', which='both', top=False, right=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2.set_xlim(11500, 29000)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel(r'Synaptic weight (nS)')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='both', which='both', top=False, right=False)

    fig.tight_layout()
    fig.savefig('./spine_volt_weight.svg', format='svg')
    fig.savefig('./spine_volt_weight.png', format='png')
    plt.show()
    plt.close(fig)


def fig_shape():
    s = 'xopen("../CA1.hoc")'
    h(s)
    for sec in h.allsec():
        if sec.hname() in DENDS:
            sec.v = -90
        else:
            sec.v = -55
    ps = h.PlotShape(False)
    ps.variable('v')
    ax = (ps.plot(plt, cmap=cm.jet).mark(h.dend5_00(0.96), size=3).mark(h.dend1_00(0.96), size=3).
          mark(h.dend2_01001(0.96), size=3).mark(h.dend2_01110(0.96), size=3).mark(h.dend2_010100(0.96), size=3))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_proj_type('ortho')
    ax.view_init(elev=90, azim=-90, vertical_axis='z')
    ax.set_box_aspect(None, zoom=1.5)
    ax.set_axis_off()
    plt.savefig('./shape.png', format='png', dpi=300)
    plt.savefig('./shape.svg', format='svg')
    plt.show()


if __name__ == '__main__':
    main()
