"""
Title: CA1_pyramidal_cell.py
Author: Matus Tomko
Mail: matus.tomko __at__ fmph.uniba.sk
"""
import numpy as np
import neuron
from neuron import h, load_mechanisms
import sys

from CA1_plasticity.model.utils import RecordingVector
from CA1_plasticity.model.utils import Synapse


def single_pulse_NetStim():
    """Sets single pulse NetStim."""
    pulse = h.NetStim()
    pulse.interval = 1
    pulse.number = 1
    pulse.start = -1e9
    pulse.noise = 0
    return pulse


def insert_gauss_noise(gauss_sd, gauss_delay):
    """
    Inserts the gaussian variance.

    Parameters
    ----------
    gauss_sd : float
        the standard deviation
    gauss_delay : float
        the delay
    """
    pulse = h.gauss_noise()
    pulse.noise = gauss_sd
    pulse.delay = gauss_delay
    return pulse


def show_synapses_PP_LTP():
    """Shows a box containing the model shape plot with marked recording sites and synapses."""
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(8, 10))

    ps = h.PlotShape(False)
    ps.show(0)
    ps.plot(fig)
    plt.plot([-250, 250], [400, 400], 0, color='black', linewidth=0.5)
    plt.plot([-250, 250], [420, 420], 0, color='black', linewidth=0.5)
    plt.plot([-250, 250], [440, 440], 0, color='black', linewidth=0.5)
    plt.plot([-250, 250], [460, 460], 0, color='black', linewidth=0.5)
    plt.plot([-250, 250], [480, 480], 0, color='black', linewidth=0.5)
    plt.plot([-250, 250], [500, 500], 0, color='black', linewidth=0.5)
    plt.plot([-250, 250], [520, 520], 0, color='black', linewidth=0.5)
    plt.scatter(0, 0, s=100, marker='>', color='red', label='Recording electrode - 0 um')
    plt.scatter(0, 255, s=100, marker='>', color='green', label='Recording electrode - 255 um')
    plt.scatter(-100, 450, s=100, marker='>', color='blue', label='Recording electrode - 450 um')
    plt.axis('off')
    plt.legend(loc='lower right')
    ax = fig.axes[0]
    ax.view_init(elev=0, azim=0, vertical_axis='y')
    plt.show()


def genDendLocs(dends, nsyn, spread):
    # insert nsyn synapses to dendrites dends, uniform spread within a branch
    locs = []
    n_dends = len(dends)
    if isinstance(nsyn, list):
        nsyn = np.repeat(nsyn[0], len(dends))
    else:
        nsyn = np.repeat(nsyn, len(dends))
    for i_dend in np.arange(0, n_dends):
        dend = dends[i_dend]
        nsyn_dend = nsyn[i_dend]
        isd = (spread[1] - spread[0]) / float(nsyn_dend)
        pos = np.arange(spread[0], spread[1], isd)[0:nsyn_dend]

        if len(pos) != nsyn_dend:
            print('error: synapse number mismatch, stop simulation! dend:', i_dend, 'created=', len(pos), '!=',
                  nsyn_dend)
            sys.exit(1)
        for p in pos:
            locs.append([dend, p])
    return locs


class CA1PyramidalCell:
    """
    A class represents the CA1 pyramidal cell model

    ...

    Attributes
    ----------
    setting : dict
        the setting dictionary
    CA1 : neuron.hoc.HocObject
        the CA1 pyramidal cell model
    soma : neuron.hoc.HocObject
        the HocObject representing the soma
    all : list
        the list containing all sections of the model
    apical : list
        the list containing all apical dendrites sections
    basal : list
        the list containing the basal dendrites sections
    v_vec : neuron.hoc.HocObject
        the somatic voltage vector
    t_vec : neuron.hoc.HocObject
        the time vector for somatic voltage
    t_rs_vec : neuron.hoc.HocObject
        the time vector using recording step from the setting
    cai_vecs : dict
        the dictionary containing intracellular calcium concentration vectors
    cal2_ica_vecs : dict
        the dictionary containing CaL channel-mediated calcium current vectors
    dend_vecs : dict
        the dictionary containing voltage vectors from dendrites
    ina_vecs : dict
        the dictionary containing sodium current vectors
    nmda_ica_vecs : dict
        the dictionary of NMDAR channel-mediated calcium current vectors
    spines_v_vecs : list
        the list containing voltage vectors from spine heads
    apc : neuron.hoc.HocObject
        the action potential counter
    apc_vec : neuron.hoc.HocObject
        the action potential times vector
    bcm : neuron.hoc.HocObject
        the BCM mechanism
    alpha_scout_vec : neuron.hoc.HocObject
        the integrated spike count scaled by alpha vector
    d_vec : neuron.hoc.HocObject
        the depression amplitude vector
    p_vec : neuron.hoc.HocObject
        the potentiation amplitude vector
    synapses : dict
        the dictionary containing synapses
    net_cons : list
        the list containing NetCons
    net_stims : list
        the list containing NetStims
    rand_streams : neuron.hoc.HocObject
        the random stream generator
    stim : neuron.hoc.HocObject
        the IClamp point process
    stim_SEClamp : neuron.hoc.HocObject
        the SEClamp point process
    vBoxShape : neuron.hoc.HocObject
        the box organizes a collection of graphs and command panels
    shplot : neuron.hoc.HocObject
        the Shape window

    Methods
    -------
    apply_TTX()
        Simulates the application of TTX as a reduction of sodium channel conductance.
    apply_nimodipine()
        Simulates the application of nimodipine as a reduction of CaL channel conductance.
    connect_spontaneous_activity(lm_syn_count, ori_syn_count, rad_syn_count, random_weights)
        Connects NetStims for spontaneous activity with synapses using NetCons.
    get_synapses(sections)
        Returns a dictionary containing synapses.
    insert_AP_counter()
        Inserts an action potential counter at the soma.
    insert_BCM()
        Inserts the BCM mechanism at the soma.
    insert_current_clamp(section, x)
        Inserts a single pulse current clamp point process to a given section.
    insert_SEClamp(self, section, x)
        Inserts a single electrode voltage clamp point process to a given section.
    insert_AMPA_NMDA_PP_synapses(sections)
        Inserts AMPA and NMDA synapses at each segment of the sections.
    insert_json_synapses(synapses)
        Inserts synapses from a dictionary loaded from a .json file.
    insert_json_synapses_on_spines(synapses)
        Inserts synapses from a dictionary loaded from a .json file on spines.
    insert_synapses_on_spines(dend_locs, sec_name, distal)
        Inserts synapses on spines in defined locations.
    insert_synapses(dend_locs, sec_name)
        Inserts synapses in defined locations.
    insert_spine(sec, x, distal)
        Inserts a spine composed of the head and neck.
    reset_recording_vectors()
        Resets all used recording vectors.
    set_cai_vectors(sections)
        Sets vectors for recording of intracellular calcium concentration from sections in the list.
    set_cal2_ica_vectors(sections)
        Sets vectors for recording of CaL channel-mediated calcium current from sections in the list.
    set_dendritic_voltage_vectors(sections)
        Sets vectors for recording of voltage from sections in the list.
    set_nax_ina_vectors(sections)
        Sets vectors for recording of Na channel-mediated sodium current from sections in the list.
    set_NMDA_ica_vectors(sections):
        Sets vectors for recording of NMDA channel-mediated calcium current from sections in the list.
    set_NetStim(noise)
        Creates, sets and returns a NetStim.
    set_AMPA_synapse(sec, x)
        Creates, sets and returns an AMPA synapse.
    set_NMDA_synapse(sec, x)
        Creates, sets and returns a NMDA synapse.
    show_synapses_PP_LTP()
        Shows a box containing the model shape plot with marked recording sites and synapses.
    """

    def __init__(self, hoc_model, path_mods, setting):
        """
        Parameters
        ----------
        hoc_model : str
            the path to model in .hoc
        path_mods : str
            the path to the directory containing the .mod files
        setting : dict
            the dictionary containing setting
        """
        self.setting = setting

        # h.nrn_load_dll(path_mods + 'x86_64/libnrnmech.so')
        load_mechanisms(path_mods)
        h.xopen(hoc_model)
        h.xopen('../../model/hoc_models/rand_stream.hoc')

        self.CA1 = h.CA1_PC_Tomko()
        self.soma = self.CA1.soma[0]
        self.all = list(self.CA1.all)
        self.apical = list(self.CA1.apical)
        self.basal = list(self.CA1.basal)

        self.v_vec = h.Vector().record(self.soma(0.5)._ref_v)
        self.t_vec = h.Vector().record(h._ref_t)
        self.t_rs_vec = h.Vector().record(h._ref_t, self.setting['simulation']['RECORDING_STEP'])
        self.cai_vecs = {}
        self.cal2_ica_vecs = {}
        self.dend_vecs = {}
        self.ina_vecs = {}
        self.nmda_ica_vecs = {}
        self.spines_v_vecs = []

        self.apc = None
        self.apc_vec = h.Vector()
        self.stim = None
        self.stim_SEClamp = None

        self.bcm = None
        self.alpha_scout_vec = h.Vector()
        self.d_vec = h.Vector()
        self.p_vec = h.Vector()

        self.synapses = {}
        for sec in self.all:
            self.synapses[sec.hname().split('.')[1]] = []
        self.syn_AMPA_count = 0
        self.syn_NMDA_count = 0
        self.net_cons = []
        self.net_stims = []
        self.rand_streams = []

        self.nsd = []
        self.ns_spon = []
        self.ns_terminals = []
        self.nc_gauss = []
        self.nc_theta = []

        self.vBoxShape = None
        self.shplot = None

        self.spine_heads = []
        self.spine_necks = []

    def add_dendritic_tapering(self, section_name, diams):
        for sec in self.all:
            if sec.hname().split('.')[1] == section_name and sec.nseg == len(diams):
                i = 0
                for seg in sec:
                    seg.diam = diams[i]
                    seg.cm = 2
                    i = i + 1

    def apply_TTX(self):
        """Simulates the application of TTX as a reduction of sodium channel conductance."""
        for sec in self.CA1.all:
            if h.ismembrane('nax', sec=sec):
                sec.gbar_nax = sec.gbar_nax / 2

    def apply_nimodipine(self):
        """Simulates the application of nimodipine as a reduction of CaL channel conductance."""
        for sec in self.CA1.all:
            if h.ismembrane('cal', sec=sec):
                sec.gcalbar_cal = 0

    def connect_spontaneous_activity(self, lm_syn_count=None, ori_syn_count=None, rad_syn_count=None,
                                     random_weights=False):
        """
        Connects NetStims for spontaneous activity with synapses using NetCons.

        Parameters
        ----------
        lm_syn_count : int
            the number of synapses in the str. lacunosum-moleculare dendrites (default is None)
        ori_syn_count : int
            the number of synapses in the str. oriens dendrites (default is None)
        rad_syn_count : int
            the number of synapses in the str. radiatum dendrites (default is None)
        random_weights : bool
            if True, generates random weights (default is False)
        """
        # generate random weights from the normal distribution
        if random_weights and None not in {lm_syn_count, ori_syn_count, rad_syn_count} \
                and lm_syn_count + ori_syn_count + rad_syn_count == self.syn_AMPA_count:
            loc = (self.setting['initial_weights']['ORIENS_MIN_WEIGHT']
                   + self.setting['initial_weights']['ORIENS_MAX_WEIGHT']) / 2
            std = np.std([self.setting['initial_weights']['ORIENS_MIN_WEIGHT'],
                          self.setting['initial_weights']['ORIENS_MAX_WEIGHT']])
            ori_weights = np.random.normal(loc=loc, scale=std, size=ori_syn_count)

            loc = (self.setting['initial_weights']['RADIATUM_MIN_WEIGHT']
                   + self.setting['initial_weights']['RADIATUM_MAX_WEIGHT']) / 2
            std = np.std([self.setting['initial_weights']['RADIATUM_MIN_WEIGHT'],
                          self.setting['initial_weights']['RADIATUM_MAX_WEIGHT']])
            rad_weights = np.random.normal(loc=loc, scale=std, size=rad_syn_count)

            loc = (self.setting['initial_weights']['LM_MIN_WEIGHT']
                   + self.setting['initial_weights']['LM_MAX_WEIGHT']) / 2
            std = np.std([self.setting['initial_weights']['LM_MIN_WEIGHT'],
                          self.setting['initial_weights']['LM_MAX_WEIGHT']])
            lm_weights = np.random.normal(loc=loc, scale=std, size=lm_syn_count)
            o = 0
            r = 0
            l = 0
            for sec in self.synapses:
                if sec in self.setting['section_lists']['ori_secs']:
                    for syn in self.synapses[sec]:
                        syn.init_weight = ori_weights[o]
                        o = o + 1
                elif sec in self.setting['section_lists']['rad_secs']:
                    for syn in self.synapses[sec]:
                        syn.init_weight = rad_weights[r]
                        r = r + 1
                elif sec in self.setting['section_lists']['lm_secs']:
                    for syn in self.synapses[sec]:
                        syn.init_weight = lm_weights[l]
                        l = l + 1

        # connect synapses with NetStims using NetCons
        for sec in self.synapses:
            if len(self.synapses[sec]) > 0:
                i = 0
                while i < len(self.synapses[sec]):
                    # for syn in self.synapses[sec]:
                    stim1 = self.set_NetStim(syn_id=i)
                    # self.net_stims.append(stim1)

                    ns_terminal = single_pulse_NetStim()
                    self.ns_terminals.append(ns_terminal)
                    self.nc_gauss.append(h.NetCon(self.net_stims[-1], ns_terminal, 0, 0, 1))

                    if self.synapses[sec][i].receptor == 'AMPA':
                        self.synapses[sec][i].ns_terminal = ns_terminal
                        nc1 = h.NetCon(ns_terminal, self.synapses[sec][i].synapse, 0, 0,
                                       self.synapses[sec][i].init_weight)
                        self.synapses[sec][i].weight_vec.record(nc1._ref_weight[1],
                                                                self.setting['simulation']['RECORDING_STEP'])
                        nc1.record(self.synapses[sec][i].input_spikes)
                        self.net_cons.append(nc1)

                    if self.synapses[sec][i + 1].receptor == 'NMDA':
                        self.synapses[sec][i + 1].ns_terminal = ns_terminal
                        nc2 = h.NetCon(ns_terminal, self.synapses[sec][i + 1].synapse, 0, 0,
                                       self.synapses[sec][i + 1].init_weight)
                        self.net_cons.append(nc2)
                    i = i + 2

    def connect_gaussian_spontaneous_activity(self):
        for sec in self.synapses:
            if len(self.synapses[sec]) > 0:
                i = 0
                while i < len(self.synapses[sec]):
                    ntr = h.NetStimRes()
                    ntr.resonance = self.setting['netstim']['NETSTIMRES_RESONANCE']
                    ntr.interval = self.setting['netstim']['NETSTIMRES_INTERVAL']
                    ntr.number = self.setting['netstim']['NETSTIMRES_NUMBER']
                    ntr.start = self.setting['netstim']['NETSTIMRES_START']
                    ntr.noise = .5
                    self.nsd.append(ntr)

                    self.ns_spon.append(insert_gauss_noise(gauss_sd=self.setting['netstim']['GAUSS_SD'],
                                                           gauss_delay=self.setting['netstim']['GAUSS_DELAY']))
                    self.nc_theta.append(h.NetCon(self.nsd[-1], self.ns_spon[-1], 0, 0, 1))

                    ns_terminal = single_pulse_NetStim()
                    self.ns_terminals.append(ns_terminal)
                    self.nc_gauss.append(h.NetCon(self.ns_spon[-1], ns_terminal, 0, 0, 1))

                    if self.synapses[sec][i].receptor == 'AMPA':
                        self.synapses[sec][i].ns_terminal = ns_terminal
                        nc1 = h.NetCon(ns_terminal, self.synapses[sec][i].synapse, 0, 0,
                                       self.synapses[sec][i].init_weight)
                        self.synapses[sec][i].weight_vec.record(nc1._ref_weight[1],
                                                                self.setting['simulation']['RECORDING_STEP'])
                        nc1.record(self.synapses[sec][i].input_spikes)
                        self.net_cons.append(nc1)

                    if self.synapses[sec][i + 1].receptor == 'NMDA':
                        self.synapses[sec][i + 1].ns_terminal = ns_terminal
                        nc2 = h.NetCon(ns_terminal, self.synapses[sec][i + 1].synapse, 0, 0,
                                       self.synapses[sec][i + 1].init_weight)
                        self.net_cons.append(nc2)
                    i = i + 2

    def connect_random_spontaneous_activity(self):
        for sec in self.synapses:
            if len(self.synapses[sec]) > 0:
                i = 0
                while i < len(self.synapses[sec]):
                    stim = self.set_NetStim(syn_id=i)
                    self.net_cons.append(h.NetCon(stim, self.synapses[sec][i].ns_terminal, 0, 0, 1))
                    self.net_cons.append(h.NetCon(stim, self.synapses[sec][i + 1].ns_terminal, 0, 0, 1))
                    i = i + 2

    def connect_ns_terminals(self):
        for sec in self.synapses:
            if len(self.synapses[sec]) > 0:
                i = 0
                while i < len(self.synapses[sec]):
                    ns_terminal = single_pulse_NetStim()
                    self.ns_terminals.append(ns_terminal)

                    if self.synapses[sec][i].receptor == 'AMPA':
                        self.synapses[sec][i].ns_terminal = ns_terminal
                        nc1 = h.NetCon(ns_terminal, self.synapses[sec][i].synapse, 0, 0,
                                       self.synapses[sec][i].init_weight)
                        self.synapses[sec][i].weight_vec.record(nc1._ref_weight[1],
                                                                self.setting['simulation']['RECORDING_STEP'])
                        nc1.record(self.synapses[sec][i].input_spikes)
                        self.net_cons.append(nc1)

                    if self.synapses[sec][i + 1].receptor == 'NMDA':
                        self.synapses[sec][i + 1].ns_terminal = ns_terminal
                        nc2 = h.NetCon(ns_terminal, self.synapses[sec][i + 1].synapse, 0, 0,
                                       self.synapses[sec][i + 1].init_weight)
                        self.net_cons.append(nc2)
                    i = i + 2

    def get_synapses(self, sections):
        """
        Returns a dictionary containing synapses.

        Parameters
        -----------
        sections : list
            the list containing the names of sections

        Returns
        -------
        dict
            a dictionary containing synapses from sections in the list
        """
        return {sec: self.synapses[sec] for sec in sections if sec in self.synapses}

    def insert_AP_counter(self):
        """Inserts an action potential counter at the soma."""

        self.apc = h.APCount(self.soma(0.5))
        self.apc.thresh = 0
        self.apc.record(self.apc_vec)

    def insert_BCM(self):
        """Inserts the BCM mechanism at the soma."""

        self.bcm = h.BCMthreshold2(self.soma(0.5))
        # self.bcm.d0 = self.setting['BCM']['BCM_D0'] * self.setting['synapse']['SCALING_FACTOR']
        # self.bcm.p0 = self.setting['BCM']['BCM_P0'] * self.setting['synapse']['SCALING_FACTOR']
        self.bcm.alpha = self.setting['BCM']['BCM_ALPHA']
        self.bcm.scount0 = self.setting['BCM']['BCM_SCOUNT0']
        self.bcm.scounttau = self.setting['BCM']['BCM_SCOUNTTAU']

        self.alpha_scout_vec.record(self.bcm._ref_alpha_scount, self.setting['simulation']['RECORDING_STEP'])
        self.d_vec.record(self.bcm._ref_d, self.setting['simulation']['RECORDING_STEP'])
        self.p_vec.record(self.bcm._ref_p, self.setting['simulation']['RECORDING_STEP'])

    def insert_current_clamp(self, section, x):
        """
        Inserts single pulse current clamp point process to a given section.

        Parmaters
        ---------
        section : neuron.hoc.HocObject
            the section into the current clamp will be placed
        x : float
            the possition on the section
        """
        self.stim = h.IClamp(section(x))

    def insert_SEClamp(self, section, x):
        """
        Inserts a single electrode voltage clamp point process to a given section.

        Parmaters
        ---------
        section : neuron.hoc.HocObject
            the section into the current clamp will be placed
        x : float
            the possition on the section
        """
        self.stim_SEClamp = h.SEClamp(section(x))

    def insert_AMPA_NMDA_PP_synapses(self, sections):
        """
        Inserts AMPA and NMDA synapses at each segment of the sections.

        Parameters
        ----------
        sections : list
            the list containing the sections names
        """
        h.distance(0, self.soma(0.5))
        for sec in self.CA1.all:
            if sec.hname().split('.')[1] in sections:
                ica_vecs = []
                for seg in sec.allseg():
                    if seg.x not in [0.0, 1.0]:
                        dist = h.distance(seg.x, sec=sec)
                        stim = self.set_NetStim(noise=1)
                        # AMPA synapse
                        hoc_ampa_syn = h.Exp2SynSTDP_multNNb_globBCM_intscount_precentred(sec(seg.x))
                        hoc_ampa_syn.tau1 = self.setting['synapse']['AMPA_TAU1']
                        hoc_ampa_syn.tau2 = self.setting['synapse']['AMPA_TAU2']
                        hoc_ampa_syn.e = self.setting['synapse']['AMPA_E']
                        hoc_ampa_syn.start = self.setting['synapse']['AMPA_START']
                        hoc_ampa_syn.dtau = self.setting['synapse']['AMPA_DTAU']
                        hoc_ampa_syn.ptau = self.setting['synapse']['AMPA_PTAU']
                        h.setpointer(self.bcm._ref_d, 'd', hoc_ampa_syn)
                        h.setpointer(self.bcm._ref_p, 'p', hoc_ampa_syn)
                        ampa_syn = Synapse(synapse=hoc_ampa_syn, section=sec, segment_x=seg.x, distance=dist,
                                           init_weight=self.setting['synapse']['AMPA_WEIGHT'], weight_vec=h.Vector(),
                                           input_spikes_vec=h.Vector(), receptor='AMPA')
                        self.synapses[sec.hname().split('.')[1]].append(ampa_syn)
                        self.syn_AMPA_count = self.syn_AMPA_count + 1

                        # NMDA synapse
                        hoc_nmda_syn = h.Exp2SynNMDA(sec(seg.x))
                        hoc_nmda_syn.tau1 = self.setting['synapse']['NMDA_TAU1']
                        hoc_nmda_syn.tau2 = self.setting['synapse']['NMDA_TAU2']
                        hoc_nmda_syn.e = self.setting['synapse']['NMDA_E']
                        nmda_syn = Synapse(synapse=hoc_nmda_syn, section=sec, segment_x=seg.x, distance=dist,
                                           init_weight=self.setting['synapse']['NMDA_WEIGHT'], weight_vec=h.Vector(),
                                           input_spikes_vec=h.Vector(), receptor='NMDA')
                        self.synapses[sec.hname().split('.')[1]].append(nmda_syn)

                        vec = h.Vector().record(hoc_nmda_syn._ref_ica)
                        ica_vec = RecordingVector(section=sec.hname(), segment_x=seg.x, vec=vec)
                        ica_vecs.append(ica_vec)
                        self.syn_NMDA_count = self.syn_NMDA_count + 1

                self.nmda_ica_vecs[sec.hname().split('.')[1]] = ica_vecs

    def insert_json_synapses(self, synapses):
        """
        Inserts synapses from a dictionary loaded from a .json file.

        Parameters
        ----------
        synapses : dict
            the dictionary containing synapses loaded from a .json file
        """
        i = 0
        for sec in self.CA1.all:
            for s in synapses[sec.hname().split('.')[1]]:
                if s['receptor'] == 'AMPA':
                    hoc_ampa_syn = self.set_AMPA_synapse(sec=sec, x=float(s['segment_x']))
                    dist = h.distance(float(s['segment_x']), sec=sec)
                    syn = Synapse(synapse=hoc_ampa_syn, synapse_id=i, section=sec, segment_x=float(s['segment_x']),
                                  distance=dist, weight_vec=h.Vector(),
                                  init_weight=s['init_weight'] * self.setting['synapse']['SCALING_FACTOR'],  # 0.0006
                                  input_spikes_vec=h.Vector(), receptor=s['receptor'], type=s['type'],
                                  d_amp_vec=h.Vector().record(hoc_ampa_syn._ref_d,
                                                             self.setting['simulation']['RECORDING_STEP']),
                                  p_amp_vec=h.Vector().record(hoc_ampa_syn._ref_p,
                                                              self.setting['simulation']['RECORDING_STEP']))
                    syn.pathway = s['pathway']
                    self.synapses[sec.hname().split('.')[1]].append(syn)
                    self.syn_AMPA_count = self.syn_AMPA_count + 1

                if s['receptor'] == 'NMDA':
                    hoc_nmda_syn = self.set_NMDA_synapse(sec=sec, x=float(s['segment_x']))
                    dist = h.distance(float(s['segment_x']), sec=sec)
                    syn = Synapse(synapse=hoc_nmda_syn, synapse_id=i, section=sec, segment_x=float(s['segment_x']),
                                  distance=dist, weight_vec=h.Vector(),
                                  init_weight=s['init_weight'] * self.setting['synapse']['SCALING_FACTOR'],  # 0.0008
                                  input_spikes_vec=h.Vector(), receptor=s['receptor'], type=s['type'],
                                  d_amp_vec=h.Vector(), p_amp_vec=h.Vector())
                    syn.pathway = s['pathway']
                    self.synapses[sec.hname().split('.')[1]].append(syn)
                    self.syn_NMDA_count = self.syn_NMDA_count + 1
                    i = i + 1

        print('Total number of AMPA synapses: ' + str(self.syn_AMPA_count))
        print('Total number of NMDA synapses: ' + str(self.syn_NMDA_count))

    def insert_json_synapses_on_spines(self, synapses):
        """
        Inserts synapses from a dictionary loaded from a .json file on spines.

        Parameters
        ----------
        synapses : dict
            the dictionary containing synapses loaded from a .json file
        """
        i = 0
        for sec in self.CA1.all:
            for s in synapses[sec.hname().split('.')[1]]:
                head = self.insert_spine(sec=sec, x=float(s['segment_x']))
                if s['receptor'] == 'AMPA':
                    hoc_ampa_syn = self.set_AMPA_synapse(sec=head, x=0.5)
                    dist = h.distance(float(s['segment_x']), sec=sec)
                    syn = Synapse(synapse=hoc_ampa_syn, synapse_id=i, section=sec, segment_x=float(s['segment_x']),
                                  distance=dist, weight_vec=h.Vector(),
                                  init_weight=s['init_weight'] * self.setting['synapse']['SCALING_FACTOR'],  # 0.0006
                                  input_spikes_vec=h.Vector(), receptor=s['receptor'], type=s['type'],
                                  d_amp_vec=h.Vector().record(hoc_ampa_syn._ref_d,
                                                              self.setting['simulation']['RECORDING_STEP']),
                                  p_amp_vec=h.Vector().record(hoc_ampa_syn._ref_p,
                                                              self.setting['simulation']['RECORDING_STEP']))
                    syn.pathway = s['pathway']
                    self.synapses[sec.hname().split('.')[1]].append(syn)
                    self.syn_AMPA_count = self.syn_AMPA_count + 1

                if s['receptor'] == 'NMDA':
                    hoc_nmda_syn = self.set_NMDA_synapse(sec=head, x=0.5)
                    dist = h.distance(float(s['segment_x']), sec=sec)
                    syn = Synapse(synapse=hoc_nmda_syn, synapse_id=i, section=sec, segment_x=float(s['segment_x']),
                                  distance=dist, weight_vec=h.Vector(),
                                  init_weight=s['init_weight'] * self.setting['synapse']['SCALING_FACTOR'],  # 0.0008
                                  input_spikes_vec=h.Vector(), receptor=s['receptor'], type=s['type'],
                                  d_amp_vec=h.Vector(), p_amp_vec=h.Vector())
                    syn.pathway = s['pathway']
                    self.synapses[sec.hname().split('.')[1]].append(syn)
                    self.syn_NMDA_count = self.syn_NMDA_count + 1
                    i = i + 1

        print('Total number of AMPA synapses: ' + str(self.syn_AMPA_count))
        print('Total number of NMDA synapses: ' + str(self.syn_NMDA_count))

    def insert_synapses_on_spines(self, dend_locs, sec_name):
        """
        Inserts synapses on spines in defined locations.

        Parameters
        ----------
        dend_locs : list
            the list containing locations
        sec_name : str
            the section name
        distal : bool
            if distal = True, AMPA and NMDA weights are divided by 2
        """
        print('adding synapses')

        AMPA_gmax = self.setting['synapse']['AMPA_GMAX'] * self.setting['synapse']['SCALING_FACTOR']
        NMDA_gmax = self.setting['synapse']['NMDA_GMAX'] * self.setting['synapse']['SCALING_FACTOR']

        for sec in self.CA1.all:
            if sec.hname().split('.')[1] == sec_name:
                for s in range(len(dend_locs)):
                    head = self.insert_spine(sec=sec, x=dend_locs[s][1])
                    hoc_ampa_syn = self.set_AMPA_synapse(sec=head, x=0.5)
                    dist = h.distance(dend_locs[s][1], sec=sec)
                    syn = Synapse(synapse=hoc_ampa_syn, synapse_id=s, section=dend_locs[s][0],
                                  segment_x=dend_locs[s][1],
                                  distance=dist, weight_vec=h.Vector(), init_weight=AMPA_gmax,
                                  input_spikes_vec=h.Vector(), receptor='AMPA', type='perforated',
                                  d_amp_vec=h.Vector().record(hoc_ampa_syn._ref_d, self.setting['simulation']['RECORDING_STEP']),
                                  p_amp_vec=h.Vector().record(hoc_ampa_syn._ref_p, self.setting['simulation']['RECORDING_STEP']))
                    self.synapses[sec.hname().split('.')[1]].append(syn)
                    self.syn_AMPA_count = self.syn_AMPA_count + 1

                    hoc_nmda_syn = self.set_NMDA_synapse(sec=head, x=0.5)
                    dist = h.distance(dend_locs[s][1], sec=sec)
                    syn = Synapse(synapse=hoc_nmda_syn, synapse_id=s, section=dend_locs[s][0],
                                  segment_x=dend_locs[s][1],
                                  distance=dist, weight_vec=h.Vector(), init_weight=NMDA_gmax,
                                  input_spikes_vec=h.Vector(), receptor='NMDA', type='perforated',
                                  d_amp_vec=h.Vector(), p_amp_vec=h.Vector())
                    self.synapses[sec.hname().split('.')[1]].append(syn)
                    self.syn_NMDA_count = self.syn_NMDA_count + 1

        print('Total number of AMPA synapses: ' + str(self.syn_AMPA_count))
        print('Total number of NMDA synapses: ' + str(self.syn_NMDA_count))

    def insert_synapses(self, dend_locs, sec_name):
        """
        Inserts synapses in defined locations.

        Parameters
        ----------
        dend_locs : list
            the list containing locations
        sec_name : str
            the section name
        """
        print('adding synapses')

        AMPA_gmax = self.setting['synapse']['AMPA_GMAX'] * self.setting['synapse']['SCALING_FACTOR']
        NMDA_gmax = self.setting['synapse']['NMDA_GMAX'] * self.setting['synapse']['SCALING_FACTOR']

        for sec in self.CA1.all:
            if sec.hname().split('.')[1] == sec_name:
                for s in range(len(dend_locs)):
                    hoc_ampa_syn = self.set_AMPA_synapse(sec=sec, x=dend_locs[s][1])
                    dist = h.distance(dend_locs[s][1], sec=sec)
                    syn = Synapse(synapse=hoc_ampa_syn, synapse_id=s, section=dend_locs[s][0],
                                  segment_x=dend_locs[s][1],
                                  distance=dist, weight_vec=h.Vector(), init_weight=AMPA_gmax,
                                  input_spikes_vec=h.Vector(), receptor='AMPA', type='perforated',
                                  d_amp_vec=h.Vector().record(hoc_ampa_syn._ref_d,
                                                              self.setting['simulation']['RECORDING_STEP']),
                                  p_amp_vec=h.Vector().record(hoc_ampa_syn._ref_p,
                                                              self.setting['simulation']['RECORDING_STEP']))
                    self.synapses[sec.hname().split('.')[1]].append(syn)
                    self.syn_AMPA_count = self.syn_AMPA_count + 1

                    hoc_nmda_syn = self.set_NMDA_synapse(sec=sec, x=dend_locs[s][1])
                    dist = h.distance(dend_locs[s][1], sec=sec)
                    syn = Synapse(synapse=hoc_nmda_syn, synapse_id=s, section=dend_locs[s][0],
                                  segment_x=dend_locs[s][1],
                                  distance=dist, weight_vec=h.Vector(), init_weight=NMDA_gmax,
                                  input_spikes_vec=h.Vector(), receptor='NMDA', type='perforated',
                                  d_amp_vec=h.Vector(), p_amp_vec=h.Vector())
                    self.synapses[sec.hname().split('.')[1]].append(syn)
                    self.syn_NMDA_count = self.syn_NMDA_count + 1

        print('Total number of AMPA synapses: ' + str(self.syn_AMPA_count))
        print('Total number of NMDA synapses: ' + str(self.syn_NMDA_count))

    def insert_spine(self, sec, x):
        """
        Inserts a spine composed of the head and neck.

        Parameters
        ----------
        sec : neuron.hoc.HocObject
            the section on which the spine will be inserted
        x : neuron.hoc.HocObject
            the location of the spine in the section

        Returns
        -------
        neuron.hoc.HocObject
            head
        """
        neck = h.Section()
        neck.L = self.setting['spine']['neck_L']
        neck.diam = self.setting['spine']['neck_diam']
        neck.insert('pas')
        neck.e_pas = self.setting['spine']['neck_e_pas']
        neck.g_pas = self.setting['spine']['neck_g_pas']
        neck.Ra = self.setting['spine']['neck_Ra']
        neck.cm = self.setting['spine']['neck_cm']

        head = h.Section()
        head.L = self.setting['spine']['head_L']
        head.diam = self.setting['spine']['head_diam']
        head.insert('pas')
        head.e_pas = self.setting['spine']['head_e_pas']
        head.g_pas = self.setting['spine']['head_g_pas']
        head.Ra = self.setting['spine']['head_Ra']
        head.cm = self.setting['spine']['head_cm']

        head.connect(neck, 1, 0)
        neck.connect(sec, x, 0)

        self.spine_necks.append(neck)
        self.spine_heads.append(head)

        vec = h.Vector().record(head(0.5)._ref_v)
        v_vec = RecordingVector(section=head.hname(), segment_x=0.5, vec=vec)
        self.spines_v_vecs.append(v_vec)

        return head

    def print_dend_params(self, dend_name):
        for dend in self.all:
            if dend.hname().split('.')[1] == dend_name:
                print(dend.name(), dend.L, dend.nseg, self.all.index(dend))
                print('---------------------------------------------------')
                for seg in dend:
                    xdist = h.distance(seg, sec=dend)
                    print(seg, xdist, seg.diam, seg.cm, seg.g_pas, seg.gbar_nax, seg.gkabar_kad, seg.gkdrbar_kdr)

    def reset_recording_vectors(self):
        """Resets all used recording vectors."""
        self.v_vec.resize(0)
        self.t_vec.resize(0)
        self.apc_vec.resize(0)
        self.t_rs_vec.resize(0)
        self.d_vec.resize(0)
        self.p_vec.resize(0)

        for sec in self.synapses:
            for syn in self.synapses[sec]:
                syn.weight_vec.resize(0)
                syn.d_amp_vec.resize(0)
                syn.p_amp_vec.resize(0)
                syn.input_spikes.resize(0)

        for d in [self.dend_vecs, self.ina_vecs, self.cai_vecs, self.cal2_ica_vecs, self.nmda_ica_vecs]:
            for sec in d:
                for vec in d[sec]:
                    vec.vector.resize(0)

        for vec in self.spines_v_vecs:
            vec.vector.resize(0)

    def set_cai_vectors(self, sections):
        """
        Sets vectors for recording of intracellular calcium concentration from sections in the list.

        Parameters
        ----------
        sections : list
            the list containing the names of sections
        """
        for sec in self.CA1.all:
            if sec.hname().split('.')[1] in sections:
                cai_vecs = []
                for seg in sec.allseg():
                    vec = h.Vector().record(seg._ref_cai)
                    cai_vec = RecordingVector(section=sec.hname(), segment_x=seg.x, vec=vec)
                    cai_vecs.append(cai_vec)
                self.cai_vecs[sec.hname().split('.')[1]] = cai_vecs

    def set_cal2_ica_vectors(self, sections):
        """
        Sets vectors for recording of CaL channel-mediated calcium current from sections in the list.

        Parameters
        ----------
        sections : list
            the list containing the names of sections
        """
        for sec in self.CA1.all:
            if sec.hname().split('.')[1] in sections and h.ismembrane('cal', sec=sec):
                cal2_ica_vecs = []
                for seg in sec.allseg():
                    vec = h.Vector().record(seg._ref_ica_cal)
                    cal2_ica_vec = RecordingVector(section=sec.hname(), segment_x=seg.x, vec=vec)
                    cal2_ica_vecs.append(cal2_ica_vec)
                self.cal2_ica_vecs[sec.hname().split('.')[1]] = cal2_ica_vecs

    def set_dendritic_voltage_vectors(self, sections):
        """
        Sets vectors for recording of voltage from sections in the list.

        Parameters
        ----------
        sections : list
            the list containing the names of sections
        """
        for sec in self.CA1.all:
            if sec.hname().split('.')[1] in sections:
                d_vecs = []
                for seg in sec.allseg():
                    vec = h.Vector().record(seg._ref_v)
                    d_vec = RecordingVector(section=sec.hname(), segment_x=seg.x, vec=vec)
                    d_vecs.append(d_vec)
                self.dend_vecs[sec.hname().split('.')[1]] = d_vecs

    def set_nax_ina_vectors(self, sections):
        """
        Sets vectors for recording of Na channel-mediated sodium current from sections in the list.

        Parameters
        ----------
        sections : list
            the list containing the names of sections
        """
        for sec in self.CA1.all:
            if sec.hname().split('.')[1] in sections and h.ismembrane('nax', sec=sec):
                na_vecs = []
                for seg in sec.allseg():
                    vec = h.Vector().record(seg._ref_ina_nax)
                    na_vec = RecordingVector(section=sec.hname(), segment_x=seg.x, vec=vec)
                    na_vecs.append(na_vec)
                self.ina_vecs[sec.hname().split('.')[1]] = na_vecs

    def set_NMDA_ica_vectors(self, sections):
        """
        Sets vectors for recording of NMDA channel-mediated calcium current from sections in the list.

        Parameters
        ----------
        sections : list
            the list containing the names of sections
        """
        for sec in self.synapses:
            if sec in sections:
                ica_vecs = []
                for syn in self.synapses[sec]:
                    if syn.receptor == 'NMDA':
                        vec = h.Vector().record(syn.synapse._ref_icc)
                        ica_vec = RecordingVector(section=sec, segment_x=float(syn.segment_x), vec=vec)
                        ica_vecs.append(ica_vec)
                self.nmda_ica_vecs[sec] = ica_vecs

    def set_NetStim(self, syn_id):
        """
        Creates, sets and returns a NetStim.

        Parameters
        ----------
        syn_id : int
            the identificator

        Returns
        -------
        neuron.hoc.HocObject
            NetStim
        """
        stim = h.NetStim()
        self.net_stims.append(stim)
        stim.start = self.setting['netstim']['NETSTIM_START']  # * np.random.rand()
        if self.setting['netstim']['NETSTIM_FREQUENCY'] == 0:
            stim.interval = 1e9
        else:
            stim.interval = 1000 / self.setting['netstim']['NETSTIM_FREQUENCY']
        stim.number = self.setting['netstim']['NETSTIM_NUMBER']
        stim.noise = self.setting['netstim']['NETSTIM_NOISE']

        rs = h.RandomStream(syn_id)
        self.rand_streams.append(rs)
        stim.noiseFromRandom(rs.r)
        rs.r.negexp(1)
        rs.start()

        return stim

    def set_AMPA_synapse(self, sec, x):
        """
        Creates, sets and returns a AMPA synapse.

        Parameters
        ----------
        sec : neuron.hoc.HocObject
            the section into which synapse will be inserted
        x : neuron.hoc.HocObject
            the synapse location at the section

        Returns
        -------
        neuron.hoc.HocObject
            synapse
        """
        # syn = h.Exp2SynSTDP_multNNb_globBCM_intscount_precentred2(sec(x))
        syn = h.Exp2SynETDP_multNNb_precentred(sec(x))
        syn.tau1 = self.setting['synapse']['AMPA_TAU1']
        syn.tau2 = self.setting['synapse']['AMPA_TAU2']
        syn.e = self.setting['synapse']['AMPA_E']
        syn.start = self.setting['synapse']['AMPA_START']
        syn.dtau = self.setting['synapse']['AMPA_DTAU']
        syn.ptau = self.setting['synapse']['AMPA_PTAU']
        syn.d = self.setting['synapse']['AMPA_D0']
        syn.p = self.setting['synapse']['AMPA_P0']
        # syn.d0 = self.setting['synapse']['AMPA_D0']
        # syn.p0 = self.setting['synapse']['AMPA_P0']
        # h.setpointer(self.bcm._ref_alpha_scount, 'alpha_scount', syn)

        '''
        if sec.hname().split('.')[1] in self.setting['section_lists']['ori_secs']:
            syn.d0 = self.setting['BCM']['BCM_ORI_D0'] * self.setting['synapse']['SCALING_FACTOR']
            syn.p0 = self.setting['BCM']['BCM_ORI_P0'] * self.setting['synapse']['SCALING_FACTOR']
        elif sec.hname().split('.')[1] in self.setting['section_lists']['rad_secs']:
            syn.d0 = self.setting['BCM']['BCM_RAD_D0'] * self.setting['synapse']['SCALING_FACTOR']
            syn.p0 = self.setting['BCM']['BCM_RAD_P0'] * self.setting['synapse']['SCALING_FACTOR']
        else:
            syn.d0 = self.setting['BCM']['BCM_LM_D0'] * self.setting['synapse']['SCALING_FACTOR']
            syn.p0 = self.setting['BCM']['BCM_LM_P0'] * self.setting['synapse']['SCALING_FACTOR']

        h.setpointer(self.bcm._ref_d, 'd', syn)
        h.setpointer(self.bcm._ref_p, 'p', syn)
        '''

        return syn

    def set_NMDA_synapse(self, sec, x):
        """
        Creates, sets and returns a NMDA synapse.

        Parameters
        ----------
        sec : neuron.hoc.HocObject
            the section into which synapse will be inserted
        x : neuron.hoc.HocObject
            the synapse location at the section

        Returns
        -------
        neuron.hoc.HocObject
            synapse
        """
        # syn = h.Exp2SynNMDA(sec(x))
        syn = h.Exp2SynNMDA2(sec(x))
        syn.tau1 = self.setting['synapse']['NMDA_TAU1']
        syn.tau2 = self.setting['synapse']['NMDA_TAU2']
        syn.e = self.setting['synapse']['NMDA_E']
        return syn
