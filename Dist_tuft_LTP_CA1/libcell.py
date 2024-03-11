import numpy as np
import neuron
from neuron import h, gui, load_mechanisms
from CA1_plasticity.model.utils import RecordingVector
from CA1_plasticity.model.utils import Synapse
import sys

class CA1:

    def __init__(self, setting):
        self.setting = setting
        load_mechanisms('./mods/')
        h.xopen('./hoc/morphology_ri06.nrn')
        h.xopen('./hoc/naceaxon.nrn')
        h.load_file('./hoc/resetNSeg.hoc')
        h.xopen('./hoc/init.hoc')
        h.load_file('./hoc/initializationAndRun.hoc')
        h.initchannels(0)

        self.soma = h.somaA
        self.distTuft = h.distTuft
        self.all_apicals = h.all_apicals
        self.all_basals = h.all_basals
        self.primary_apical_list = h.primary_apical_list

        self.v_vec = h.Vector().record(self.soma(0.5)._ref_v)
        self.t_vec = h.Vector().record(h._ref_t)
        self.t_rs_vec = h.Vector().record(h._ref_t, self.setting['simulation']['RECORDING_STEP'])
        self.dend_vecs = {}
        self.nmda_ica_vecs = {}
        self.ina_vecs = {}
        self.calH_ica_vecs = {}
        self.cai_vecs = {}
        self.ogb_vecs = {}
        self.pmp_vecs = {}
        self.spines_v_vecs = []

        self.stim = None
        self.bcm = None
        self.alpha_scout_vec = h.Vector()
        self.d_vec = h.Vector()
        self.p_vec = h.Vector()

        self.synapses = {}
        for dend in self.all_apicals:
            self.synapses[dend.hname()] = []
        for dend in self.all_basals:
            self.synapses[dend.hname()] = []
        self.syn_AMPA_count = 0
        self.syn_NMDA_count = 0
        self.net_cons = []
        self.net_stims = []
        self.rand_streams = []
        self.vec_stims = []
        self.apc_vec = h.Vector()

        self.ns_spon = []
        self.ns_terminals = []
        self.vecs = []

        self.spine_heads = []
        self.spine_necks = []

        self.stim_SEClamp = None
        self.iclamp_t_vec = None
        self.iclamp_amps_vec = None
        self.ppStim = h.ppStim

    def add_synapses_distTuf(self):
        neurite_area = 0
        neur_areas = h.Vector(h.numDistNeurites)
        neur_names = []
        sec_list_dist = []
        num_dist_neurites = 0
        for sec in self.distTuft:
            num_dist_neurites = num_dist_neurites + 1
            sec_list_dist.append(sec)
            for x in sec:
                neurite_area = neurite_area + x.area()
            neur_areas.x[num_dist_neurites - 1] = neurite_area
            neur_names.append(sec.hname())
            neurite_area = 0

        # DISTRIBUTE SYNAPSES
        dist_neur_sum = neur_areas.sum()
        norm_neur_areas = h.Vector()
        nnaInt = h.Vector()
        norm_neur_areas.copy(neur_areas)
        norm_neur_areas.div(dist_neur_sum)
        nnaInt.integral(norm_neur_areas)

        rand_gen = h.Random(self.setting['simulation']['SEED'])
        rand_gen.uniform(0, 1)
        rand_gen_anat = h.Random(self.setting['simulation']['SEED'] + 1e6)
        rand_gen_anat.uniform(0, 1)

        # AMPA_gmax = self.setting['synapse']['AMPA_GMAX'] * self.setting['synapse']['SCALING_FACTOR']
        # NMDA_gmax = self.setting['synapse']['NMDA_GMAX'] * self.setting['synapse']['SCALING_FACTOR']

        # Generate random synaptic weigths from normal distribution
        num_weights = 150
        mean_weight = self.setting['synapse']['AMPA_GMAX_MEAN']
        sigma = self.setting['synapse']['AMPA_GMAX_SIGMA']

        # Generate random weights from a normal distribution
        # synaptic_weights = np.random.normal(loc=mean_weight, scale=sigma, size=num_weights)

        # Generate random weights from a log-normal distribution
        synaptic_weights = np.random.lognormal(mean=np.log(mean_weight), sigma=sigma, size=num_weights)

        cur_syn = 0
        h('access somaA')
        h('distance()')
        for i in range(150):
            cur_rand = rand_gen.repick()
            while cur_rand > nnaInt.x[cur_syn]:
                cur_syn = cur_syn + 1

            h('access ' + neur_names[cur_syn])
            cur_syn = 0
            cur_rand_anat = rand_gen_anat.repick()
            cur_rand_anat_B = (int(cur_rand_anat * h.nseg) * 2 + 1) / (h.nseg * 2)

            syn_ampa = h.Exp2SynETDP_multNNb_precentred(cur_rand_anat_B)
            syn_ampa.tau1 = self.setting['synapse']['AMPA_TAU1']
            syn_ampa.tau2 = self.setting['synapse']['AMPA_TAU2']
            syn_ampa.e = self.setting['synapse']['AMPA_E']
            syn_ampa.start = self.setting['synapse']['AMPA_START']
            syn_ampa.dtau = self.setting['synapse']['AMPA_DTAU']
            syn_ampa.ptau = self.setting['synapse']['AMPA_PTAU']
            syn_ampa.d = self.setting['synapse']['AMPA_D0']
            syn_ampa.p = self.setting['synapse']['AMPA_P0']

            AMPA_gmax = synaptic_weights[i] * self.setting['synapse']['SCALING_FACTOR']
            syn = Synapse(synapse=syn_ampa, synapse_id=i, section=h.secname(),
                          segment_x=cur_rand_anat_B,
                          distance=h.distance(cur_rand_anat_B), weight_vec=h.Vector(),
                          init_weight=AMPA_gmax,
                          input_spikes_vec=h.Vector(), receptor='AMPA', type='perforated',
                          d_amp_vec=h.Vector(), p_amp_vec=h.Vector())

            self.synapses[h.secname()].append(syn)
            self.syn_AMPA_count = self.syn_AMPA_count + 1

            syn_nmda = h.Exp2SynNMDA_SLM(cur_rand_anat_B)
            syn_nmda.tau1 = self.setting['synapse']['NMDA_TAU1']
            syn_nmda.tau2 = self.setting['synapse']['NMDA_TAU1']
            syn_nmda.e = self.setting['synapse']['NMDA_E']

            NMDA_gmax = synaptic_weights[i] * self.setting['synapse']['SCALING_FACTOR']
            syn = Synapse(synapse=syn_nmda, synapse_id=i, section=h.secname(),
                          segment_x=cur_rand_anat_B,
                          distance=h.distance(cur_rand_anat_B), weight_vec=h.Vector(),
                          init_weight=NMDA_gmax,
                          input_spikes_vec=h.Vector(), receptor='NMDA', type='perforated',
                          d_amp_vec=h.Vector(), p_amp_vec=h.Vector())
            self.synapses[h.secname()].append(syn)
            self.syn_NMDA_count = self.syn_NMDA_count + 1

        h('access somaA')
        print('Total number of AMPA synapses: ' + str(self.syn_AMPA_count))
        print('Total number of NMDA synapses: ' + str(self.syn_NMDA_count))

    def print_dend_params(self, dend_name):
        h('access somaA')
        h('distance()')
        for dend in self.all_apicals:
            if dend.hname() == dend_name:
                print(dend.name(), dend.L, dend.nseg)
                print(dend.psection()['density_mechs'].keys())
                print('---------------------------------------------------')
                for seg in dend:
                    xdist = h.distance(seg, sec=dend)
                    print(seg, xdist, seg.diam, seg.cm, seg.g_pas, seg.gkabar_kad, seg.gkabar_kap, seg.gkdrbar_kdr,
                          seg.gbar_nax, seg.gcalbar_calH)

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
        self.stim_SEClamp.dur1 = self.setting['protocol']['SEClamp']['DUR1']
        self.stim_SEClamp.amp1 = self.setting['protocol']['SEClamp']['AMP1']

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

    def set_recording_vectors_dist_tuft(self):
        for sec in self.all_apicals:
            if sec.hname() in self.setting['dends_recordings']:
                d_vecs = []
                na_vecs = []
                calH_ica_vecs = []
                cai_vecs = []
                ogb_vecs = []
                pmp_vecs = []
                for seg in [0.1, 0.5, 0.9]:  # for seg in sec.allseg():
                    d_vec = RecordingVector(section=sec.hname(), segment_x=seg, vec=h.Vector().record(sec(seg)._ref_v))
                    d_vecs.append(d_vec)
                    na_vec = RecordingVector(section=sec.hname(), segment_x=seg,
                                             vec=h.Vector().record(sec(seg)._ref_ina_nax))
                    na_vecs.append(na_vec)
                    calH_vec = RecordingVector(section=sec.hname(), segment_x=seg,
                                               vec=h.Vector().record(sec(seg)._ref_ica_calH))
                    calH_ica_vecs.append(calH_vec)
                    cai_vec = RecordingVector(section=sec.hname(), segment_x=seg,
                                              vec=h.Vector().record(sec(seg)._ref_cai))
                    cai_vecs.append(cai_vec)

                self.dend_vecs[sec.hname()] = d_vecs
                self.ina_vecs[sec.hname()] = na_vecs
                self.calH_ica_vecs[sec.hname()] = calH_ica_vecs
                self.cai_vecs[sec.hname()] = cai_vecs
                self.ogb_vecs[sec.hname()] = ogb_vecs
                self.pmp_vecs[sec.hname()] = pmp_vecs

    def apply_TTX(self):
        """Simulates the application of TTX as a reduction of sodium channel conductance."""
        ttxScale = 0.5  # amount that 20 nM TTX scales the available Na conductance; 1 = noblock; 0 = complete block
        gnainit0 = 0.042    # Na conductance at soma
        gnaslope0 = 0.000025    # Na channel density decay per um
        gnainit = gnainit0 * ttxScale
        gnaslope = gnaslope0 * ttxScale

        h('access somaA')
        h('area(0.5)')
        h('distance()')
        for sec in self.all_apicals:
            for x in sec:
                xdist = h.distance(x, sec=sec)
                x.gbar_nax = gnainit - xdist * gnaslope


    def set_theta_burst_iclamp(self, stim):
        """
        Sets times for current injections for the theta burst pairing stimulation protocol.

        Parameters
        ----------
        stim : neuron.hoc.HocObject
            the object of single pulse current clamp point process
        """
        stim.delay = 0
        stim.dur = 1e9
        stim.amp = 0

        t_start = self.setting['protocol']['theta_burst_iclamp']['TB_ICLAMP_START']
        t_vec = np.zeros(0)
        for pattern in range(self.setting['protocol']['theta_burst_iclamp']['TB_ICLAMP_PATTERNS_NUM']):
            for burst in range(self.setting['protocol']['theta_burst_iclamp']['TB_ICLAMP_BURSTS_NUM']):
                t_stop = t_start + 1 + (self.setting['protocol']['theta_burst_iclamp']['TB_ICLAMP_STIMULI_NUM'] - 1) * \
                         self.setting['protocol']['theta_burst_iclamp']['TB_ICLAMP_INTERSPIKE_INTERVAL']
                burst_vec = np.arange(t_start,
                                      t_stop,
                                      self.setting['protocol']['theta_burst_iclamp']['TB_ICLAMP_INTERSPIKE_INTERVAL'])
                t_vec = np.concatenate((t_vec, burst_vec), axis=0)
                # t_start = t_vec[-1] + self.setting['protocol']['theta_burst_pairing']['TBP_INTERBURST_INTERVAL']
                t_start = t_start + self.setting['protocol']['theta_burst_iclamp']['TB_ICLAMP_INTERBURST_INTERVAL']
            t_start = t_start + self.setting['protocol']['theta_burst_iclamp']['TB_ICLAMP_PATTERNS_INTERVAL']

        t_vec_iclamp = [0]
        for t in t_vec:
            t_vec_iclamp.append(t)
            t_vec_iclamp.append(t)
            t_vec_iclamp.append(t + self.setting['protocol']['theta_burst_iclamp']['TB_ICLAMP_DUR'])
            t_vec_iclamp.append(t + self.setting['protocol']['theta_burst_iclamp']['TB_ICLAMP_DUR'])
        t_vec_iclamp.append(self.setting['simulation']['TSTOP'])
        self.iclamp_t_vec = h.Vector(t_vec_iclamp)

        amps_vec = []
        for i in range(len(t_vec_iclamp)):
            if (i % 4 == 0) or (i % 4 == 1):
                amps_vec.append(0.0)
            elif (i % 4 == 2) or (i % 4 == 3):
                amps_vec.append(self.setting['protocol']['theta_burst_iclamp']['TB_ICLAMP_AMP'])
        self.iclamp_amps_vec = h.Vector(amps_vec)
        self.iclamp_amps_vec.play(stim._ref_amp, self.iclamp_t_vec, 1)

    def set_theta_burst(self):
        """
        Sets the theta burst stimulation protocol using Vecstim objects.
        """
        for sec in self.synapses:
            s = 0
            while s < len(self.synapses[sec]):
                t_start = self.setting['protocol']['theta_burst']['TB_START'] + np.random.rand()
                t_vec = np.zeros(0)
                for pattern in range(self.setting['protocol']['theta_burst']['TB_PATTERNS_NUM']):
                    for burst in range(self.setting['protocol']['theta_burst']['TB_BURSTS_NUM']):
                        t_stop = t_start + 1 + (
                                self.setting['protocol']['theta_burst']['TB_STIMULI_NUM'] - 1) * \
                                 self.setting['protocol']['theta_burst']['TB_INTERSPIKE_INTERVAL']
                        burst_vec = np.arange(t_start, t_stop,
                                              self.setting['protocol']['theta_burst']['TB_INTERSPIKE_INTERVAL'])
                        t_vec = np.concatenate((t_vec, burst_vec), axis=0)
                        t_start = t_start + self.setting['protocol']['theta_burst']['TB_INTERBURST_INTERVAL']
                    t_start = t_start + self.setting['protocol']['theta_burst']['TB_PATTERNS_INTERVAL']

                vec = h.Vector(t_vec)
                self.vecs.append(vec)
                vec_stim = h.VecStim()
                vec_stim.play(vec)
                self.vec_stims.append(vec_stim)
                nc_AMPA = h.NetCon(vec_stim, self.synapses[sec][s].ns_terminal, 0, 0, 1)
                self.net_cons.append(nc_AMPA)
                self.synapses[sec][s].stimulated = True
                nc_NMDA = h.NetCon(vec_stim, self.synapses[sec][s + 1].ns_terminal, 0, 0, 1)
                self.net_cons.append(nc_NMDA)
                self.synapses[sec][s + 1].stimulated = True
                s = s + 2

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

        for d in [self.dend_vecs, self.ina_vecs, self.cai_vecs, self.calH_ica_vecs, self.nmda_ica_vecs, self.pmp_vecs]:
            for sec in d:
                for vec in d[sec]:
                    vec.vector.resize(0)

        for vec in self.spines_v_vecs:
            vec.vector.resize(0)


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


def single_pulse_NetStim():
    """Sets single pulse NetStim."""
    pulse = h.NetStim()
    pulse.interval = 1
    pulse.number = 1
    pulse.start = -1e9
    pulse.noise = 0
    return pulse
