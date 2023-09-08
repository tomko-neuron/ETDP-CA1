import matplotlib.pyplot as plt
import numpy as np
import neuron
from neuron import h, gui, load_mechanisms
from CA1_plasticity.model.utils import RecordingVector
from CA1_plasticity.model.utils import Synapse
import sys


class CA1:

    def __init__(self, hoc_model, path_mods, setting, ttx=False):
        self.setting = setting
        load_mechanisms(path_mods)
        # h.nrn_load_dll('C:/Users/tomko/PycharmProjects/Neuron_CA1/Subthreshold_LTP_CA1/mods/nrnmech.dll')
        s = 'xopen("' + hoc_model + '")'
        h(s)
        propsCA1(self, ttx=ttx)
        self._topol()
        self._biophys()
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
        for dend in self.dends:
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
        self.ppStim = h.SpGen2(0.5)

    def _topol(self):
        self.soma = h.soma
        self.hill = h.hill
        self.iseg = h.iseg
        self.node = h.node
        self.inode = h.inode
        self.dends = []
        self.distTuft = h.distTuft
        for sec in h.allsec():
            self.dends.append(sec)
            n_seg = int(max(1, round(sec.L / self.seclength + 0.5)))
            if (n_seg - int(n_seg / 2) * 2) == 0:
                n_seg = n_seg + 1
            sec.nseg = n_seg
        for i in np.arange(8):
            self.dends.pop()  # Remove soma and axons from the list
        # self.axon.connect(self.soma,1,0)

    def _biophys(self):
        for sec in h.allsec():
            sec.cm = self.CM
            sec.insert('pas')
            sec.e_pas = self.E_PAS
            sec.g_pas = 1.0 / self.RM
            sec.Ra = self.RA

        h.soma.g_pas = 1.0 / self.RM_soma

        h.node[0].g_pas = 1.0 / self.RM_node
        h.node[0].cm = self.CM

        h.node[1].g_pas = 1.0 / self.RM_node
        h.node[1].cm = self.CM

        h.inode[0].g_pas = 1.0 / self.RM
        h.inode[0].cm = self.CM_inode

        h.inode[1].g_pas = 1.0 / self.RM
        h.inode[1].cm = self.CM_inode

        h.inode[2].g_pas = 1.0 / self.RM
        h.inode[2].cm = self.CM_inode

        # compensate for spines
        h('access soma')
        h('distance()')
        for sec in self.dends:
            # for sec in h.all_apicals: # in the Katz model compensation was done only for apicals.
            # print('-----------------------------------')
            # print(sec.name(), sec.L, sec.nseg)
            nseg = sec.nseg
            iseg = 0
            prox = False
            dist = False
            for seg in sec:
                # 0. calculate the distance from soma
                xx = iseg * 1.0 / nseg + 1.0 / nseg / 2.0
                xdist = h.distance(xx, sec=sec)
                # 1. calculate the diameter of the segment
                xdiam = seg.diam
                # print(sec.name(), xx, xdiam)
                if (xdist > self.spinelimit) + (xdiam < self.spinediamlimit):
                    seg.cm = self.CM * self.spinefactor
                    seg.g_pas = self.spinefactor * 1.0 / self.RM  # _dend
                    # sec.Ra = self.RA_dend  # Ra is a section variable...
                if xx < 0.4 and xdist <= 91:
                    prox = True
                if xx > 0.6 and xdist >= 177:
                    dist = True
                iseg = iseg + 1
            if prox and dist:
                print(sec.hname())

    def print_dend_params(self, dend_name):
        h('access soma')
        h('distance()')
        print(h.soma.L, h.soma.nseg, h.dend5_0.L, h.dend5_0.nseg)
        for dend in self.dends:
            if dend.hname() == dend_name:
                print(dend.name(), dend.L, dend.nseg, self.dends.index(dend))
                print(dend.psection()['density_mechs'].keys())
                print('---------------------------------------------------')
                for seg in dend:
                    xdist = h.distance(seg, sec=dend)
                    print(seg, xdist, seg.diam, seg.cm, seg.g_pas, seg.gkabar_kad, seg.gkabar_kap, seg.gkdrbar_kdr,
                          seg.gbar_nad)

    def add_synapses_on_spines(self, data, dend_name):
        print('adding synapses')

        AMPA_gmax = self.setting['synapse']['AMPA_GMAX'] * self.setting['synapse']['SCALING_FACTOR']
        NMDA_gmax = self.setting['synapse']['NMDA_GMAX'] * self.setting['synapse']['SCALING_FACTOR']
        h('access soma')
        h('distance()')
        for dend in self.dends:
            if dend.hname() == dend_name:
                for s in range(data.Ensyn):
                    head = self.insert_spine(data=data, sec=dend, x=data.Elocs[s][1])
                    hoc_ampa_syn = self.set_AMPA_synapse(sec=head, x=0.5)
                    dist = h.distance(data.Elocs[s][1], sec=dend)
                    syn = Synapse(synapse=hoc_ampa_syn, synapse_id=s, section=dend,
                                  segment_x=data.Elocs[s][1],
                                  distance=dist, weight_vec=h.Vector(), init_weight=AMPA_gmax,
                                  input_spikes_vec=h.Vector(), receptor='AMPA', type='perforated',
                                  d_amp_vec=h.Vector(),
                                  p_amp_vec=h.Vector())
                    self.synapses[dend.hname()].append(syn)
                    self.syn_AMPA_count = self.syn_AMPA_count + 1

                    hoc_nmda_syn = self.set_NMDA_synapse(sec=head, x=0.5)
                    dist = h.distance(data.Elocs[s][1], sec=dend)
                    syn = Synapse(synapse=hoc_nmda_syn, synapse_id=s, section=dend,
                                  segment_x=data.Elocs[s][1],
                                  distance=dist, weight_vec=h.Vector(), init_weight=NMDA_gmax,
                                  input_spikes_vec=h.Vector(), receptor='NMDA', type='perforated',
                                  d_amp_vec=h.Vector(), p_amp_vec=h.Vector())
                    self.synapses[dend.hname()].append(syn)
                    self.syn_NMDA_count = self.syn_NMDA_count + 1

        print('Total number of AMPA synapses: ' + str(self.syn_AMPA_count))
        print('Total number of NMDA synapses: ' + str(self.syn_NMDA_count))

    def insert_spine(self, data, sec, x):
        neck = h.Section()
        neck.L = data.sneck_len
        neck.diam = data.sneck_diam
        neck.insert("pas")
        neck.e_pas = self.E_PAS
        neck.g_pas = 1.0 / self.RM
        neck.Ra = self.RA
        neck.cm = self.CM

        head = h.Section()
        head.L = data.shead_len
        head.diam = data.shead_diam
        head.insert("pas")
        head.e_pas = self.E_PAS
        head.g_pas = 1.0 / self.RM
        head.Ra = self.RA
        head.cm = self.CM

        head.connect(neck, 1, 0)
        neck.connect(sec, x, 0)

        self.spine_necks.append(neck)
        self.spine_heads.append(head)

        vec = h.Vector().record(head(0.5)._ref_v)
        v_vec = RecordingVector(section=head.hname(), segment_x=0.5, vec=vec)
        self.spines_v_vecs.append(v_vec)

        return head

    def set_AMPA_synapse(self, sec, x):
        syn = h.Exp2SynETDP_multNNb_precentred(sec(x))
        syn.tau1 = self.setting['synapse']['AMPA_TAU1']
        syn.tau2 = self.setting['synapse']['AMPA_TAU2']
        syn.e = self.setting['synapse']['AMPA_E']
        syn.start = self.setting['synapse']['AMPA_START']
        syn.dtau = self.setting['synapse']['AMPA_DTAU']
        syn.ptau = self.setting['synapse']['AMPA_PTAU']
        syn.d = self.setting['synapse']['AMPA_D0']
        syn.p = self.setting['synapse']['AMPA_P0']
        return syn

    def set_NMDA_synapse(self, sec, x):
        syn = h.Exp2SynNMDA(sec(x))
        syn.tau1 = self.setting['synapse']['NMDA_TAU1']
        syn.tau2 = self.setting['synapse']['NMDA_TAU2']
        syn.e = self.setting['synapse']['NMDA_E']
        return syn

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

    def set_dendritic_voltage_vectors(self, sections):
        for sec in self.dends:
            if sec.hname() in sections:
                d_vecs = []
                for seg in sec.allseg():
                    vec = h.Vector().record(seg._ref_v)
                    d_vec = RecordingVector(section=sec.hname(), segment_x=seg.x, vec=vec)
                    d_vecs.append(d_vec)
                self.dend_vecs[sec.hname()] = d_vecs

    def set_nad_ina_vectors(self, sections):
        """
        Sets vectors for recording of Na channel-mediated sodium current from sections in the list.

        Parameters
        ----------
        sections : list
            the list containing the names of sections
        """
        for sec in self.dends:
            if sec.hname() in sections and h.ismembrane('nad', sec=sec):
                na_vecs = []
                for seg in sec.allseg():
                    vec = h.Vector().record(seg._ref_ina_nad)
                    na_vec = RecordingVector(section=sec.hname(), segment_x=seg.x, vec=vec)
                    na_vecs.append(na_vec)
                self.ina_vecs[sec.hname()] = na_vecs

    def apply_TTX(self):
        """Simulates the application of TTX as a reduction of sodium channel conductance."""
        for sec in self.dends:
            if h.ismembrane('nad', sec=sec):
                sec.gbar_nad = sec.gbar_nad * 0.2

    def set_Makara_LTP(self, num_synapses, start, interval, num, protocol):
        for sec in self.synapses:
            s = 0
            t = 0
            while s < len(self.synapses[sec]) and s < num_synapses * 2:
                stim = h.NetStim()
                self.net_stims.append(stim)
                if protocol:
                    stim.start = start  # + (t * 0.1)
                    self.synapses[sec][s].stimulated = True
                    self.synapses[sec][s + 1].stimulated = True
                else:
                    stim.start = start + (t * 200)
                stim.interval = interval
                stim.number = num
                stim.noise = 0
                nc_AMPA = h.NetCon(stim, self.synapses[sec][s].ns_terminal, 0, 0, 1)
                self.net_cons.append(nc_AMPA)
                nc_NMDA = h.NetCon(stim, self.synapses[sec][s + 1].ns_terminal, 0, 0, 1)
                self.net_cons.append(nc_NMDA)
                s = s + 2
                t = t + 1

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


def propsCA1(model, ttx=False):
    # Passive properties
    model.CELSIUS = 35
    model.v_init = -68.3  # -68.3 for theta and -72 for replay
    model.RA = 100  # 150.00           # internal resistivity in ohm-cm
    model.RA_dend = 100  # 200
    model.CM = 1  # 1                # 0.75     # specific membrane capacitance in uF/cm^2
    model.CM_inode = 0.04  # capacitance in myelin

    model.RM = 20000  # was 3000 for hCond;  20000       # specific membrane resistivity at in ohm-cm^2
    model.RM_dend = 20000  # 10 000 - 40 000; only used in Robustness analysis in Adam's paper
    model.RM_soma = 40000  # was 3000 for hCond;  20000       # specific membrane resistivity at the soma in ohm-cm^2
    model.RM_inode = 40000  # 200000          # inter-nodal resistivity with myelin
    model.RM_node = 50  # 200000          # nodal resistivity

    model.E_PAS = -66  # -66 - set to v_init if passive
    model.seclength = 10  # um, the length of a section
    model.spinefactor = 2  # 2 factor by which to change passive properties
    model.spinelimit = 100  # 100 distance beyond which to modify passive properties to account for spines
    model.spinediamlimit = 1  # 100 distance beyond which to modify passive properties to account for spines

    # Active properties - Values from the Spruston-lab (Katz et al., 2009) fitted only to trunk data!
    if ttx:
        model.gna = 0.03 * 0.2  # 0.03; 0.01 sodium conductance in terminal branches
        model.gna_trunk = 0.04 * 0.2  # 0.04 sodium conductance
        model.naslope = 0.001 * 0.2  # 0.001 is 'strong' propagation on the trunk
    else:
        model.gna = 0.03  # 0.03; 0.01 sodium conductance in terminal branches
        model.gna_trunk = 0.04  # 0.04 sodium conductance
        model.naslope = 0.001  # 0.001 is 'strong' propagation on the trunk
    model.gna_axon = 0.04  # 0.04 sodium conductance in the axon
    model.gna_soma = 0.2  # 0.04 - 0.2 sodium conductance in the soma
    model.gna_node = 15  # 30 - 15 sodium conductance in the axon
    model.nalimit = 500
    model.gna_dend_hotSpot = 5

    model.gkdr = 0.02  # 0.005 delayed rectifier density in terminal branches
    model.gkdr_trunk = 0.040  # 0.04 delayed rectifier density in the trunk
    model.gkdr_soma = 0.04  # 0.04 delayed rectifier density at the soma
    model.gkdr_axon = 0.04  # 0.04 delayed rectifier density at the axon

    model.gka = model.gkdr  # 0.005 A-type potassium density in terminal branches
    model.gka_trunk = 0.048  # 0.048  A-type potassium starting density in the trunk
    model.dlimit = 500  # cut-off for increase of A-type density
    model.dprox = 100  # distance to switch from proximal to distal type
    model.dslope = 0.01  # slope of A-type density

    model.gcalH_distTuft = 0.00125  # L-type Ca density, from Poirazi et al., 2003


def init_active(model, axon=False, soma=False, dend=True, dendNa=False,
                dendCa=False):
    if axon:
        model.axon.insert('na')
        model.axon.gbar_na = model.gna_axon
        model.axon.insert('kv')
        model.axon.gbar_kv = model.gkv_axon
        model.axon.ena = model.Ena
        model.axon.ek = model.Ek
        print('active conductances added in the axon')

    if soma:
        model.soma.insert('na')
        model.soma.gbar_na = model.gna_soma
        model.soma.insert('kv')
        model.soma.gbar_kv = model.gkv_soma
        model.soma.insert('km')
        model.soma.gbar_km = model.gkm_soma
        model.soma.insert('kca')
        model.soma.gbar_kca = model.gkca_soma
        model.soma.insert('ca')
        model.soma.gbar_ca = model.gca_soma
        model.soma.insert('it')
        model.soma.gbar_it = model.git_soma
        model.soma.insert('cad')
        model.soma.ena = model.Ena
        model.soma.ek = model.Ek
        model.soma.eca = model.Eca
        print('somatic active conductances enabled')

    if dend:
        for d in model.dends:
            d.insert('na')
            d.gbar_na = model.gna_dend * dendNa
            d.insert('kv')
            d.gbar_kv = model.gkv_dend
            d.insert('km')
            d.gbar_km = model.gkm_dend
            d.insert('kca')
            d.gbar_kca = model.gkca_dend
            d.insert('ca')
            d.gbar_ca = model.gca_dend * dendCa
            d.insert('it')
            d.gbar_it = model.git_dend * dendCa
            d.insert('cad')
            d.ena = model.Ena
            d.ek = model.Ek
            d.eca = model.Eca

        print('active dendrites enabled', dendNa, model.gna_dend)


def init_passiveCA1(model):
    for sec in h.allsec():
        sec.e_pas = model.v_init


def init_activeCA1(model, soma=True, dend=True):
    if soma:
        model.soma.insert('nax')
        model.soma.gbar_nax = model.gna_soma
        model.soma.insert('kdr')
        model.soma.gkdrbar_kdr = model.gkdr_soma
        model.soma.insert('kap')
        model.soma.gkabar_kap = model.gka

        model.hill.insert('nax')
        model.hill.gbar_nax = model.gna_axon
        model.hill.insert('kdr')
        model.hill.gkdrbar_kdr = model.gkdr_axon
        model.soma.insert('kap')
        model.soma.gkabar_kap = model.gka

        model.iseg.insert('nax')
        model.iseg.gbar_nax = model.gna_axon
        model.iseg.insert('kdr')
        model.iseg.gkdrbar_kdr = model.gkdr_axon
        model.iseg.insert('kap')
        model.soma.gkabar_kap = model.gka

        model.node[0].insert('nax')
        model.node[0].gbar_nax = model.gna_node
        model.node[0].insert('kdr')
        model.node[0].gkdrbar_kdr = model.gkdr_axon
        model.node[0].insert('kap')
        model.node[0].gkabar_kap = model.gka * 0.2

        model.node[1].insert('nax')
        model.node[1].gbar_nax = model.gna_node
        model.node[1].insert('kdr')
        model.node[1].gkdrbar_kdr = model.gkdr_axon
        model.node[1].insert('kap')
        model.node[1].gkabar_kap = model.gka * 0.2

        model.inode[0].insert('nax')
        model.inode[0].gbar_nax = model.gna_axon
        model.inode[0].insert('kdr')
        model.inode[0].gkdrbar_kdr = model.gkdr_axon
        model.inode[0].insert('kap')
        model.inode[0].gkabar_kap = model.gka * 0.2

        model.inode[1].insert('nax')
        model.inode[1].gbar_nax = model.gna_axon
        model.inode[1].insert('kdr')
        model.inode[1].gkdrbar_kdr = model.gkdr_axon
        model.inode[1].insert('kap')
        model.inode[1].gkabar_kap = model.gka * 0.2

        model.inode[2].insert('nax')
        model.inode[2].gbar_nax = model.gna_axon
        model.inode[2].insert('kdr')
        model.inode[2].gkdrbar_kdr = model.gkdr_axon
        model.inode[2].insert('kap')
        model.inode[2].gkabar_kap = model.gka * 0.2

        print('somatic and axonal active conductances enabled')

    if dend:

        for d in model.dends:
            d.insert('nad')
            d.gbar_nad = model.gna
            d.insert('kdr')
            d.gkdrbar_kdr = model.gkdr
            d.insert('kap')
            d.gkabar_kap = 0
            d.insert('kad')
            d.gkabar_kad = 0

        h('access soma')
        h('distance()')

        # for the apicals: KA-type depends on distance
        # density is as in terminal branches - independent of the distance
        for sec in h.all_apicals:
            nseg = sec.nseg
            iseg = 0
            for seg in sec:
                xx = iseg * 1.0 / nseg + 1.0 / nseg / 2.0
                xdist = h.distance(xx, sec=sec)
                if xdist > model.dprox:
                    seg.gkabar_kad = model.gka
                else:
                    seg.gkabar_kap = model.gka
                iseg = iseg + 1

        h('access soma')
        h('distance()')

        # distance dependent A-channel densities in apical trunk dendrites
        #      1. densities increase till 'dlimit' with dslope
        #      2. proximal channels switch to distal at 'dprox'
        #      3. sodiom channel density also increases with distance
        for sec in h.primary_apical_list:
            nseg = sec.nseg
            sec.insert('nax')
            iseg = 0
            for seg in sec:
                # 0. calculate the distance from soma
                xx = iseg * 1.0 / nseg + 1.0 / nseg / 2.0
                xdist = h.distance(xx, sec=sec)
                # 1. densities increase till 'dlimit' with dslope
                if xdist > model.dlimit:
                    xdist = model.dlimit
                # 2. proximal channels switch to distal at 'dprox'
                if xdist > model.dprox:
                    seg.gkabar_kad = model.gka_trunk * (1 + xdist * model.dslope)
                else:
                    seg.gkabar_kap = model.gka_trunk * (1 + xdist * model.dslope)
                iseg = iseg + 1
                # 3. sodiom channel density also increases with distance
                if xdist > model.nalimit:
                    xdist = model.nalimit
                seg.gbar_nax = model.gna_trunk * (1 + xdist * model.naslope)
                # print(sec.name(), model.gna_trunk, xdist, model.naslope, seg.gbar_nax)
                seg.gbar_nad = 0
                seg.gkdrbar_kdr = model.gkdr_trunk

        # for the basals: all express proximal KA-type
        # density does not increase with the distance
        for sec in h.all_basals:
            for seg in sec:
                seg.gkabar_kap = model.gka

        # for sec in h.parent_list:
        #     for seg in sec:
        #         seg.gbar_nad = model.gna * 0
        #         sec.insert('nax')
        #         seg.gbar_nax = 30 * model.gna_trunk*(1+model.nalimit*model.naslope)

        print('active dendrites enabled')


def CA1_hotSpot(model, iden, gNa_factor=10):
    n_seg = model.dends[iden].nseg
    model.dends[iden].nseg = max(n_seg, 10)
    nmax = model.dends[iden].nseg

    # d = model.dends[iden]
    # d.insert('nafast'); d.gbar_nafast = 0

    s = 0
    min_spot = np.floor(2. * nmax / 6.)
    max_spot = np.floor(4. * nmax / 6.)
    for seg in model.dends[iden]:
        s += 1
        if ((s >= min_spot) & (s < max_spot)):
            seg.gbar_nad = model.gna * gNa_factor
            # seg.gbar_nafast = model.gna * gNa_factor
            # print s
    print('hotspots added')


def hotSpot(model):
    for section in model.dends:
        s = 0
        nmax = section.nseg
        min_spot = np.ceil(3. * nmax / 7.)
        max_spot = np.ceil(4. * nmax / 7.)
        for seg in section:
            if ((s > min_spot) & (s <= max_spot)):
                seg.gbar_na = model.gna_dend_hotSpot
                # seg.gbar_kv = model.gkv_dend_hotSpot
                # print section.name(), 'hotspot added', s
            else:
                seg.gbar_na = 0
                # seg.gbar_kv = 0
            s += 1
    print('hotspots added')


def add_somaStim(model, p=0.5, onset=20, dur=1, amp=0):
    model.stim = h.IClamp(model.soma(p))
    model.stim.delay = onset
    model.stim.dur = dur
    model.stim.amp = amp  # nA


def add_dendStim(model, p=0.5, dend=10, onset=20, dur=1, amp=0):
    model.stim = h.IClamp(model.dends[dend](p))
    model.stim.delay = onset
    model.stim.dur = dur
    model.stim.amp = amp  # nA


def add_dendStim4(model, dends, onset=20, dur=1, amp=0):
    model.stim1 = h.IClamp(model.dends[dends[0]](0.5))
    model.stim1.delay = onset
    model.stim1.dur = dur
    model.stim1.amp = (1) * amp  # nA

    model.stim2 = h.IClamp(model.dends[dends[1]](0.5))
    model.stim2.delay = onset
    model.stim2.dur = dur
    model.stim2.amp = (1) * amp  # nA

    model.stim3 = h.IClamp(model.dends[dends[2]](0.5))
    model.stim3.delay = onset
    model.stim3.dur = dur
    model.stim3.amp = (1) * amp  # nA

    model.stim4 = h.IClamp(model.dends[dends[3]](0.5))
    model.stim4.delay = onset
    model.stim4.dur = dur
    model.stim4.amp = (1) * amp  # nA


def synDist(model, locs):
    nsyn = len(locs)
    DSyn = np.zeros([nsyn, nsyn])
    fromSyn = 0
    for loc in locs:
        fromDend = loc[0]
        fromX = loc[1]
        fromSection = model.dends[fromDend]
        h.distance(0, fromX, sec=fromSection)

        toSyn = 0
        for toLoc in locs:
            toDend = toLoc[0]
            toX = toLoc[1]
            toSection = model.dends[toDend]
            x = h.distance(toX, sec=toSection)
            DSyn[toSyn, fromSyn] = x
            toSyn = toSyn + 1
        fromSyn = fromSyn + 1
    return DSyn


def surface_area(model):
    sa = model.soma.diam * model.soma.L * np.pi
    ndends = len(model.dends)
    for i in range(ndends):
        sa = sa + model.dends[i].diam * model.dends[i].L * np.pi
        print('dendrite', model.dends[i].name(), 'diam', model.dends[i].diam, 'length', model.dends[i].L)
    return sa


def single_pulse_NetStim():
    """Sets single pulse NetStim."""
    pulse = h.NetStim()
    pulse.interval = 1
    pulse.number = 1
    pulse.start = -1e9
    pulse.noise = 0
    return pulse


def add_syns(model, data):
    print('adding synapses using the new function!')
    model.AMPAlist = []
    model.ncAMPAlist = []
    model.AMPA_NS_terminals = []
    model.NMDA_NS_terminals = []
    model.input_AMPA_spikes = []
    AMPA_gmax = data.Agmax / 1000.  # Set in nS and convert to muS

    if data.SPINES:
        # h.nspines = len(data.Elocs)
        h('nspines = 0')
        h.nspines = len(data.Elocs)
        print('we will create', h.nspines, 'dendritic spines for excitatory synapses')
        h('create shead[nspines]')
        h('create sneck[nspines]')

    if (data.NMDA):
        model.NMDAlist = []
        model.ncNMDAlist = []
        NMDA_gmax = data.Ngmax / 1000.  # Set in nS and convert to muS

    spi = 0  # indexing the spines - in hoc

    print(data.Elocs)
    for loc in data.Elocs:
        locInd = int(loc[0])
        if locInd == -1:
            synloc = model.soma
        else:
            if data.SPINES:

                neck = h.sneck[spi]
                neck.L = data.sneck_len
                neck.diam = data.sneck_diam
                neck.insert("pas")
                neck.e_pas = model.E_PAS
                neck.g_pas = 1.0 / model.RM
                neck.Ra = model.RA
                neck.cm = model.CM

                head = h.shead[spi]
                head.L = data.shead_len
                head.diam = data.shead_diam
                head.insert("pas")
                head.e_pas = model.E_PAS
                head.g_pas = 1.0 / model.RM
                head.Ra = model.RA
                head.cm = model.CM

                head.connect(neck, 1, 0)
                neck.connect(model.dends[int(loc[0])], loc[1], 0)
                synloc = h.shead[spi]
                synpos = 0.5
                spi = spi + 1
            else:
                synloc = model.dends[int(loc[0])]
                synpos = float(loc[1])
        # print loc[0], loc[1]
        AMPA = h.Exp2Syn(synpos, sec=synloc)
        AMPA.tau1 = data.Atau1
        AMPA.tau2 = data.Atau2

        NS_AMPA = h.NetStim()
        NS_AMPA.interval = 1
        NS_AMPA.number = 1
        NS_AMPA.start = -1e9
        NS_AMPA.noise = 0

        NC = h.NetCon(NS_AMPA, AMPA, 0, 0, AMPA_gmax)  # NetCon(source, target, threshold, delay, weight)
        input_spikes = h.Vector()
        NC.record(input_spikes)
        model.AMPAlist.append(AMPA)
        model.ncAMPAlist.append(NC)
        model.AMPA_NS_terminals.append(NS_AMPA)
        model.input_AMPA_spikes.append(input_spikes)

        if data.NMDA:
            if data.constNMDA:
                NMDA = h.Exp2SynNMDAv0(synpos, sec=synloc)
            else:
                NMDA = h.Exp2SynNMDA(synpos, sec=synloc)
            NMDA.tau1 = data.Ntau1
            NMDA.tau2 = data.Ntau2

            NS_NMDA = h.NetStim()
            NS_NMDA.interval = 1
            NS_NMDA.number = 1
            NS_NMDA.start = -1e9
            NS_NMDA.noise = 0

            NC = h.NetCon(NS_NMDA, NMDA, 0, 0, NMDA_gmax)
            x = float(loc[1])
            model.NMDAlist.append(NMDA)
            model.ncNMDAlist.append(NC)
            model.NMDA_NS_terminals.append(NS_NMDA)
    print('AMPA synapses added')
    if data.NMDA:
        print('dExp NMDA synapses generated')

    if data.GABA:
        model.GABAlist = []
        model.ncGABAlist = []
        GABA_gmax = data.Igmax / 1000.  # Set in nS and convert to muS

        if (data.GABA_B):
            model.GABA_Blist = []
            model.ncGABA_Blist = []
            GABAB_gmax = data.Bgmax / 1000.  # Set in nS and convert to muS

        for loc in data.Ilocs:
            locInd = int(loc[0])
            if (locInd == -1):
                synloc = model.soma
            else:
                synloc = model.dends[int(loc[0])]
            GABA = h.Exp2Syn(float(loc[1]), sec=synloc)
            GABA.tau1 = data.Itau1
            GABA.tau2 = data.Itau2
            GABA.e = data.Irev
            NC = h.NetCon(h.nil, GABA, 0, 0, GABA_gmax)
            model.GABAlist.append(GABA)
            model.ncGABAlist.append(NC)

            if (data.GABA_B):
                GABAB = h.Exp2Syn(float(loc[1]), sec=synloc)
                GABAB.tau1 = data.Btau1
                GABAB.tau2 = data.Btau2
                GABAB.e = data.Brev
                NC = h.NetCon(h.nil, GABAB, 0, 0, GABAB_gmax)
                model.GABA_Blist.append(GABAB)
                model.ncGABA_Blist.append(NC)

        print('inhibitory synapses generated')
        if (data.GABA_B):
            print('GABA_B synapses generated')

# ----------------------------------------------------------
# SIMULATION RUN

def simulate(model, t_stop=100, NMDA=False, recDend=False, i_recDend=11, x_recDend=0.5, spines=False, i_recSpine=11,
             recGDend=False, i_recSyn=[0, 1, 2, 3, ]):
    trec, vrec = h.Vector(), h.Vector()
    gRec, iRec, vDendRec, gDendRec, iDendRec = [], [], [], [], []
    # gNMDA_rec, iNMDA_rec = [], []
    trec.record(h._ref_t)
    vrec.record(model.soma(0.5)._ref_v)

    # if NMDA:
    #     for n in np.arange(0, len(model.NMDAlist)):
    #         loc = model.NMDAlist[n].get_loc()
    #         h.pop_section()
    #         gNMDA_rec.append(h.Vector())
    #         iNMDA_rec.append(h.Vector())
    #         gNMDA_rec[n].record(model.NMDAlist[n]._ref_g)
    #         iNMDA_rec[n].record(model.NMDAlist[n]._ref_i)
    #     gRec.append(gNMDA_rec)
    #     iRec.append(iNMDA_rec)
    if recDend:
        n = 0
        for i_dend in i_recDend:
            vDendRec.append(h.Vector())
            x_dend = x_recDend[n]
            vDendRec[n].record(model.dends[i_dend](x_dend)._ref_v)
            n += 1

        # vDendRec.append(h.Vector())
        # vDendRec[n].record(model.node[1](0.5)._ref_v)
        # n+=1

        if spines:
            for i_spine in i_recSpine:
                vDendRec.append(h.Vector())
                vDendRec[n].record(h.shead[i_spine](0.5)._ref_v)
                n += 1

    if recGDend:
        # # print 'recording conductances'
        # gDendRec.append(h.Vector())
        # x_dend = x_recDend[0]
        # # gDendRec[0].record(model.dends[i_recDend[0]](x_dend)._ref_thegna_nad)
        # gDendRec[0].record(model.dends[i_recDend[0]](x_dend)._ref_gna_na)

        # gDendRec.append(h.Vector())
        # # gDendRec[1].record(model.dends[i_recDend[0]](x_dend)._ref_gkdr_kdr)
        # gDendRec[1].record(model.dends[i_recDend[0]](x_dend)._ref_gk_kv)

        # gDendRec.append(h.Vector())
        # # gDendRec[2].record(model.dends[i_recDend[0]](x_dend)._ref_gka_kap)
        # gDendRec[2].record(model.dends[i_recDend[0]](x_dend)._ref_gk_km)

        # gDendRec.append(h.Vector())
        # # gDendRec[3].record(model.dends[i_recDend[0]](x_dend)._ref_gka_kad)
        # gDendRec[3].record(model.dends[i_recDend[0]](x_dend)._ref_gk_kca)

        # gDendRec.append(h.Vector())
        # # gDendRec[4].record(model.AMPAlist[9]._ref_g)
        # gDendRec[4].record(model.AMPAlist[19]._ref_g)

        gDendRec.append(h.Vector())
        gDendRec[0].record(model.NMDAlist[i_recSyn[0]]._ref_g)

        gDendRec.append(h.Vector())
        gDendRec[1].record(model.NMDAlist[i_recSyn[1]]._ref_g)

        gDendRec.append(h.Vector())
        gDendRec[2].record(model.NMDAlist[i_recSyn[2]]._ref_g)

        gDendRec.append(h.Vector())
        gDendRec[3].record(model.NMDAlist[i_recSyn[3]]._ref_g)

        iDendRec.append(h.Vector())
        iDendRec[0].record(model.NMDAlist[i_recSyn[0]]._ref_i)

        iDendRec.append(h.Vector())
        iDendRec[1].record(model.NMDAlist[i_recSyn[1]]._ref_i)

        iDendRec.append(h.Vector())
        iDendRec[2].record(model.NMDAlist[i_recSyn[2]]._ref_i)

        iDendRec.append(h.Vector())
        iDendRec[3].record(model.NMDAlist[i_recSyn[3]]._ref_i)

    h.celsius = model.CELSIUS
    h.finitialize(model.v_init)
    neuron.run(t_stop)
    return np.array(trec), np.array(vrec), np.array(vDendRec), np.array(gDendRec), np.array(
        iDendRec)  # , np.array(caDendRec), np.array(vSecRec)
