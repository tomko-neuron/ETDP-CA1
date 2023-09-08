"""
Title: stimulation_protocols.py
Author: Matus Tomko
Mail: matus.tomko __at__ fmph.uniba.sk
"""
import numpy as np
from neuron import h, gui
from CA1_plasticity.model.utils import Synapse


class StimulationProtocol:
    """
    A class used to set a stimulation protocol

    ...

    Attributes
    ----------
    setting : dict
        the setting dictionary
    net_cons : list
        the list of neuron.hoc.NetCons
    ppStims : list
        the list of neuron.hoc.spGen2s
    vec_stims : list
        the list of neuron.hoc.VecStims
    iclamp_t_vec : neuron.hoc.Vector
        the time vector for time varying current stimuli
    iclamp_amps_vec : neuron.hoc.Vector
        the vector of amplitudes for time varying current stimuli

    Methods
    -------
    create_VecStim(t_vec, synapse)
        Creates a vector of stimulus times.
    set_Dong_sequential_stimulation(synapses)
        Sets the sequential stimulation protocol using Vecstim objects for Dong et al. experiments.
    set_Pavlowsky_Alarcon_HFS(synapses)
        Sets the HFS stimulation protocol using Vecstim objects for Pavlowsky & Alarcon experiments.
    set_Pavlowsky_Alarcon_LFS(synapses)
        Sets the LFS stimulation protocol using Vecstim objects for Pavlowsky & Alarcon experiments.
    set_Pavlowsky_Alarcon_PP(synapses)
        Sets the paired-pulses stimulation protocol using Vecstim objects for Pavlowsky & Alarcon experiments.
    set_ppStim(synapses)
        Sets the paired-pulses stimulation protocol using spGen2 objects.
    set_square_pulse(synapses)
        Sets the square pulse stimulation protocol using Vecstim objects.
    set_theta_burst(synapses)
        Sets the theta burst stimulation protocol using Vecstim objects.
    set_theta_burst_iclamp(stim)
        Sets times for current injections for the theta burst pairing stimulation protocol.
    set_SEClamp(stim)
        Sets parameters for a single electrode voltage clamp with three levels
    set_Makara_LTP(synapses, num_synapses)
        Sets the LTP induction protocol according to the Makara et al.(2020).
    set_Makara_LTP_prestim_test_activity(synapses, num_synapses)
        Sets the preLTP test activity according to the Makara et al.(2020).
    set_Makara_LTP_poststim_test_activity(self, synapses, num_synapses)
        Sets the postLTP test activity according to the Makara et al.(2020).
    """

    def __init__(self, setting):
        """
        Parameters
        ----------
        setting : dict
            the dictionary containing setting
        """
        self.setting = setting
        self.net_cons = []
        self.ppStims = []
        self.vec_stims = []
        self.iclamp_t_vec = h.Vector()
        self.iclamp_amps_vec = h.Vector()

    def create_VecStim(self, t_vec, synapse):
        """
        Creates a vector stream of events for given synapse.

        Parameters
        ----------
        t_vec : numpy.ndarray
            the time vector
        synapse : Synapse
            synapse
        """
        vec = h.Vector(t_vec)
        vec_stim = h.VecStim()
        vec_stim.play(vec)
        nc = h.NetCon(vec_stim, synapse.ns_terminal, 0, 0, 1)
        self.net_cons.append(nc)
        self.vec_stims.append(vec_stim)


    def set_Dong_sequential_stimulation(self, synapses):
        """
        Sets the sequential stimulation protocol using Vecstim objects for Dong et al. experiments.

        Parameters
        ----------
        synapses : dict
            the dictionary containing synapses
        """
        for sec in synapses:
            for syn in synapses[sec]:
                if np.random.rand() < self.setting['protocol']['Dong_SSt']['DONG_STIMULATED_PERC']:
                    if syn.pathway == 'SCH':
                        t_start = self.setting['protocol']['Dong_SSt']['DONG_SCH_START'] + np.random.rand()
                    elif syn.pathway == 'COM':
                        t_start = self.setting['protocol']['Dong_SSt']['DONG_COM_START'] + np.random.rand()
                    else:
                        continue
                    t_stop = t_start + self.setting['protocol']['Dong_SSt']['DONG_PULSES_NUM'] * \
                             self.setting['protocol']['Dong_SSt']['DONG_INTERPULSE_INTERVAL']
                    t_vec = np.arange(t_start, t_stop, self.setting['protocol']['Dong_SSt']['DONG_INTERPULSE_INTERVAL'])
                    self.create_VecStim(t_vec=t_vec,
                                        synapse=syn)
                    syn.stimulated = True
                else:
                    continue

    def set_Pavlowsky_Alarcon_HFS(self, synapses):
        """
        Sets the HFS stimulation protocol using Vecstim objects for Pavlowsky & Alarcon experiments.

        Parameters
        ----------
        synapses : dict
            the dictionary containing synapses
        """
        for sec in synapses:
            s = 0
            while s < len(synapses[sec]):
                # for syn in synapses[sec]:
                if np.random.rand() < self.setting['protocol']['Pavlowsky_Alarcon']['HFS_STIMULATED_PERC']:
                    t_start = self.setting['protocol']['Pavlowsky_Alarcon']['HFS_START'] + np.random.rand()
                    t_vec = np.zeros(0)
                    for i in range(self.setting['protocol']['Pavlowsky_Alarcon']['HFS_TRAINS_NUM']):
                        vec = np.arange(t_start, t_start + 1000, 10)
                        t_vec = np.concatenate((t_vec, vec), axis=0)
                        t_start = t_start + 1000 + self.setting['protocol']['Pavlowsky_Alarcon']['HFS_INTERTRAIN']
                    self.create_VecStim(t_vec=t_vec, synapse=synapses[sec][s])
                    self.create_VecStim(t_vec=t_vec, synapse=synapses[sec][s + 1])
                    synapses[sec][s].stimulated = True
                    synapses[sec][s + 1].stimulated = True
                    s = s + 2
                else:
                    s = s + 2

    def set_Pavlowsky_Alarcon_LFS(self, synapses):
        """
        Sets the LFS stimulation protocol using Vecstim objects for Pavlowsky & Alarcon experiments.

        Parameters
        ----------
        synapses : dict
            the dictionary containing synapses
        """
        for sec in synapses:
            s = 0
            while s < len(synapses[sec]):
                # for syn in synapses[sec]:
                if np.random.rand() < self.setting['protocol']['Pavlowsky_Alarcon']['LFS_STIMULATED_PERC']:
                    t_start = self.setting['protocol']['Pavlowsky_Alarcon']['LFS_START'] + np.random.rand()
                    t_vec = np.zeros(0)
                    vec = np.arange(t_start,
                                    t_start + self.setting['protocol']['Pavlowsky_Alarcon']['LFS_STIM_LEN'],
                                    1000)
                    t_vec = np.concatenate((t_vec, vec), axis=0)
                    self.create_VecStim(t_vec=t_vec, synapse=synapses[sec][s])
                    self.create_VecStim(t_vec=t_vec, synapse=synapses[sec][s + 1])
                    synapses[sec][s].stimulated = True
                    synapses[sec][s + 1].stimulated = True
                    s = s + 2
                else:
                    s = s + 2

    def set_Pavlowsky_Alarcon_PP(self, synapses):
        """
        Sets the paired-pulses stimulation protocol using Vecstim objects for Pavlowsky & Alarcon experiments.

        Parameters
        ----------
        synapses : dict
            the dictionary containing synapses
        """
        for sec in synapses:
            s = 0
            while s < len(synapses[sec]):
                # for syn in synapses[sec]:
                if np.random.rand() < self.setting['protocol']['Pavlowsky_Alarcon']['PP_STIMULATED_PERC']:
                    t_start = self.setting['protocol']['Pavlowsky_Alarcon']['PP_START'] + np.random.rand()
                    t_vec = np.zeros(0)
                    for i in range(self.setting['protocol']['Pavlowsky_Alarcon']['PP_NUM']):
                        vec = [t_start, t_start + 50]
                        t_vec = np.concatenate((t_vec, vec), axis=0)
                        t_start = t_start + 1000
                    print(t_vec)
                    self.create_VecStim(t_vec=t_vec, synapse=synapses[sec][s])
                    self.create_VecStim(t_vec=t_vec, synapse=synapses[sec][s + 1])
                    synapses[sec][s].stimulated = True
                    synapses[sec][s + 1].stimulated = True
                    s = s + 2
                else:
                    s = s + 2

    def set_ppStim(self, synapses):
        """
        Sets the paired-pulses stimulation protocol using spGen2 objects.

        Parameters
        ----------
        synapses : dict
            the dictionary containing synapses
        """
        for sec in synapses:
            for syn in synapses[sec]:
                ppStim = h.SpGen2(0.5)
                ppStim.APinburst = self.setting['protocol']['theta_burst']['PP_STIM_APINBURST']
                ppStim.t01 = self.setting['protocol']['theta_burst']['PP_STIM_T01']
                self.ppStims.append(ppStim)
                if syn.receptor == 'AMPA':
                    nc = h.NetCon(ppStim, syn.synapse, 0, 0, self.setting['protocol']['theta_burst']['PP_WEIGHT'])
                    self.net_cons.append(nc)
                    syn.weight_vec.record(nc._ref_weight[1], self.setting['simulation']['RECORDING_STEP'])
                    nc.record(syn.input_spikes)
                elif syn.receptor == 'NMDA':
                    nc = h.NetCon(ppStim, syn.synapse, 0, 0, self.setting['protocol']['theta_burst']['PP_WEIGHT'])
                    self.net_cons.append(nc)
                    nc.record(syn.input_spikes)
                syn.stimulated = True

    def set_square_pulse(self, synapses):
        """
        Sets the square pulse stimulation protocol using Vecstim objects.

        Parameters
        ----------
        synapses : dict
            the dictionary containing synapses
        """
        for sec in synapses:
            for syn in synapses[sec]:
                t_start = self.setting['protocol']['square_pulses']['SQ_START'] + np.random.rand()
                t_vec = np.zeros(0)

                for i in range(self.setting['protocol']['square_pulses']['SQ_PULSES_NUM']):
                    pulse_vec = np.arange(t_start,
                                          self.setting['protocol']['square_pulses']['SQ_INTERSPIKE_INTERVAL'] *
                                          self.setting['protocol']['square_pulses']['SQ_STIMULI_NUM'] + t_start,
                                          self.setting['protocol']['square_pulses']['SQ_INTERSPIKE_INTERVAL'])
                    t_vec = np.concatenate((t_vec, pulse_vec), axis=0)
                    t_start = t_start + self.setting['protocol']['square_pulses']['SQ_INTERSPIKE_INTERVAL'] * \
                              self.setting['protocol']['square_pulses']['SQ_STIMULI_NUM'] + \
                              self.setting['protocol']['square_pulses']['SQ_INTERPULSE_INTERVAL']

                self.create_VecStim(t_vec=t_vec, synapse=syn)
                syn.stimulated = True

    def set_theta_burst(self, synapses):
        """
        Sets the theta burst stimulation protocol using Vecstim objects.

        Parameters
        ----------
        synapses : dict
            the dictionary containing synapses
        """
        for sec in synapses:
            s = 0
            while s < len(synapses[sec]):
                if np.random.rand() < self.setting['protocol']['theta_burst']['TB_STIMULATED_PERC']:
                    t_start = self.setting['protocol']['theta_burst']['TB_START']#  + np.random.rand()
                    t_vec = np.zeros(0)
                    for pattern in range(self.setting['protocol']['theta_burst']['TB_PATTERNS_NUM']):
                        for burst in range(self.setting['protocol']['theta_burst']['TB_BURSTS_NUM']):
                            t_stop = t_start + 1 + (
                                    self.setting['protocol']['theta_burst']['TB_STIMULI_NUM'] - 1) * \
                                     self.setting['protocol']['theta_burst']['TB_INTERSPIKE_INTERVAL']
                            burst_vec = np.arange(t_start,
                                                  t_stop,
                                                  self.setting['protocol']['theta_burst']['TB_INTERSPIKE_INTERVAL'])
                            t_vec = np.concatenate((t_vec, burst_vec), axis=0)
                            # t_start = t_vec[-1] + self.setting['protocol']['theta_burst_pairing']['TBP_INTERBURST_INTERVAL']
                            t_start = t_start + self.setting['protocol']['theta_burst']['TB_INTERBURST_INTERVAL']
                        t_start = t_start + self.setting['protocol']['theta_burst']['TB_PATTERNS_INTERVAL']
                    self.create_VecStim(t_vec=t_vec, synapse=synapses[sec][s])
                    self.create_VecStim(t_vec=t_vec, synapse=synapses[sec][s + 1])
                    synapses[sec][s].stimulated = True
                    synapses[sec][s + 1].stimulated = True
                    s = s + 2
                else:
                    s = s + 2

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

    def set_SEClamp(self, stim):
        """
        Sets parameters for a single electrode voltage clamp with three levels.

        Parameters
        ----------
        stim : neuron.hoc.HocObject
            the object of a single electrode voltage clamp point process
        """
        stim.dur1 = self.setting['protocol']['SEClamp']['DUR1']
        stim.amp1 = self.setting['protocol']['SEClamp']['AMP1']

    def set_Makara_LTP(self, synapses, num_synapses):
        """
        Sets the LTP induction protocol according to the Makara et al.(2020).

        Parameters
        ----------
        synapses : dict
            the dictionary containing synapses
        num_synapses : int
            the number of stimulated synapses
        """
        for sec in synapses:
            s = 0
            t = 0
            while s < len(synapses[sec]) and s < num_synapses * 2:
                t_start = self.setting['protocol']['Makara_LTP']['LTP_START'] + (t * 0.1)
                t_stop = t_start + 1 + (
                        self.setting['protocol']['Makara_LTP']['LTP_STIMULI_NUM'] - 1) * \
                        self.setting['protocol']['Makara_LTP']['LTP_INTERVAL']
                t_vec = np.arange(t_start, t_stop, self.setting['protocol']['Makara_LTP']['LTP_INTERVAL'])
                self.create_VecStim(t_vec=t_vec, synapse=synapses[sec][s])
                self.create_VecStim(t_vec=t_vec, synapse=synapses[sec][s + 1])
                synapses[sec][s].stimulated = True
                synapses[sec][s + 1].stimulated = True
                s = s + 2
                t = t + 1

    def set_Makara_LTP_prestim_test_activity(self, synapses, num_synapses):
        """
        Sets the preLTP test activity according to the Makara et al.(2020).

        Parameters
        ----------
        synapses : dict
            the dictionary containing synapses
        num_synapses : int
            the number of stimulated synapses
        """
        for sec in synapses:
            s = 0
            t = 0
            while s < len(synapses[sec]) and s < num_synapses * 2:
                t_start = self.setting['protocol']['Makara_LTP']['PRELTP_TEST_START'] + (t * 200)
                t_stop = t_start + 1 + (
                        self.setting['protocol']['Makara_LTP']['PRELTP_TEST_NUM'] - 1) * \
                         self.setting['protocol']['Makara_LTP']['PRELTP_TEST_INTERVAL']
                t_vec = np.arange(t_start, t_stop, self.setting['protocol']['Makara_LTP']['PRELTP_TEST_INTERVAL'])
                self.create_VecStim(t_vec=t_vec, synapse=synapses[sec][s])
                self.create_VecStim(t_vec=t_vec, synapse=synapses[sec][s + 1])
                s = s + 2
                t = t + 1
                if t == num_synapses:
                    return

    def set_Makara_LTP_poststim_test_activity(self, synapses, num_synapses):
        """
        Sets the postLTP test activity according to the Makara et al.(2020).

        Parameters
        ----------
        synapses : dict
            the dictionary containing synapses
        num_synapses : int
            the number of stimulated synapses
        """
        for sec in synapses:
            s = 0
            t = 0
            while s < len(synapses[sec]) and s < num_synapses * 2:
                t_start = self.setting['protocol']['Makara_LTP']['POSTLTP_TEST_START'] + (t * 200)
                t_stop = t_start + 1 + (
                        self.setting['protocol']['Makara_LTP']['POSTLTP_TEST_NUM'] - 1) * \
                         self.setting['protocol']['Makara_LTP']['POSTLTP_TEST_INTERVAL']
                t_vec = np.arange(t_start, t_stop, self.setting['protocol']['Makara_LTP']['POSTLTP_TEST_INTERVAL'])
                self.create_VecStim(t_vec=t_vec, synapse=synapses[sec][s])
                self.create_VecStim(t_vec=t_vec, synapse=synapses[sec][s + 1])
                s = s + 2
                t = t + 1
