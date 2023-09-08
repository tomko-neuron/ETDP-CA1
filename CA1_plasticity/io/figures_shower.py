"""
Title: figure_shower.py
Author: Matus Tomko
Mail: matus.tomko __at__ fmph.uniba.sk
"""

import efel
import gzip
import multiprocessing
import os
import pickle

import matplotlib.patches as mpatches
import numpy as np
import scipy.stats as stats
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing


def calculate_average_weights(data, secs, key, synapse_type, return_dict):
    """
    Calculates average weights.

    Parameters
    ----------
    data : dict
        the dictionary containing the data
    secs : list
        the list of sections names
    key : str
        the key for a return dictionary
    synapse_type : str
        the synapse type (perforated, nonperforated)
    return_dict : dict
        the shared dictionary
    """
    stim_weights = []
    unstim_weights = []
    for sec in data['synapses']:
        if sec in secs:
            for syn in data['synapses'][sec]:
                if syn['receptor'] == 'AMPA' and syn['type'] == synapse_type:
                    if syn['stimulated']:
                        stim_weights.append(syn['weight'])
                    else:
                        unstim_weights.append(syn['weight'])

    if len(stim_weights) > 0:
        stim_weights = np.array(stim_weights)
        avg_stim_weights = np.mean(stim_weights, axis=0)
        return_dict[key + ' stim'] = avg_stim_weights

    if len(unstim_weights) > 0:
        unstim_weights = np.array(unstim_weights)
        avg_unstim_weights = np.mean(unstim_weights, axis=0)
        return_dict[key + ' unstim'] = avg_unstim_weights

    return_dict['T'] = data['T']


def calculate_average_SCH_COM_weights(data, secs, layer, pathway, return_dict):
    """
    Calculates average weights.

    Parameters
    ----------
    data : dict
        the dictionary containing the data
    secs : list
        the list of sections names
    layer : str
        the layer (ori, rad, lm)
    pathway : str
        the pathway (SCH, COM, LM)
    return_dict : dict
        the shared dictionary
    """
    weights = []
    for sec in secs:
        for syn in data['synapses'][sec]:
            if syn['pathway'] == pathway:
                weights.append(syn['weight'])

    weights = np.array(weights)
    av_weights = np.mean(weights, axis=0)
    if pathway is None:
        return_dict[layer] = av_weights
    else:
        return_dict[layer + '_' + pathway] = av_weights
    return_dict['T'] = data['T']


class FiguresShower:
    """
    A class used to show figures

    ...

    Attributes
    ----------
    path_recordings : str
        path to the directory containing recordings
    path_saving : str
        the path to the directory where the figures will be saved
    setting : dict
        the dictionary containing setting
    save_figures : bool
        the flag indicates whether the figures will be saved (default False)

    COM : Patch
    LM : Patch
    SCH : Patch
    ori_stim : Patch
    ori_unstim : Patch
    rad_stim : Patch
    rad_unstim : Patch
    lm_stim : Patch
    lm_unstim : Patch

    ori_secs : list
    rad_secs : list
    lm_secs : list

    Methods
    -------
    show_alpha_scount()
        Shows integrated spike count scaled by alpha.
    show_d_p_amplitude()
        Shows depression and potentiation amplitudes.
    show_cal_ica()
        Shows CaL channel-mediated calcium current traces for each section.
    show_intracellular_calcium()
        Shows intracellular calcium concentration traces for each section.
    show_na_current()
        Shows Na current traces for each section.
    show_nmda_ica_current()
        Shows NMDA channel-mediated calcium current traces for each section.
    show_average_weights(secs, keys)
        Shows average weights for given sections.
    show_average_weights_change( secs, keys, baseline)
        Shows evolution of average weights in time.
    show_IO_characteristics()
        Shows average input and output firing frequencies.
    show_SCH_COM_average_weights()
        Shows average Schaffer and commissural weights.
    show_SCH_COM_average_weights_change(baseline)
        Shows evolution of average Schaffer and commissural weights in time.
    show_SCH_COM_weights()
        Shows synaptic weights for each section labeled as Schaffer, commissural or perforant.
    show_weights(synapse_type='perforated')
        Shows synaptic weights for each section.
    show_amplitudes(synapse_type='perforated'):
        Shows amplitudes for each section
    show_weights_distance(start=0, stop=0, synapse_type)
        Shows the percentage change for each synapse between the weight at start time and the weight at stop time
        as a scatter plot. X-axis represents distance of a synapse from the soma.
    show_weights_histogram(t=0, sections='all', synapse_type='all')
        Shows the histogram of synaptic weights in the time t.
    show_dendritic_voltage()
        Shows dendritic voltage for each section.
    show_voltage_on_spines()
        Shows voltage on spines.
    show_input_spikes()
        Shows input spikes.
    show_somatic_voltage()
        Shows the somatic voltage.
    """
    def __init__(self, path_recordings, path_saving, setting, save_figures=False):
        """
        Parameters
        ----------
        path_recordings : str
            path to the directory containing recordings
        path_saving : str
            the path to the directory where the figures will be saved
        setting : dict
            the dictionary containing setting
        save_figures : bool
            the flag indicates whether the figures will be saved (default False)
        """
        self.path_recordings = path_recordings
        self.path_saving = path_saving
        self.setting = setting
        self.save_figures = save_figures

        self.COM = mpatches.Patch(color='blue', label='Commissural')
        self.LM = mpatches.Patch(color='green', label='Perforant')
        self.SCH = mpatches.Patch(color='red', label='Schaffer')
        self.ori_stim = mpatches.Patch(color='tomato', label='ori stim')
        self.ori_unstim = mpatches.Patch(color='salmon', label='ori unstim')
        self.rad_stim = mpatches.Patch(color='navy', label='rad stim')
        self.rad_unstim = mpatches.Patch(color='mediumblue', label='rad unstim')
        self.lm_stim = mpatches.Patch(color='forestgreen', label='lm stim')
        self.lm_unstim = mpatches.Patch(color='limegreen', label='lm unstim')

        self.ori_secs = ['oriprox1', 'oriprox2', 'oridist1_1', 'oridist1_2', 'oridist2_1', 'oridist2_2']
        self.rad_secs = ['radTprox1', 'radTprox2', 'radTmed1', 'radTmed2', 'radTdist1', 'radTdist2',
                         'rad_t1', 'rad_t3', 'rad_t2']
        self.lm_secs = ['lm_thick1', 'lm_medium1', 'lm_thin1', 'lm_thick2', 'lm_medium2', 'lm_thin2']

        try:
            if not os.path.exists(self.path_saving):
                os.makedirs(self.path_saving)
        except OSError as e:
            if e.errno != 17:
                raise
            pass

        plt.rc('font', family='sans-serif')
        plt.rc('xtick', labelsize='x-small')
        plt.rc('ytick', labelsize='x-small')

    def show_alpha_scount(self):
        """Shows integrated spike count scaled by alpha."""
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'bcm.p', 'rb'))
        print('Final ISC: ' + str(data['alpha_scount'][-1]))
        fig = plt.figure(figsize=[15, 4])
        plt.plot(data['T_BCM'], data['alpha_scount'])
        plt.xlabel('Time (ms)')
        plt.ylabel('<c> * alpha')
        fig.tight_layout()
        if self.save_figures:
            plt.savefig(self.path_saving + 'alpha_scount' + '.svg', format='svg')
            print('The Integrated spike count scaled by alpha was saved in the directory: ' + self.path_saving)
        plt.show()
        plt.close()

    def show_d_p_amplitude(self):
        """Shows depression and potentiation amplitudes."""
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'bcm.p', 'rb'))
        print('Final depression amplitude: ' + str(data['D_amp'][-1]))
        print('Final potentiation amplitude: ' + str(data['P_amp'][-1]))
        fig = plt.figure(figsize=[15, 4])
        plt.plot(data['T_BCM'], data['D_amp'], label='Depression amplitude')
        plt.plot(data['T_BCM'], data['P_amp'], label='Potentiation amplitude')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        fig.tight_layout()
        if self.save_figures:
            plt.savefig(self.path_saving + 'amplitudes' + '.svg', format='svg')
            print('The Amplitudes were saved in the directory: ' + self.path_saving)
        plt.show()
        plt.close()

    def show_cal_ica(self):
        """Shows CaL channel-mediated calcium current traces for each section."""
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'currents.p', 'rb'))
        vecs = data['cal_ica']
        for sec in vecs:
            fig = plt.figure(figsize=[12, 4])
            for vec in vecs[sec]:
                plt.plot(data['T'], vec['vector'], label=sec + '(' + str(vec['segment_x']) + ')')
            plt.xlabel('Time (ms)')
            plt.ylabel('I(Ca Cal) (nA)')
            lgd = plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
            fig.tight_layout()
            if self.save_figures:
                plt.savefig(self.path_saving + 'cal_ica_' + sec + '.svg', format='svg',
                            bbox_extra_artists=(lgd,), bbox_inches='tight')
                print('The Ca current was saved in the directory: ' + self.path_saving)
            plt.show()
            plt.close()

    def show_intracellular_calcium(self):
        """Shows intracellular calcium concentration traces for each section."""
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'currents.p', 'rb'))
        vecs = data['cai']
        for sec in vecs:
            fig = plt.figure(figsize=[12, 4])
            for vec in vecs[sec]:
                plt.plot(data['T'], vec['vector'], label=sec + '(' + str(vec['segment_x']) + ')')
            plt.xlabel('Time (ms)')
            plt.ylabel('Intracellular Ca concentration (uM)')
            lgd = plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
            fig.tight_layout()
            if self.save_figures:
                plt.savefig(self.path_saving + 'cai_' + sec + '.svg', format='svg',
                            bbox_extra_artists=(lgd,), bbox_inches='tight')
                print('The free calcium was saved in the directory: ' + self.path_saving)
            plt.show()
            plt.close()

    def show_calcium_pump(self):
        """Shows calcium pump current traces for each section."""
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'currents.p', 'rb'))
        vecs = data['pmp']
        for sec in vecs:
            fig = plt.figure(figsize=[12, 4])
            for vec in vecs[sec]:
                plt.plot(data['T'], vec['vector'], label=sec + '(' + str(vec['segment_x']) + ')')
            plt.xlabel('Time (ms)')
            plt.ylabel('Calcium pump current (mA/cm2)')
            lgd = plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
            fig.tight_layout()
            if self.save_figures:
                plt.savefig(self.path_saving + 'cdp_pmp_' + sec + '.svg', format='svg',
                            bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.show()
            plt.close()

    def show_calcium_indicator(self):
        """Shows calcium indicator traces for each section."""
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'currents.p', 'rb'))
        vecs = data['ogb']
        for sec in vecs:
            fig = plt.figure(figsize=[12, 4])
            for vec in vecs[sec]:
                plt.plot(data['T'], vec['vector'], label=sec + '(' + str(vec['segment_x']) + ')')
            plt.xlabel('Time (ms)')
            plt.ylabel('[Ca] OGB (mM)')
            lgd = plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
            fig.tight_layout()
            if self.save_figures:
                plt.savefig(self.path_saving + 'cdp_obg_' + sec + '.svg', format='svg',
                            bbox_extra_artists=(lgd,), bbox_inches='tight')
            plt.show()
            plt.close()

    def show_na_current(self):
        """Shows Na current traces for each section."""
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'currents.p', 'rb'))
        vecs = data['ina']
        for sec in vecs:
            fig = plt.figure(figsize=[12, 4])
            for vec in vecs[sec]:
                plt.plot(data['T'], vec['vector'], label=sec + '(' + str(vec['segment_x']) + ')')
            plt.xlabel('Time (ms)')
            plt.ylabel('I(Na) (mA/cm2)')
            lgd = plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
            fig.tight_layout()
            if self.save_figures:
                plt.savefig(self.path_saving + 'ina_' + sec + '.svg', format='svg',
                            bbox_extra_artists=(lgd,), bbox_inches='tight')
                print('The Na current traces was saved in the directory: ' + self.path_saving)
            plt.show()
            plt.close()

    def show_nmda_ica_current(self):
        """Shows NMDA channel-mediated calcium current traces for each section."""
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'currents.p', 'rb'))
        vecs = data['nmda_ica']
        for sec in vecs:
            if len(vecs[sec]) > 0:
                fig = plt.figure(figsize=[12, 4])
                for vec in vecs[sec]:
                    plt.plot(data['T'], vec['vector'], label=sec + '(' + str(vec['segment_x']) + ')')
                plt.xlabel('Time (ms)')
                plt.ylabel('I(NMDA) (nA)')
                lgd = plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
                fig.tight_layout()
                if self.save_figures:
                    plt.savefig(self.path_saving + 'nmda_ca' + sec + '.svg', format='svg',
                            bbox_extra_artists=(lgd,), bbox_inches='tight')
                    print('The NMDA Ca current was saved in the directory: ' + self.path_saving)
                plt.show()
                plt.close()

    def show_average_weights(self, secs, keys):
        """
        Shows average weights for given sections.

        Parameters
        ----------
        secs : list
            the list of section names lists
        keys : list
            the list of keys, labels
        """
        assert len(secs) == len(keys), 'The length of secs and keys must be the same'

        data = pickle.load(gzip.GzipFile(self.path_recordings + 'synapses.p', 'rb'))
        manager = multiprocessing.Manager()
        return_dict_perforated = manager.dict()
        return_dict_nonperforated = manager.dict()

        procs = []
        for i in range(len(secs)):
            p = multiprocessing.Process(target=calculate_average_weights,
                                        args=(data, secs[i], keys[i], 'perforated', return_dict_perforated))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
        del procs

        procs = []
        for i in range(len(secs)):
            p = multiprocessing.Process(target=calculate_average_weights,
                                        args=(data, secs[i], keys[i], 'nonperforated', return_dict_nonperforated))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
        del procs

        fig = plt.figure(figsize=[15, 4])
        for key in return_dict_perforated:
            if key != 'T':
                plt.plot(return_dict_perforated['T'], return_dict_perforated[key], label=key + ' perforated')
        for key in return_dict_nonperforated:
            if key != 'T':
                plt.plot(return_dict_nonperforated['T'], return_dict_nonperforated[key], label=key + ' nonperforated')

        lgd = plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
        plt.ylim(bottom=0)
        plt.title('Average weights')
        plt.xlabel('Time (ms)')
        plt.ylabel('Weight (uS)')
        fig.tight_layout()
        if self.save_figures:
            plt.savefig(self.path_saving + 'average_weights' + '.svg', format='svg',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            print('The Average weights were saved in the directory: ' + self.path_saving)
        plt.show()
        plt.close()

    def show_average_weights_change(self, secs, keys, baseline):
        """
        Shows evolution of average weights in time.

        Parameters
        ----------
        secs : list
            the list of section names lists
        keys : list
            the list of keys, labels
        baseline : int
            the baseline
        """
        assert len(secs) == len(keys), 'The length of secs and keys must be the same'

        data = pickle.load(gzip.GzipFile(self.path_recordings + 'synapses.p', 'rb'))
        manager = multiprocessing.Manager()
        return_dict_perforated = manager.dict()
        return_dict_nonperforated = manager.dict()

        procs = []
        for i in range(len(secs)):
            p = multiprocessing.Process(target=calculate_average_weights,
                                        args=(data, secs[i], keys[i], 'perforated', return_dict_perforated))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
        del procs

        procs = []
        for i in range(len(secs)):
            p = multiprocessing.Process(target=calculate_average_weights,
                                        args=(data, secs[i], keys[i], 'nonperforated', return_dict_nonperforated))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
        del procs

        base_idx, = np.where(np.isclose(return_dict_perforated['T'], baseline))
        fig = plt.figure(figsize=[15, 4])

        for key in return_dict_perforated:
            if key != 'T':
                baseline_w = np.mean(return_dict_perforated[key][:base_idx[0]])
                return_dict_perforated[key] = \
                    [(w / baseline_w) for w in return_dict_perforated[key]]
                if key == 'ori stim':
                    plt.plot(return_dict_perforated['T'], return_dict_perforated[key],
                             label=key + ' perforated',
                             color='tomato')
                elif key == 'ori unstim':
                    plt.plot(return_dict_perforated['T'], return_dict_perforated[key],
                             label=key + ' perforated',
                             color='salmon')
                elif key == 'rad stim':
                    plt.plot(return_dict_perforated['T'], return_dict_perforated[key],
                             label=key + ' perforated',
                             color='navy')
                elif key == 'rad unstim':
                    plt.plot(return_dict_perforated['T'], return_dict_perforated[key],
                             label=key + ' perforated',
                             color='mediumblue')
                elif key == 'lm stim':
                    plt.plot(return_dict_perforated['T'], return_dict_perforated[key],
                             label=key + ' perforated',
                             color='forestgreen')
                elif key == 'lm unstim':
                    plt.plot(return_dict_perforated['T'], return_dict_perforated[key],
                             label=key + ' perforated',
                             color='limegreen')
                elif key.split(' ')[1] == 'stim':
                    plt.plot(return_dict_perforated['T'], return_dict_perforated[key],
                             label=key + ' perforated')
                elif key.split(' ')[1] == 'unstim':
                    plt.plot(return_dict_perforated['T'], return_dict_perforated[key],
                             label=key + ' perforated')
        plt.axhline(y=1, linewidth=1)
        plt.xlabel('Time (ms)')
        plt.ylabel('weight(LTP) / weight(control)')
        plt.legend()
        fig.tight_layout()
        if self.save_figures:
            plt.savefig(self.path_saving + 'weights_changes_perforated' + '.svg', format='svg')
            print('The change in average weights of perforated synapses over time was saved in the directory: '
                  + self.path_saving)
        plt.show()
        plt.close()

        fig = plt.figure(figsize=[15, 4])
        for key in return_dict_nonperforated:
            if key != 'T':
                baseline_w = np.mean(return_dict_nonperforated[key][:base_idx[0]])
                return_dict_nonperforated[key] = \
                    [((w - baseline_w) / baseline_w) * 100 for w in return_dict_nonperforated[key]]
                if key == 'ori stim':
                    plt.plot(return_dict_nonperforated['T'], return_dict_nonperforated[key],
                             label=key + ' nonperforated',
                             color='tomato')
                elif key == 'ori unstim':
                    plt.plot(return_dict_nonperforated['T'], return_dict_nonperforated[key],
                             label=key + ' nonperforated',
                             color='salmon')
                if key == 'rad stim':
                    plt.plot(return_dict_nonperforated['T'], return_dict_nonperforated[key],
                             label=key + ' nonperforated',
                             color='navy')
                elif key == 'rad unstim':
                    plt.plot(return_dict_nonperforated['T'], return_dict_nonperforated[key],
                             label=key + ' nonperforated',
                             color='mediumblue')
                if key == 'lm stim':
                    plt.plot(return_dict_nonperforated['T'], return_dict_nonperforated[key],
                             label=key + ' nonperforated',
                             color='forestgreen')
                elif key == 'lm unstim':
                    plt.plot(return_dict_nonperforated['T'], return_dict_nonperforated[key],
                             label=key + ' nonperforated',
                             color='limegreen')
        plt.axhline(y=0, linewidth=1)
        plt.xlabel('Time (ms)')
        plt.ylabel('Weight (% change)')
        plt.legend()
        fig.tight_layout()
        if self.save_figures:
            plt.savefig(self.path_saving + 'weights_changes_nonperforated' + '.svg', format='svg')
            print('The change in average weights of nonperforated synapses over time was saved in the directory: '
                  + self.path_saving)
        plt.show()
        plt.close()

    def show_input_spikes(self):
        """Shows input spikes of AMPA synapses."""
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'synapses.p', 'rb'))
        synapses = data['synapses']
        vecs = []

        for sec in synapses:
            if len(synapses[sec]) > 0:
                for syn in synapses[sec]:
                    if syn['receptor'] == 'AMPA':
                        vecs.append(syn['input_spikes'])

        fig = plt.figure(figsize=[15, 4])
        plt.eventplot(positions=vecs, orientation='horizontal')
        #plt.xlim((1190000, 1200000))
        plt.xlabel('Time (ms)')
        plt.ylabel('Synapse id')
        fig.tight_layout()
        if self.save_figures:
            plt.savefig(self.path_saving + 'input_spikes' + '.svg', format='svg')
            print('The input spikes were saved in the directory: ' + self.path_saving)
        plt.show()
        plt.close()

    def show_SCH_COM_average_weights(self):
        """Shows average Schaffer and commissural weights."""
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'synapses.p', 'rb'))
        secs = [self.ori_secs, self.ori_secs, self.rad_secs, self.rad_secs, self.lm_secs,
                self.ori_secs + self.rad_secs, self.ori_secs + self.rad_secs]
        layers = ['ori', 'ori', 'rad', 'rad', 'lm', 'ori_rad', 'ori_rad']
        pathways = ['SCH', 'COM', 'SCH', 'COM', None, 'SCH', 'COM']

        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        procs = []
        for i in range(len(secs)):
            p = multiprocessing.Process(target=calculate_average_SCH_COM_weights,
                                        args=(data, secs[i], layers[i], pathways[i], return_dict))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
        del procs

        fig = plt.figure(figsize=[15, 4])
        for key in return_dict:
            if key != 'T':
                plt.plot(return_dict['T'], return_dict[key], label=key)

        lgd = plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
        plt.xlabel('Time (ms)')
        plt.ylabel('Weight (uS)')
        fig.tight_layout()
        if self.save_figures:
            plt.savefig(self.path_saving + 'average_SCH_COM_weights' + '.svg', format='svg',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            print('The average weights were saved in the directory: ' + self.path_saving)
        plt.show()
        plt.close()

    def show_SCH_COM_average_weights_change(self, baseline):
        """
        Shows evolution of average Schaffer and commissural weights in time.

        Parameters
        ----------
        baseline : int
            the baseline
        """
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'synapses.p', 'rb'))
        secs = [self.ori_secs, self.ori_secs, self.rad_secs, self.rad_secs, self.lm_secs,
                self.ori_secs + self.rad_secs, self.ori_secs + self.rad_secs]
        layers = ['ori', 'ori', 'rad', 'rad', 'lm', 'ori_rad', 'ori_rad']
        pathways = ['SCH', 'COM', 'SCH', 'COM', None, 'SCH', 'COM']

        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        procs = []
        for i in range(len(secs)):
            p = multiprocessing.Process(target=calculate_average_SCH_COM_weights,
                                        args=(data, secs[i], layers[i], pathways[i], return_dict))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()
        del procs

        base_idx, = np.where(np.isclose(return_dict['T'], baseline))

        fig = plt.figure(figsize=[15, 4])
        for key in return_dict:
            if key != 'T':
                baseline_w = np.mean(return_dict[key][:base_idx[0]])
                return_dict[key] = [((w - baseline_w) / baseline_w) * 100 for w in return_dict[key]]
                plt.plot(return_dict['T'], return_dict[key], label=key)

        plt.axhline(y=0, linewidth=1)
        plt.ylim(-25, 25)
        plt.xlabel('Time (ms)')
        plt.ylabel('Weight (% change)')
        lgd = plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
        fig.tight_layout()
        if self.save_figures:
            plt.savefig(self.path_saving + 'weights_changes' + '.svg', format='svg',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            print('The change in average weights over the time was saved in the directory: ' + self.path_saving)
        plt.show()
        plt.close()

    def show_SCH_COM_weights(self):
        """Shows synaptic weights for each section labeled as Schaffer, commissural or perforant."""
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'synapses.p', 'rb'))
        synapses = data['synapses']
        for sec in synapses:
            if len(synapses[sec]) > 0:
                fig = plt.figure(figsize=[15, 4])
                plt.title(sec)
                plt.xlabel('Time (ms)')
                plt.ylabel('Weight (uS)')
                legend = False
                for syn in synapses[sec]:
                    if syn['pathway'] == 'SCH':
                        plt.plot(data['T'], syn['weight'], color='red')
                        legend = True
                    elif syn['pathway'] == 'COM':
                        plt.plot(data['T'], syn['weight'], color='blue')
                        legend = True
                    elif syn['pathway'] == 'LM':
                        plt.plot(data['T'], syn['weight'], color='green')
                        legend = True
                    else:
                        plt.plot(data['T'], syn['weight'])
                        legend = False
                if legend:
                    plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left', handles=[self.SCH, self.COM, self.LM])
                fig.tight_layout()
                if self.save_figures:
                    plt.savefig(self.path_saving + 'weights_' + sec + '.svg', format='svg')
                    print('The weights were saved in the directory: ' + self.path_saving)
                plt.show()
                plt.close()

    def show_weights(self, synapse_type='perforated'):
        """Shows synaptic weights for each section.

        Parameters
        ----------
        synapse_type : str
            the synapse type (perforated or nonperforated)
        """
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'synapses.p', 'rb'))
        synapses = data['synapses']
        for sec in synapses:
            if len(synapses[sec]) > 0:
                fig = plt.figure(figsize=[15, 4])
                plt.title('Section: ' + sec + '\nSynapse type: ' + synapse_type)
                plt.xlabel('Time (ms)')
                plt.ylabel('Weight (uS)')
                for syn in synapses[sec]:
                    if (len(data['T']) == len(syn['weight'])) and (syn['type'] == synapse_type):
                        plt.plot(data['T'], syn['weight'])
                plt.ylim(bottom=0)
                fig.tight_layout()
                if self.save_figures:
                    plt.savefig(self.path_saving + 'weights_' + sec + '_' + synapse_type + '.svg', format='svg')
                    print('The weights were saved in the directory: ' + self.path_saving)
                plt.show()
                plt.close()

    def show_amplitudes(self, synapse_type='perforated'):
        """Shows amplitudes for each section.

        Parameters
        ----------
        synapse_type : str
            the synapse type (perforated or nonperforated)
        """
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'synapses.p', 'rb'))
        synapses = data['synapses']
        for sec in synapses:
            if len(synapses[sec]) > 0:
                fig = plt.figure(figsize=[15, 5])
                plt.subplot(2, 1, 1)
                plt.title('Section: ' + sec + '\nSynapse type: ' + synapse_type)
                plt.ylabel('Depression amplitudes')
                for syn in synapses[sec]:
                    if (len(data['T']) == len(syn['d_amp_vec'])) and (syn['type'] == synapse_type):
                        plt.plot(data['T'], syn['d_amp_vec'])

                plt.subplot(2, 1, 2)
                plt.ylabel('Potentiation amplitudes')
                plt.xlabel('Time (ms)')
                for syn in synapses[sec]:
                    if (len(data['T']) == len(syn['p_amp_vec'])) and (syn['type'] == synapse_type):
                        plt.plot(data['T'], syn['p_amp_vec'])

                fig.tight_layout()
                if self.save_figures:
                    plt.savefig(self.path_saving + 'weights_' + sec + '_' + synapse_type + '.svg', format='svg')
                    print('The weights were saved in the directory: ' + self.path_saving)
                plt.show()
                plt.close()

    def show_weights_distance(self, start=0, stop=0, synapse_type='perforated'):
        """
        Shows the percentage change for each synapse between the weight at start time and the weight at stop time
        as a scatter plot. X-axis represents distance of a synapse from the soma.

        Parameters
        ----------
        start : int
            the start time
        stop : int
            the stop time
        synapse_type : str
            the synapse type (perforated or nonperforated)
        """
        from cycler import cycler
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'synapses.p', 'rb'))
        index_start, = np.where(np.isclose(data['T'], start))
        if len(index_start) == 0:
            start = 0.0
            index_start = 0
        else:
            index_start = index_start[0]
            start = data['T'][index_start]

        index_stop, = np.where(np.isclose(data['T'], stop))
        if len(index_stop) == 0:
            stop = 0.0
            index_stop = 0
        else:
            index_stop = index_stop[0]
            stop = data['T'][index_stop]

        fig, ax = plt.subplots()
        ax.set_prop_cycle(cycler(color=plt.get_cmap('tab20').colors))
        for sec in data['synapses']:
            if sec in self.ori_secs:
                for syn in data['synapses'][sec]:
                    if syn['receptor'] == 'AMPA' and syn['type'] == synapse_type:
                        weight_change = syn['weight'][index_stop] / syn['weight'][index_start]
                        # weight_change = ((syn['weight'][index_stop] - syn['weight'][index_start])
                        #                  / syn['weight'][index_start]) * 100
                        if syn['stimulated']:
                            plt.scatter(x=syn['distance'] * (-1), y=weight_change, color='tomato')
                        else:
                            plt.scatter(x=syn['distance'] * (-1), y=weight_change, color='salmon')
            elif sec in self.rad_secs:
                for syn in data['synapses'][sec]:
                    if syn['receptor'] == 'AMPA' and syn['type'] == synapse_type:
                        weight_change = syn['weight'][index_stop] / syn['weight'][index_start]
                        # weight_change = ((syn['weight'][index_stop] - syn['weight'][index_start])
                        #                 / syn['weight'][index_start]) * 100
                        if syn['stimulated']:
                            plt.scatter(x=syn['distance'], y=weight_change, color='navy')
                        else:
                            plt.scatter(x=syn['distance'], y=weight_change, color='mediumblue')
            elif sec in self.lm_secs:
                for syn in data['synapses'][sec]:
                    if syn['receptor'] == 'AMPA' and syn['type'] == synapse_type:
                        weight_change = syn['weight'][index_stop] / syn['weight'][index_start]
                        # weight_change = ((syn['weight'][index_stop] - syn['weight'][index_start])
                        #                  / syn['weight'][index_start]) * 100
                        if syn['stimulated']:
                            plt.scatter(x=syn['distance'], y=weight_change, color='forestgreen')
                        else:
                            plt.scatter(x=syn['distance'], y=weight_change, color='limegreen')
            else:
                weight_change = []
                distances = []
                for syn in data['synapses'][sec]:
                    if syn['receptor'] == 'AMPA' and syn['type'] == synapse_type:
                        weight_change.append(syn['weight'][index_stop] / syn['weight'][index_start])
                        distances.append(syn['distance'])
                        # weight_change = ((syn['weight'][index_stop] - syn['weight'][index_start])
                        #                  / syn['weight'][index_start]) * 100
                        # plt.scatter(x=syn['distance'], y=weight_change, label=sec)
                if len(weight_change) > 0:
                    plt.scatter(x=distances, y=weight_change, label=sec)

        plt.axhline(y=1.0, linewidth=1)
        # plt.ylim(1.0, 1.4)
        plt.xlabel('Distance from the soma (um)')
        plt.ylabel('Weight change (w_stop / w_start)')
        plt.title('Start time: ' + str(start) + ' ms; stop time: ' + str(stop) + ' ms' +
                  '\nSynapse type: ' + synapse_type)
        # plt.legend()
        lgd = plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
        fig.tight_layout()
        if self.save_figures:
            plt.savefig(self.path_saving + 'wdist' + '.svg', format='svg',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            print('The percentage change of synaptic weights was saved in the directory: ' + self.path_saving)
        plt.show()
        plt.close()

    def show_weights_histogram(self, t=0, sections='all', synapse_type='all'):
        """
        Shows the histogram of synaptic weights in the time t.

        Parameters
        ----------
        t : int
            the time
        sections : str
            the sections (all, ori, rad, lm)
        synapse_type : str
            the synapse type (all, perforated, nonperforated)
        """
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'synapses.p', 'rb'))
        index, = np.where(np.isclose(data['T'], t))
        if len(index) == 0:
            t = 0.0
            index = 0
        else:
            index = index[0]
            t = data['T'][index]

        all_synapses = []
        all_perforated = []
        all_nonperforated = []
        ori = []
        ori_perforated = []
        ori_nonperforated = []
        rad = []
        rad_perforated = []
        rad_nonperforated = []
        lm = []
        lm_perforated = []
        lm_nonperforated = []

        for sec in data['synapses']:
            if sec in self.ori_secs:
                for syn in data['synapses'][sec]:
                    if syn['receptor'] == 'AMPA':
                        all_synapses.append(syn['weight'][index])
                        ori.append(syn['weight'][index])
                        if syn['type'] == 'perforated':
                            ori_perforated.append(syn['weight'][index])
                            all_perforated.append(syn['weight'][index])
                        if syn['type'] == 'nonperforated':
                            ori_nonperforated.append(syn['weight'][index])
                            all_nonperforated.append(syn['weight'][index])
            if sec in self.rad_secs:
                for syn in data['synapses'][sec]:
                    if syn['receptor'] == 'AMPA':
                        all_synapses.append(syn['weight'][index])
                        rad.append(syn['weight'][index])
                        if syn['type'] == 'perforated':
                            rad_perforated.append(syn['weight'][index])
                            all_perforated.append(syn['weight'][index])
                        if syn['type'] == 'nonperforated':
                            rad_nonperforated.append(syn['weight'][index])
                            all_nonperforated.append(syn['weight'][index])
            if sec in self.lm_secs:
                for syn in data['synapses'][sec]:
                    if syn['receptor'] == 'AMPA':
                        all_synapses.append(syn['weight'][index])
                        lm.append(syn['weight'][index])
                        if syn['type'] == 'perforated':
                            lm_perforated.append(syn['weight'][index])
                            all_perforated.append(syn['weight'][index])
                        if syn['type'] == 'nonperforated':
                            lm_nonperforated.append(syn['weight'][index])
                            all_nonperforated.append(syn['weight'][index])

        fig = plt.figure(figsize=[10, 5])
        if sections == 'all' and synapse_type == 'all':
            plt.hist(all_synapses, bins=15)
            plt.title('All synaptic weights' + ' , t = ' + str(t) + ' ms')
        elif sections == 'all' and synapse_type == 'perforated':
            plt.hist(all_perforated, bins=20)
            plt.title('All perforated synaptic weights' + ' , t = ' + str(t) + ' ms')
        elif sections == 'all' and synapse_type == 'nonperforated':
            plt.hist(all_nonperforated)
            plt.title('All nonperforated synaptic weights' + ' , t = ' + str(t) + ' ms')
        elif sections == 'ori' and synapse_type == 'all':
            plt.hist(ori)
            plt.title('Ori synaptic weights' + ' , t = ' + str(t) + ' ms')
        elif sections == 'ori' and synapse_type == 'perforated':
            plt.hist(ori_perforated)
            plt.title('Ori perforated synaptic weights' + ' , t = ' + str(t) + ' ms')
        elif sections == 'ori' and synapse_type == 'nonperforated':
            plt.hist(ori_nonperforated, bins=30)
            plt.title('Ori nonperforated synaptic weights' + ' , t = ' + str(t) + ' ms')
        elif sections == 'rad' and synapse_type == 'all':
            plt.hist(rad)
            plt.title('Rad synaptic weights' + ' , t = ' + str(t) + ' ms')
        elif sections == 'rad' and synapse_type == 'perforated':
            plt.hist(rad_perforated)
            plt.title('Rad perforated synaptic weights' + ' , t = ' + str(t) + ' ms')
        elif sections == 'rad' and synapse_type == 'nonperforated':
            plt.hist(rad_nonperforated, bins=30)
            plt.title('Rad nonperforated synaptic weights' + ' , t = ' + str(t) + ' ms')
        elif sections == 'lm' and synapse_type == 'all':
            plt.hist(lm)
            plt.title('LM synaptic weights' + ' , t = ' + str(t) + ' ms')
        elif sections == 'lm' and synapse_type == 'perforated':
            plt.hist(lm_perforated)
            plt.title('LM perforated synaptic weights' + ' , t = ' + str(t) + ' ms')
        elif sections == 'lm' and synapse_type == 'nonperforated':
            plt.hist(lm_nonperforated, bins=30)
            plt.title('LM nonperforated synaptic weights' + ' , t = ' + str(t) + ' ms')

        plt.xlabel('Weights (uS)')
        plt.ylabel('Frequency')
        fig.tight_layout()
        if self.save_figures:
            plt.savefig(self.path_saving + 'histogram_t_' + str(t) + '_ms' + '.svg', format='svg')
            print('The histogram of synaptic weights was saved in the directory: ' + self.path_saving)
        plt.show()
        plt.close()

    def show_dendritic_voltage(self, threshold):
        """Shows dendritic voltage for each section."""
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'voltages.p', 'rb'))
        d_vecs = data['V_dends']
        for sec in d_vecs:
            fig = plt.figure(figsize=[12, 4])
            plt.plot(data['T'], data['V_soma'], label='soma(0.5)')
            for d_vec in d_vecs[sec]:
                plt.plot(data['T'], d_vec['vector'], label=sec + '(' + str(d_vec['segment_x']) + ')')
            plt.ylim((-80, 30))
            # plt.xlim((50, 200))
            plt.hlines(y=threshold, xmin=0, xmax=10000)
            plt.title(sec)
            plt.xlabel('Time (ms)')
            plt.ylabel('Voltage (mV)')
            lgd = plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
            fig.tight_layout()
            if self.save_figures:
                plt.savefig(self.path_saving + 'dendritic_voltage_' + sec + '.svg', format='svg',
                            bbox_extra_artists=(lgd,), bbox_inches='tight')
                print('The dendritic voltage was saved in the directory: ' + self.path_saving)
            plt.show()
            plt.close()

    def show_voltage_on_spines(self, threshold):
        """Shows voltage on spines."""
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'voltages.p', 'rb'))
        v_vecs = data['V_spines']

        fig = plt.figure(figsize=[12, 4])
        for i in range(len(v_vecs)):
            plt.plot(data['T'], v_vecs[i].vector, label='spine_' + str(i))
        plt.ylim((-75, 10))
        # plt.xlim((50, 200))
        plt.hlines(y=threshold, xmin=0, xmax=40000)
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')
        lgd = plt.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
        fig.tight_layout()
        if self.save_figures:
            plt.savefig(self.path_saving + 'spines_voltage' + '.svg', format='svg',
                        bbox_extra_artists=(lgd,), bbox_inches='tight')
            print('The voltage from spines was saved in the directory: ' + self.path_saving)
        plt.show()
        plt.close()

    def show_IO_characteristics(self, baseline):
        """
        Shows average input and output firing frequencies.

        Parameters
        ----------
        baseline : int
            the time for calculating output frequency
        """
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'voltages.p', 'rb'))

        apc_n = data['APs']
        i = 0
        while apc_n[i] <= baseline:
            i = i + 1

        freq = len(apc_n[0:i]) / (baseline / 1000.0)

        print("Average input frequency: " + str(self.setting['netstim']['NETSTIM_FREQUENCY']))
        print("Number of output spikes during baseline: " + str(len(apc_n[0:i])))
        print("Output firing frequency of baseline: " + str(freq))
        print('-------------------------------------------------')

    def show_somatic_voltage(self):
        """Shows the somatic voltage."""
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'voltages.p', 'rb'))

        fig = plt.figure(figsize=[10, 2.5])
        plt.plot(data['T'], data['V_soma'], label='soma(0.5)')
        # plt.ylim((-80, 40))
        # plt.xlim((50, 200))
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')
        # ticks = [0,60000,120000,180000,240000,300000]
        # tick_labels = [0,1,2,3,4,5]
        # plt.xticks(ticks, tick_labels)
        #plt.legend()
        fig.tight_layout()
        if self.save_figures:
            plt.savefig(self.path_saving + 'somatic_voltage' + '.svg', format='svg')
            print('The somatic voltage was saved in the directory: ' + self.path_saving)
        plt.show()
        plt.close()

    def show_somatic_voltage_derivative(self):
        """Shows the somatic voltage derivative."""
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'voltages.p', 'rb'))
        v = data['V_soma'][:40000]
        t = data['T'][:40000]
        v2 = []
        t2 = []
        for i in range(0, len(v), 40):
            v2.append(v[i])
            t2.append(t[i])

        time_step = t2[1] - t2[0]
        n = len(v2)
        dv = [(v2[1] - v2[0])]
        dt = [t2[0]]
        for i in range(1, n-1, 1):
            dv.append(((v2[i + 1] - v2[i - 1]) / 2))
            dt.append(t2[i])
        dv.append((v2[n - 1] - v2[n - 2]))
        dt.append(t2[-1])

        fig = plt.figure(figsize=[10, 2.5])
        plt.plot(dt, dv, label='soma(0.5)')
        # plt.ylim((-80, 40))
        # plt.xlim((50, 200))
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage (mV)')
        # ticks = [0,60000,120000,180000,240000,300000]
        # tick_labels = [0,1,2,3,4,5]
        # plt.xticks(ticks, tick_labels)
        #plt.legend()
        fig.tight_layout()
        if self.save_figures:
            plt.savefig(self.path_saving + 'somatic_voltage' + '.svg', format='svg')
            print('The somatic voltage was saved in the directory: ' + self.path_saving)
        plt.show()
        plt.close()

    def show_mean_normalized_change_EPSP_amplitude(self, section, i):
        data = pickle.load(gzip.GzipFile(self.path_recordings + 'voltages.p', 'rb'))
        efel.setThreshold(-71)
        trace_test = {}
        trace_test['T'] = data['T']
        trace_test['V'] = data['V_dends'][section][i]['vector']
        trace_test['stim_start'] = [100]
        trace_test['stim_end'] = [6000]

        trace_stim = {}
        trace_stim['T'] = data['T']
        trace_stim['V'] = data['V_dends'][section][i]['vector']
        trace_stim['stim_start'] = [25000]
        trace_stim['stim_end'] = [30000]

        traces = [trace_test, trace_stim]
        traces_results = efel.getFeatureValues(traces, ['maximum_voltage', 'voltage_base'])
        print(traces_results)
        amp_test = traces_results[0]['maximum_voltage'][0] - traces_results[0]['voltage_base'][0]
        amp_stim = traces_results[1]['maximum_voltage'][0] - traces_results[1]['voltage_base'][0]
        print("amp_stim / amp_test: " + str(amp_stim / amp_test))
