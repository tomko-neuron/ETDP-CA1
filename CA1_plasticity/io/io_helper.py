"""
Title: io_helper.py
Author: Matus Tomko
Mail: matus.tomko __at__ fmph.uniba.sk
"""
import glob
import gzip
import json
import multiprocessing
import os
import pickle

import numpy as np


def prepare_recording_vector(rec_vec, i):
    """
    Prepares a recording vectors for saving.

    Parameters
    ----------
    rec_vec : neuron.hoc.HocObject
        the recording vector

    Returns
    -------
    vec : dict
        the recording vector as a dictionary
    """
    vec = {
        'section': rec_vec.section,
        'segment_x': rec_vec.segment_x,
        'vector': np.array(rec_vec.vector)
    }
    return vec


class IOHelper:
    """
    A class used to load and save data

    ...

    Attributes
    ----------
    path_saving : str
        the path to the directory where the data is stored or will be saved
    path_settings : str
        the path to the synapses .json file
    npool : str, optional
        the number of pool processes (default multiprocessing.cpu_count() - 1)
    setting : dict
        the dictionary containing all settings

    Methods
    -------
    save_recordings(synapses, tw_vec, v_soma_vec, t_vec, dend_vecs, p_vec, d_vec, alpha_scount_vec, ta_vec,
                    apc_vec, cai_vecs, cal_ica_vecs, ina_vecs, nmda_ica_vecs)
        Saves the recorded data.
    save_segment_recordings(segment, v_soma_vec, t_vec, apc_vec, dend_vecs, ta_vec, alpha_scount_vec, d_vec,
                            p_vec, synapses, tw_vec, cai_vecs, cal_ica_vecs, ina_vecs, nmda_ica_vecs)
        Saves the recorded data to a dictionary structures in binary temporary files from one simulation segment.
    reconstruct_segmented_bcm():
        Creates a final BCM file from temporary files created from a segmented simulation.
    reconstruct_segmented_currents():
        Creates a final currents file from temporary files created from a segmented simulation.
    reconstruct_segmented_synapses():
        Creates a final synapses file from temporary files created from a segmented simulation.
    reconstruct_segmented_voltages():
        Creates a final voltages file from temporary files created from a segmented simulation.
    delete_temporary_files():
        Deletes all temporary files.
    prepare_dict_recording_vectors(vecs)
        Prepares a dictionary of recording vectors for saving.
    load_synapses(json_synapses)
        Loads setting from the synapses.json file.
    save_synapses(synapses)
        Saves synapses to a .json file.
    load_setting()
        Loads setting from the setting.json file.
    save_setting(setting)
        Saves setting to the setting.json file.
    set_final_weights_as_init_weights(input_path, output_path, json_synapses)
        Sets the final weights as the initial weights.
    """

    def __init__(self, path_saving, path_settings):
        """
        Parameters
        ----------
        path_saving : str
            the path to the directory where the data will be saved
        path_settings : str
            the path to the directory with settings
        """

        self.path_saving = path_saving
        self.path_settings = path_settings
        self.npool = multiprocessing.cpu_count() - 1
        self.setting = None

        try:
            if not os.path.exists(self.path_saving):
                os.makedirs(self.path_saving)
            if not os.path.exists(self.path_saving + 'tmp/'):
                os.makedirs(self.path_saving + 'tmp/')
        except OSError as e:
            if e.errno != 17:
                raise
            pass

    def save_recordings(self, synapses, tw_vec, v_soma_vec, t_vec, dend_vecs, p_vec, d_vec, alpha_scount_vec, ta_vec,
                        apc_vec, cai_vecs, cal_ica_vecs, ina_vecs, nmda_ica_vecs, pmp_vecs, ogb_vecs, spines_v_vecs):
        """
        Saves the recorded data to a dictionary structures in binary files.

        Parameters
        ----------
        synapses : dict
            the dictionary of synapses
        tw_vec : neuron.hoc.HocObject
            the time vector for synaptic weights
        v_soma_vec : neuron.hoc.HocObject
            the somatic voltage vector
        t_vec : neuron.hoc.HocObject
            the time vector for voltage
        dend_vecs : dict
            the dictionary containing voltage vectors from dendrites
        p_vec : neuron.hoc.HocObject
            the potentiation amplitude vector
        d_vec : neuron.hoc.HocObject
            the depression amplitude vector
        alpha_scount_vec : neuron.hoc.HocObject
            the integrated spike count vector
        ta_vec : neuron.hoc.HocObject
            the time vector for amplitudes
        apc_vec : neuron.hoc.HocObject
            the vector of times of fired action potentials
        cai_vecs : dict
            the dictionary containing intracellular calcium concentration vectors
        cal_ica_vecs : dict
            the dictionary containing CaL channel-mediated calcium current vectors
        ina_vecs : dict
            the dictionary containing sodium current vectors
        nmda_ica_vecs : dict
            the dictionary of NMDAR channel-mediated calcium current vectors
        pmp_vecs : dist
            the dictionary containing calcium pump vectors
        ogb_vecs : dist
            the dictionary containing calcium indicator vectors
        spines_v_vecs : list
            the list containing voltage vectors from spine heads
        """
        # a dictionary of synapses
        synapses_dict = {}
        for sec in synapses:
            synapses_list = []
            for syn in synapses[sec]:
                s = {
                    'name': str(syn.synapse),
                    'synapse_id': str(syn.synapse_id),
                    'section': str(syn.section),
                    'segment_x': syn.segment_x,
                    'distance': syn.distance,
                    'weight': np.array(syn.weight_vec.as_numpy()),
                    'input_spikes': np.array(syn.input_spikes.as_numpy()),
                    'stimulated': syn.stimulated,
                    'receptor': syn.receptor,
                    'pathway': syn.pathway,
                    'type': syn.type,
                    'd_amp_vec': np.array(syn.d_amp_vec.as_numpy()),
                    'p_amp_vec': np.array(syn.p_amp_vec.as_numpy())
                }
                synapses_list.append(s)
            synapses_dict[sec] = synapses_list

        print('Saving recordings...')

        # saving of synapses
        weights = {}
        weights['T'] = np.array(tw_vec.as_numpy())
        weights['synapses'] = synapses_dict
        pickle.dump(weights, gzip.GzipFile(self.path_saving + 'synapses.p', 'wb'))
        print('The synapses were saved in the directory: ' + self.path_saving)

        # saving of voltages
        voltages = {}
        voltages['T'] = np.array(t_vec.as_numpy())
        voltages['V_soma'] = np.array(v_soma_vec.as_numpy())
        voltages['APs'] = np.array(apc_vec.as_numpy())
        voltages['V_dends'] = self.prepare_dict_recording_vectors(vecs=dend_vecs)
        voltages['V_spines'] = spines_v_vecs
        pickle.dump(voltages, gzip.GzipFile(self.path_saving + 'voltages.p', 'wb'))
        print('The voltages were saved in the directory: ' + self.path_saving)

        # saving of currents
        currents = {}
        currents['T'] = np.array(t_vec.as_numpy())
        currents['cai'] = self.prepare_dict_recording_vectors(vecs=cai_vecs)
        currents['cal_ica'] = self.prepare_dict_recording_vectors(vecs=cal_ica_vecs)
        currents['nmda_ica'] = self.prepare_dict_recording_vectors(vecs=nmda_ica_vecs)
        currents['ina'] = self.prepare_dict_recording_vectors(vecs=ina_vecs)
        currents['pmp'] = self.prepare_dict_recording_vectors(vecs=pmp_vecs)
        currents['ogb'] = self.prepare_dict_recording_vectors(vecs=ogb_vecs)
        pickle.dump(currents, gzip.GzipFile(self.path_saving + 'currents.p', 'wb'))
        print('The currents were saved in the directory: ' + self.path_saving)

        # saving of BCM parameters
        bcm = {}
        bcm['P_amp'] = np.array(p_vec.as_numpy())
        bcm['D_amp'] = np.array(d_vec.as_numpy())
        bcm['alpha_scount'] = np.array(alpha_scount_vec.as_numpy())
        bcm['T_BCM'] = np.array(ta_vec.as_numpy())
        pickle.dump(bcm, gzip.GzipFile(self.path_saving + 'bcm.p', 'wb'))
        print('The BCM recordings were saved in the directory: ' + self.path_saving)

    def save_segment_recordings(self, segment, v_soma_vec, t_vec, apc_vec, dend_vecs, ta_vec, alpha_scount_vec, d_vec,
                                p_vec, synapses, tw_vec, cai_vecs, cal_ica_vecs, ina_vecs, nmda_ica_vecs, pmp_vecs,
                                ogb_vecs, spines_v_vecs):
        """
        Saves the recorded data to a dictionary structures in binary temporary files from one simulation segment.

        Parameters
        ----------
        segment : int
            the segment id
        synapses : dict
            the dictionary of synapses
        tw_vec : neuron.hoc.HocObject
            the time vector for synaptic weights
        v_soma_vec : neuron.hoc.HocObject
            the somatic voltage vector
        t_vec : neuron.hoc.HocObject
            the time vector for voltage
        dend_vecs : dict
            the dictionary containing voltage vectors from dendrites
        p_vec : neuron.hoc.HocObject
            the potentiation amplitude vector
        d_vec : neuron.hoc.HocObject
            the depression amplitude vector
        alpha_scount_vec : neuron.hoc.HocObject
            the integrated spike count vector
        ta_vec : neuron.hoc.HocObject
            the time vector for amplitudes
        apc_vec : neuron.hoc.HocObject
            the vector of times of fired action potentials
        cai_vecs : dict
            the dictionary containing intracellular calcium concentration vectors
        cal_ica_vecs : dict
            the dictionary containing CaL channel-mediated calcium current vectors
        ina_vecs : dict
            the dictionary containing sodium current vectors
        nmda_ica_vecs : dict
            the dictionary of NMDAR channel-mediated calcium current vectors
        pmp_vecs : dist
            the dictionary containing calcium pump vectors
        ogb_vecs : dist
            the dictionary containing calcium indicator vectors
        spines_v_vecs : list
            the list containing voltage vectors from spine heads
        """
        # a dictionary of synapses
        synapses_dict = {}
        for sec in synapses:
            synapses_list = []
            for syn in synapses[sec]:
                s = {
                    'name': str(syn.synapse),
                    'synapse_id': str(syn.synapse_id),
                    'section': str(syn.section),
                    'segment_x': syn.segment_x,
                    'distance': syn.distance,
                    'weight': np.array(syn.weight_vec.as_numpy()),
                    'input_spikes': np.array(syn.input_spikes.as_numpy()),
                    'stimulated': syn.stimulated,
                    'receptor': syn.receptor,
                    'pathway': syn.pathway,
                    'type': syn.type,
                    'd_amp_vec': np.array(syn.d_amp_vec.as_numpy()),
                    'p_amp_vec': np.array(syn.p_amp_vec.as_numpy())
                }
                synapses_list.append(s)
            synapses_dict[sec] = synapses_list

        # saving of synapses
        weights = {}
        weights['T'] = np.array(tw_vec.as_numpy())
        weights['synapses'] = synapses_dict
        pickle.dump(weights, gzip.GzipFile(self.path_saving + 'tmp/synapses_' + str(segment) + '.p', 'wb'))

        # saving of voltages
        voltages = {}
        voltages['T'] = np.array(t_vec.as_numpy())
        voltages['V_soma'] = np.array(v_soma_vec.as_numpy())
        voltages['APs'] = np.array(apc_vec.as_numpy())
        voltages['V_dends'] = self.prepare_dict_recording_vectors(vecs=dend_vecs)
        voltages['V_spines'] = spines_v_vecs
        pickle.dump(voltages, gzip.GzipFile(self.path_saving + 'tmp/voltages_' + str(segment) + '.p', 'wb'))

        # saving of BCM parameters
        bcm = {}
        bcm['P_amp'] = np.array(p_vec.as_numpy())
        bcm['D_amp'] = np.array(d_vec.as_numpy())
        bcm['alpha_scount'] = np.array(alpha_scount_vec.as_numpy())
        bcm['T_BCM'] = np.array(ta_vec.as_numpy())
        pickle.dump(bcm, gzip.GzipFile(self.path_saving + 'tmp/bcm_' + str(segment) + '.p', 'wb'))

        # saving of currents
        currents = {}
        currents['T'] = np.array(t_vec.as_numpy())
        currents['cai'] = self.prepare_dict_recording_vectors(vecs=cai_vecs)
        currents['cal_ica'] = self.prepare_dict_recording_vectors(vecs=cal_ica_vecs)
        currents['nmda_ica'] = self.prepare_dict_recording_vectors(vecs=nmda_ica_vecs)
        currents['ina'] = self.prepare_dict_recording_vectors(vecs=ina_vecs)
        currents['pmp'] = self.prepare_dict_recording_vectors(vecs=pmp_vecs)
        currents['ogb'] = self.prepare_dict_recording_vectors(vecs=ogb_vecs)
        pickle.dump(currents, gzip.GzipFile(self.path_saving + 'tmp/currents_' + str(segment) + '.p', 'wb'))

    def reconstruct_segmented_bcm(self):
        """Creates a final BCM file from temporary files created from a segmented simulation."""
        bcm = {}
        bcm['P_amp'] = []
        bcm['D_amp'] = []
        bcm['alpha_scount'] = []
        bcm['T_BCM'] = []
        pickle.dump(bcm, gzip.GzipFile(self.path_saving + 'bcm.p', 'wb'))

        for segment in range(self.setting['simulation']['NUM_SEGMENTS']):
            bcm = pickle.load(gzip.GzipFile(self.path_saving + 'bcm.p', 'rb'))
            data = pickle.load(gzip.GzipFile(self.path_saving + 'tmp/bcm_' + str(segment) + '.p', 'rb'))
            bcm['P_amp'] = np.concatenate((bcm['P_amp'], data['P_amp']), axis=None)
            bcm['D_amp'] = np.concatenate((bcm['D_amp'], data['D_amp']), axis=None)
            bcm['T_BCM'] = np.concatenate((bcm['T_BCM'], data['T_BCM']), axis=None)
            if segment + 1 == self.setting['simulation']['NUM_SEGMENTS']:
                bcm['alpha_scount'] = data['alpha_scount']
            pickle.dump(bcm, gzip.GzipFile(self.path_saving + 'bcm.p', 'wb'))

    def reconstruct_segmented_currents(self):
        """Creates a final current file from temporary files created from a segmented simulation."""
        currents = {}
        currents['T'] = []
        currents['cai'] = {}
        currents['cal_ica'] = {}
        currents['nmda_ica'] = {}
        currents['ina'] = {}
        currents['pmp'] = {}
        currents['ogb'] = {}
        pickle.dump(currents, gzip.GzipFile(self.path_saving + 'currents.p', 'wb'))

        for segment in range(self.setting['simulation']['NUM_SEGMENTS']):
            currents = pickle.load(gzip.GzipFile(self.path_saving + 'currents.p', 'rb'))
            data = pickle.load(gzip.GzipFile(self.path_saving + 'tmp/currents_' + str(segment) + '.p', 'rb'))
            currents['T'] = np.concatenate((currents['T'], data['T']), axis=None)
            for key in ['cai', 'cal_ica', 'nmda_ica', 'ina', 'pmp', 'ogb']:
                for sec in data[key]:
                    if segment == 0:
                        currents[key][sec] = []
                        for i in range(len(data[key][sec])):
                            currents[key][sec].append(data[key][sec][i])
                    else:
                        for i in range(len(data[key][sec])):
                            currents[key][sec][i]['vector'] = np.concatenate((currents[key][sec][i]['vector'],
                                                                            data[key][sec][i]['vector']),
                                                                            axis=None)
            pickle.dump(currents, gzip.GzipFile(self.path_saving + 'currents.p', 'wb'))

    def reconstruct_segmented_synapses(self):
        """Creates a final synapse file from temporary files created from a segmented simulation."""
        synapses = {}
        synapses['T'] = []
        synapses['synapses'] = {}
        pickle.dump(synapses, gzip.GzipFile(self.path_saving + 'synapses.p', 'wb'))

        for segment in range(self.setting['simulation']['NUM_SEGMENTS']):
            # reconstructing of synapses
            synapses = pickle.load(gzip.GzipFile(self.path_saving + 'synapses.p', 'rb'))
            data = pickle.load(gzip.GzipFile(self.path_saving + 'tmp/synapses_' + str(segment) + '.p', 'rb'))

            synapses['T'] = np.concatenate((synapses['T'], data['T']), axis=None)
            for sec in data['synapses']:
                if segment == 0:
                    synapses['synapses'][sec] = []
                    for i in range(len(data['synapses'][sec])):
                        synapses['synapses'][sec].append(data['synapses'][sec][i])
                else:
                    for i in range(len(data['synapses'][sec])):
                        synapses['synapses'][sec][i]['weight'] = np.concatenate(
                            (synapses['synapses'][sec][i]['weight'],
                             data['synapses'][sec][i]['weight']), axis=None)
                        synapses['synapses'][sec][i]['input_spikes'] = np.concatenate(
                            (synapses['synapses'][sec][i]['input_spikes'],
                             data['synapses'][sec][i]['input_spikes']), axis=None)
                        synapses['synapses'][sec][i]['d_amp_vec'] = np.concatenate(
                            (synapses['synapses'][sec][i]['d_amp_vec'],
                             data['synapses'][sec][i]['d_amp_vec']), axis=None)
                        synapses['synapses'][sec][i]['p_amp_vec'] = np.concatenate(
                            (synapses['synapses'][sec][i]['p_amp_vec'],
                             data['synapses'][sec][i]['p_amp_vec']), axis=None)
            pickle.dump(synapses, gzip.GzipFile(self.path_saving + 'synapses.p', 'wb'))

    def reconstruct_segmented_voltages(self):
        """Creates a final voltage file from temporary files created from a segmented simulation."""
        voltages = {}
        voltages['T'] = []
        voltages['V_soma'] = []
        voltages['APs'] = []
        voltages['V_dends'] = {}
        voltages['V_spines'] = []
        pickle.dump(voltages, gzip.GzipFile(self.path_saving + 'voltages.p', 'wb'))

        for segment in range(self.setting['simulation']['NUM_SEGMENTS']):
            voltages = pickle.load(gzip.GzipFile(self.path_saving + 'voltages.p', 'rb'))
            data = pickle.load(gzip.GzipFile(self.path_saving + 'tmp/voltages_' + str(segment) + '.p', 'rb'))
            voltages['T'] = np.concatenate((voltages['T'], data['T']), axis=None)
            voltages['V_soma'] = np.concatenate((voltages['V_soma'], data['V_soma']), axis=None)
            voltages['APs'] = np.concatenate((voltages['APs'], data['APs']), axis=None)
            for sec in data['V_dends']:
                if segment == 0:
                    voltages['V_dends'][sec] = []
                    for i in range(len(data['V_dends'][sec])):
                        voltages['V_dends'][sec].append(data['V_dends'][sec][i])
                else:
                    for i in range(len(data['V_dends'][sec])):
                        voltages['V_dends'][sec][i]['vector'] = np.concatenate((voltages['V_dends'][sec][i]['vector'],
                                                                                data['V_dends'][sec][i]['vector']),
                                                                               axis=None)
            for i in range(len(data['V_spines'])):
                if segment == 0:
                    voltages['V_spines'].append(data['V_spines'][i])
                else:
                    voltages['V_spines'][i].vector = np.concatenate((voltages['V_spines'][i].vector,
                                                                     data['V_spines'][i].vector), axis=None)
            pickle.dump(voltages, gzip.GzipFile(self.path_saving + 'voltages.p', 'wb'))

    def delete_temporary_files(self):
        """Deletes all temporary files."""
        files = glob.glob(self.path_saving + 'tmp/*')
        for f in files:
            os.remove(f)

    def prepare_dict_recording_vectors(self, vecs):
        """
        Prepares a dictionary of recording vectors for saving.

        Parameters
        ----------
        vecs : dict
            the dictionary of recording vectors

        Returns
        -------
        vecs_dict : dict
            the dictionary of recording vectors
        """
        pool = multiprocessing.Pool(processes=self.npool, maxtasksperchild=1)
        vecs_dict = {}
        for sec in vecs:
            saved_nmda_ica_vecs = [pool.apply(prepare_recording_vector, args=(vecs[sec][i], i)) for i in
                                   range(len(vecs[sec]))]
            vecs_dict[sec] = saved_nmda_ica_vecs
        pool.terminate()
        pool.join()
        del pool
        return vecs_dict

    def load_synapses(self, json_synapses):
        """
        Loads setting from the .json file.

        Parameters
        ----------
        json_synapses : object
            the synapses in .json file

        Returns
        --------
        synapses : dict
            the dictionary of synapses

        Raises
        ------
        FileNotFoundError
        """
        try:
            with open(self.path_settings + json_synapses, 'r') as f:
                synapses = json.load(f)
                return synapses
        except FileNotFoundError as fnf_error:
            raise fnf_error

    def save_synapses(self, synapses):
        """
        Saves synapses to a .json file.

        Parameters
        ----------
        synapses : dict
            the dictionary of synapses
        """
        data = {}
        for sec in synapses:
            synapses_list = []
            for s in synapses[sec]:
                w = np.array(s.weight_vec.as_numpy())
                syn = {
                    'name': str(s.synapse),
                    'section': str(s.section),
                    'segment_x': s.segment_x,
                    'distance': s.distance,
                    'stimulated': s.stimulated,
                    'receptor': s.receptor,
                    'pathway': s.pathway,
                    'type': s.type,
                    'synapse_id': s.synapse_id
                }
                if len(w) > 0:
                    syn['init_weight'] = s.init_weight
                    syn['final_weight'] = w[-1]
                else:
                    syn['init_weight'] = s.init_weight
                    syn['final_weight'] = s.init_weight
                synapses_list.append(syn)
            data[sec] = synapses_list

        with open(self.path_saving + 'synapses.json', 'w') as f:
            json.dump(data, f, indent=4)
            print('Synapses were saved in the directory: ' + self.path_saving)

    def load_setting(self):
        """
        Loads setting from the setting.json file.

        Returns
        -------
        setting : dict
            the dictionary of setting

        Raises
        ------
        FileNotFoundError
        """
        try:
            with open(self.path_settings + 'setting.json', 'r') as f:
                setting = json.load(f)
                self.setting = setting
                return setting
        except FileNotFoundError as fnf_error:
            raise fnf_error

    def save_setting(self, setting):
        """
        Saves setting to the setting.json file.

        Parameters
        ----------
        setting : dict
            the dictionary of setting
        """
        with open(self.path_saving + 'setting.json', 'w') as f:
            json.dump(setting, f, indent=4)
            print('Setting was saved in the directory: ' + self.path_saving)

    def set_final_weights_as_init_weights(self, input_path, output_path, json_synapses):
        """
        Sets the final weights as the initial weights.

        Parameters
        ----------
        input_path : str
            the path to read file
        output_path : str
            the path to save file
        json_synapses : object
            the synapses in .json file
        """
        self.path_settings = input_path
        self.path_saving = output_path
        synapses = self.load_synapses(json_synapses)
        for sec in synapses:
            for syn in synapses[sec]:
                if syn['receptor'] == 'AMPA':
                    syn['init_weight'] = syn['final_weight']
                    syn['final_weight'] = None

        with open(self.path_saving + 'synapses.json', 'w') as f:
            json.dump(synapses, f, indent=4)
            print('Synapses were saved in the directory: ' + self.path_saving)
