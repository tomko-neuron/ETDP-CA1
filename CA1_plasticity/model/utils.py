"""
Title: utils.py
Author: Matus Tomko
Mail: matus.tomko __at__ fmph.uniba.sk
"""


class RecordingVector:
    """
    A class used to represent a vector for recording voltage or current

    ...

    Attributes
    ----------
    section : str
        the section name
    segment_x : float
        the recording site on the section
    vector : neuron.hoc.HocObject
        the vector for recording voltage or current
    """

    def __init__(self, section, segment_x, vec):
        """
        Parameters
        ----------
        section : str
            the section name
        segment_x : float
            the recording site on the section
        vec : neuron.hoc.HocObject
            the vector for recording voltage or current
        """

        self.section = section
        self.segment_x = segment_x
        self.vector = vec


class Synapse:
    """
    A class used to represent a Synapse

    ...

    Attributes
    ----------
    synapse : neuron.hoc.HocObject
        the HocObject of the synapse
    synapse_id : int
        the synapse identificator
    section : str
            the section name
    segment_x : float
        the position of the synapse on the section
    distance : float
        the distance of the synapse from the soma
    init_weight : float
        the initial weight of the synapse
    weight_vec : neuron.hoc.HocObject
        the vector for recording the synaptic weight over time
    input_spikes_vec : neuron.hoc.HocObject
        the vector for recording the times of the spikes arriving to the synapse
    stimulated : bool, optional
        the attribute determining the application of the stimulation protocol to the synapse (default False)
    receptor : str
        the receptor of the synapse (AMPA, NMDA, or None)
    pathway : str, optional
        the hippocampal pathway representing by the synapse (COM, SCH, LM, or None)
    type : str, optional
        the type of the synapse (nonperforated, perforated)
    ns_terminal : str
            the single pulse NetStim connected to the synapse
    """

    def __init__(self, synapse, synapse_id, section, segment_x, distance, init_weight, weight_vec, input_spikes_vec,
                 receptor, type, d_amp_vec, p_amp_vec):
        """
        Parameters
        ----------
        synapse : neuron.hoc.HocObject
            the HocObject of the synapse
        synapse_id: int
            the synapse identificator
        section : str
            the section name
        segment_x : float
            the position of the synapse on the section
        distance : float
            the distance of the synapse from the soma
        init_weight : float
            the initial weight of the synapse
        weight_vec : neuron.hoc.HocObject
            the vector for recording the synaptic weight over time
        input_spikes_vec : neuron.hoc.HocObject
            the vector for recording the times of the spikes arriving to the synapse
        receptor : str
            the receptor of the synapse (AMPA, NMDA, or None)
        type : str
            the type of the synapse (nonperforated, perforated)
        d_amp_vec : neuron.hoc.HocObject
            the vector for recording the depression amplitude over time
        p_amp_vec : neuron.hoc.HocObject
            the vector for recording the potentiation amplitude over time
        """

        self.synapse = synapse
        self.synapse_id = synapse_id
        self.section = section
        self.segment_x = segment_x
        self.distance = distance
        self.init_weight = init_weight
        self.weight_vec = weight_vec
        self.input_spikes = input_spikes_vec
        self.stimulated = False
        self.receptor = receptor
        self.pathway = None
        self.type = type
        self.ns_terminal = None
        self.d_amp_vec = d_amp_vec
        self.p_amp_vec = p_amp_vec
