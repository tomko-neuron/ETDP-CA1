"""
Creates CA1 pyramidal cell model, sets parameters, runs simulation in NEURON and shows figures.

Title: main.py
Author: Matus Tomko
Mail: matus.tomko __at__ savba.sk
"""
import time

from neuron import h, gui
import libcell as lb
import saveClass as sc
import numpy as np


from CA1_plasticity.io.figures_shower import FiguresShower
from CA1_plasticity.io.io_helper import IOHelper
from CA1_plasticity.model.CA1_pyramidal_cell import CA1PyramidalCell

exec(open('./sim_functs.py').read())


def prepare_model(data):
    """
    Creates an instance of the model and sets the model.

    Returns
    -------
    CA1PyramidalCell
        an instance of the CA1PyramidalCell class
    """
    CA1_cell = lb.CA1(hoc_model='./CA1.hoc', path_mods='./mods/', setting=setting)
    lb.init_activeCA1(model=CA1_cell, soma=data.ACTIVE_soma, dend=data.ACTIVE_dend)
    CA1_cell.print_dend_params(dend_name='dend2_01001')
    data.Elocs = lb.genDendLocs(data.locDend, 3, [0.96, 0.98])
    data.Ensyn = 3
    CA1_cell.add_synapses_on_spines(data=data, dend_name='dend2_01001')
    CA1_cell.connect_ns_terminals()
    CA1_cell.set_Makara_LTP(num_synapses=4, start=500, interval=2000, num=6, protocol=False)
    CA1_cell.set_Makara_LTP(num_synapses=3, start=12000, interval=333.33, num=50, protocol=True)
    CA1_cell.set_Makara_LTP(num_synapses=4, start=29000, interval=2000, num=6, protocol=False)
    CA1_cell.set_dendritic_voltage_vectors(sections=['dend2_01001'])

    return CA1_cell


def set_init_parameters():
    # ----------------------------------------------------------------------------
    # Data saving object; # Data storage lists
    data = sc.EmptyObject()
    data.vdata, data.vDdata, data.Gdata, data.Idata, data.stim = [], [], [], [], []

    PAR1 = 3
    PAR2 = 1

    # ----------------------------------------------------------------------------
    # Simulation CONTROL
    syntypes = ['alltree', 'local_regular', 'global_regular_lrand', 'global_regular_lreg']

    data.model = 'CA1'  # L23 or CA1
    data.stimType = 'LTP'  # poisson, place, replay, minis, nIter, SStim, DStim, LTP
    data.synType = 'single'  # synapse distribution - NULL, single, alltree, clustered, clust2, local_regular, global_regular_lrand, global_regular_lreg
    data.locBias = 'distal'
    # data.synType = syntypes[PAR1-1] # synapse distribution
    data.actType = 'active'  # passive, aSoma, aDend, active
    if data.model == 'L23' and data.stimType == 'replay' and data.synType == 'clust2':
        data.synType = 'clust3'

    # 1min 20s / 10s simulation for active
    # 30s / 10s simulation for passive
    # 60s / 10s simulation for passive spines
    data.modulateK = False  # True or False, global modulation, similar to Losonczy 2006
    data.modulateK_parents = False  # True or False
    data.modulateK_local = False  # True or False
    data.removeIspikes = 0  # 0-1. The probability of removing an inhibitory spike
    data.modulateRmRa = False  # True or False
    data.modulateRmRaSeg = False  # True or False
    data.randomW = False  # True or False
    data.measureInputRes = False
    data.iden = 75  # dend5_00
    data.randClust = False  # randomize location of clustered synapses within the branch

    data.constNMDA = False  # if TRUE we use voltage independent NMDA receptors
    data.modulateNa = False  # True or False - switch off the Na in the branches

    # only when data.stimType = 'place'
    data.placeType = 'balanced'  # balanced, random_N (number of neurons), random_NR (number and peak firing rate)
    if data.placeType == 'balanced':
        data.randomW = False

    # only when data.stimType = 'nIter'
    data.direction = 'IN'  # 'OUT' or 'IN' only for nsyn - direction of stimulation sequence

    data.AHS = False  # active hotspots
    # only when data.stimType = 'place'
    data.stimseed = PAR2  # PAR2 - 1-10

    # ---------------------------------------------------------------------------
    data.SAVE = True
    data.SHOWTRACES = True
    data.SHOWSYNS = True

    # number of iterations - only when data.stimType = 'place'
    # this corresponds to different trials with identical synapses
    data.nIter = 3  # max is usually 16
    # time parameters
    data.TSTOP = 10
    if data.model == 'L23':
        data.TSTOP = 24
    if data.model == 'CA1' and data.stimType == 'replay':
        data.TSTOP = 0.3

    # ----------------------------------------------------------------------------
    # synapse parameters
    # ----------------------------------------------------------------------------
    data.NMDA = True
    data.GABA = False
    data.GABA_B = False

    data.SPINES = True
    data.Lmin = 60
    weight_factor_A = 2  # multiplicative weight scale of the clustered synapses - AMPA
    weight_factor_N = 2  # multiplicative weight scale of the clustered synapses - NMDA
    data.g_factor = 1  # all synapses are scaled with this factor

    # cluster parameters - only when data.stimType = 'place'
    # clust: 240 = 1x240 = 4x60 = 12x20 = 48x5 = 240x1
    # clust2: 240 = 4x60 = 8x30 = 12x20 = 24x10 = 48x5 = 120*2 = 240x1
    Nclusts = np.array([4, 8, 12, 24, 48, 120, 240])
    Nsyn_per_clusts = np.array([60, 30, 20, 10, 5, 2, 1])
    # Nclusts = np.array([500, 1000, 2000]) # use this only if all synapses are clustered in CA1
    # Nclusts = np.array([480, 960, 1920]) # use this only if all synapses are clustered in L23
    # Nsyn_per_clusts = np.array([4, 2, 1])
    # data.Lmin = 5

    Nclust = Nclusts[PAR1 - 1]
    Ncell_per_clust = Nsyn_per_clusts[PAR1 - 1]

    if data.stimType == 'minis':
        Nclust = 10
        Ncell_per_clust = 1

    mazeCenter = True
    if mazeCenter:
        inmazetype = 'midMaze'
    else:
        inmazetype = 'randMaze'
    exec(open('./init_params.py').read())
    np.random.seed(data.stimseed)
    return data


def set_simulation_settings():
    """Sets the simlation settings."""
    h.tstop = setting['simulation']['TSTOP']
    h.dt = setting['simulation']['DT']
    h.v_init = -68.3 # -65
    h.celsius = 35
    h.finitialize(h.v_init)
    h.fcurrent()
    h.cvode_active(0)


def run_full_simulation(cell: CA1PyramidalCell):
    """
    Runs a full simulation.

    Parameters
    ----------
    cell : CA1PyramidalCell
        the object of CA1PyramidalCell class
    """
    h.continuerun(setting['simulation']['TSTOP'])
    ioh.save_recordings(synapses=cell.synapses, tw_vec=cell.t_rs_vec, v_soma_vec=cell.v_vec,
                        t_vec=cell.t_vec, p_vec=cell.p_vec, d_vec=cell.d_vec, ta_vec=cell.t_rs_vec,
                        alpha_scount_vec=cell.alpha_scout_vec, dend_vecs=cell.dend_vecs,
                        apc_vec=cell.apc_vec, cai_vecs=cell.cai_vecs, cal_ica_vecs=cell.cal2_ica_vecs,
                        nmda_ica_vecs=cell.nmda_ica_vecs, ina_vecs=cell.ina_vecs, spines_v_vecs=cell.spines_v_vecs,
                        pmp_vecs=[], ogb_vecs=[])
    ioh.save_setting(setting=setting)
    ioh.save_synapses(synapses=cell.synapses)


def run_segmented_simulation(cell: CA1PyramidalCell):
    """
    Runs a segmented simulation.

    Parameters
    ----------
    cell : CA1PyramidalCell
        the object of CA1PyramidalCell class
    """
    t_stop = 0

    for s in range(setting['simulation']['NUM_SEGMENTS']):
        t_stop = t_stop + setting['simulation']['SEG_DURATION']
        h.continuerun(t_stop)
        ioh.save_segment_recordings(segment=s, v_soma_vec=cell.v_vec, t_vec=cell.t_vec, apc_vec=cell.apc_vec,
                                    dend_vecs=cell.dend_vecs, ta_vec=cell.t_rs_vec, d_vec=cell.d_vec, p_vec=cell.p_vec,
                                    alpha_scount_vec=cell.alpha_scout_vec, synapses=cell.synapses,
                                    tw_vec=cell.t_rs_vec, cai_vecs=cell.cai_vecs, cal_ica_vecs=cell.cal2_ica_vecs,
                                    nmda_ica_vecs=cell.nmda_ica_vecs, ina_vecs=cell.ina_vecs,
                                    spines_v_vecs=cell.spines_v_vecs, pmp_vecs=[], ogb_vecs=[])

        if s < setting['simulation']['NUM_SEGMENTS'] - 1:
            cell.reset_recording_vectors()
    ioh.reconstruct_segmented_bcm()
    ioh.reconstruct_segmented_currents()
    ioh.reconstruct_segmented_synapses()
    ioh.reconstruct_segmented_voltages()
    ioh.save_setting(setting=setting)
    ioh.save_synapses(synapses=cell.synapses)


def figures():
    """Shows selected figures."""
    fs = FiguresShower(setting=setting, save_figures=False,
                       path_saving='./recordings/dend2_01001/3_synapses_distal',
                       path_recordings='./recordings/dend2_01001/3_synapses_distal/')
    fs.show_somatic_voltage()
    fs.show_dendritic_voltage(threshold=-37)
    fs.show_input_spikes()
    fs.show_voltage_on_spines(threshold=-37)
    fs.show_weights(synapse_type='perforated')
    fs.show_average_weights_change(secs=['dend2_01001'],
                                   keys=['dend2_01001'], baseline=12000)
    fs.show_weights_distance(start=12000, stop=29000, synapse_type='perforated')


if __name__ == '__main__':
    dends = [['dend1_00', 1], ['dend2_01001', 21], ['dend2_010100', 24], ['dend3_0100', 56], ['dend3_0101', 57],
             ['dend4_010', 64], ['dend5_00', 75]]

    ioh = IOHelper(path_saving='./recordings/dend2_01001/8_synapses_distal/',
                   path_settings='./settings/')
    try:
        setting = ioh.load_setting()
    except Exception as e:
        print(e)
        print('Loading setting failed. Program terminated.')
    data = set_init_parameters()
    data.iden = 21
    cell = prepare_model(data)
    set_simulation_settings()
    # run_segmented_simulation(cell=cell)
    ioh.delete_temporary_files()
    # figures()
