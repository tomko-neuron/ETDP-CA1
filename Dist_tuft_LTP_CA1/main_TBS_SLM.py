"""
Creates CA1 pyramidal cell model, sets parameters, runs simulation in NEURON and shows figures.

Title: main_TBS_SLM.py
Author: Matus Tomko
Mail: matus.tomko __at__ savba.sk
"""
import libcell as lb
from neuron import h, gui
from CA1_plasticity.io.figures_shower import FiguresShower
from CA1_plasticity.io.io_helper import IOHelper


def prepare_model():
    """
    Creates an instance of the model and sets the model.

    Returns
    -------
    CA1PyramidalCell
        an instance of the CA1PyramidalCell class
    """
    CA1_cell = lb.CA1(setting=setting)
    CA1_cell.add_synapses_distTuf()
    CA1_cell.connect_ns_terminals()
    # CA1_cell.insert_current_clamp(section=CA1_cell.soma, x=0.5)
    # CA1_cell.insert_SEClamp(section=CA1_cell.soma, x=0.5)
    CA1_cell.set_theta_burst()
    # CA1_cell.set_theta_burst_iclamp(stim=CA1_cell.stim)
    CA1_cell.set_recording_vectors_dist_tuft()
    # CA1_cell.apply_TTX()

    return CA1_cell


def set_simulation_settings():
    """Sets the simlation settings."""
    h.tstop = setting['simulation']['TSTOP']
    h.dt = setting['simulation']['DT']
    h.v_init = -70.0
    h.celsius = 35
    h.finitialize(h.v_init)
    h.fcurrent()
    h.cvode_active(0)


def run_full_simulation(cell):
    """
    Runs a full simulation.
    """

    h.continuerun(setting['simulation']['TSTOP'])
    ioh.save_recordings(synapses=cell.synapses, tw_vec=cell.t_rs_vec, v_soma_vec=cell.v_vec,
                        t_vec=cell.t_vec, p_vec=cell.p_vec, d_vec=cell.d_vec, ta_vec=cell.t_rs_vec,
                        alpha_scount_vec=cell.alpha_scout_vec, dend_vecs=cell.dend_vecs,
                        apc_vec=cell.apc_vec, cai_vecs=cell.cai_vecs, cal_ica_vecs=cell.calH_ica_vecs,
                        nmda_ica_vecs=cell.nmda_ica_vecs, ina_vecs=cell.ina_vecs, pmp_vecs=cell.pmp_vecs,
                        ogb_vecs=cell.ogb_vecs, spines_v_vecs=cell.spines_v_vecs)
    ioh.save_setting(setting=setting)
    ioh.save_synapses(synapses=cell.synapses)


def run_segmented_simulation(cell):
    """
    Runs a segmented simulation.
    """
    t_stop = 0

    for s in range(setting['simulation']['NUM_SEGMENTS']):
        t_stop = t_stop + setting['simulation']['SEG_DURATION']
        h.continuerun(t_stop)
        ioh.save_segment_recordings(segment=s, synapses=cell.synapses, tw_vec=cell.t_rs_vec, v_soma_vec=cell.v_vec,
                                    t_vec=cell.t_vec, p_vec=cell.p_vec, d_vec=cell.d_vec, ta_vec=cell.t_rs_vec,
                                    alpha_scount_vec=cell.alpha_scout_vec, dend_vecs=cell.dend_vecs,
                                    apc_vec=cell.apc_vec, cai_vecs=cell.cai_vecs, cal_ica_vecs=cell.calH_ica_vecs,
                                    nmda_ica_vecs=cell.nmda_ica_vecs, ina_vecs=cell.ina_vecs, pmp_vecs=cell.pmp_vecs,
                                    ogb_vecs=cell.ogb_vecs, spines_v_vecs=cell.spines_v_vecs)

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
                       path_saving='./recordings/no_inac/5stim_3xTBS/01/',
                       path_recordings='./recordings/no_inac/5stim_3xTBS/01/')
    fs.show_somatic_voltage()
    fs.show_dendritic_voltage(threshold=-37)
    fs.show_na_current()
    fs.show_nmda_ica_current()
    fs.show_cal_ica()
    fs.show_intracellular_calcium()
    fs.show_weights(synapse_type='perforated')
    fs.show_average_weights(secs=['dendA5_011111111111111110', 'dendA5_0111111111111111100'],
                            keys=['dendA5_011111111111111110', 'dendA5_0111111111111111100'])
    fs.show_average_weights_change(secs=['dendA5_011111111111111110', 'dendA5_0111111111111111100'],
                                   keys=['dendA5_011111111111111110', 'dendA5_0111111111111111100'], baseline=200)
    fs.show_weights_distance(start=200, stop=9999, synapse_type='perforated')


if __name__ == '__main__':
    ioh = IOHelper(path_saving='./recordings/no_inac/5stim_3xTBS/01/',
                   path_settings='./settings_distLTP/')
    try:
        setting = ioh.load_setting()
    except Exception as e:
        print(e)
        print('Loading setting failed. Program terminated.')
    cell = prepare_model()
    set_simulation_settings()
    # run_segmented_simulation(cell=cell)
    ioh.delete_temporary_files()
    # figures()
