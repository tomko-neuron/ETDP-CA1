# ETDP-CA1
This repository contains the code for the models used in "A voltage-based Event-Timing-Dependent Plasticity rule accounts for LTP subthreshold and suprathreshold for dendritic spikes in CA1 pyramidal neurons " Tomko et al. (2023).

To run:

```
git clone https://github.com/tomko-neuron/ETDP-CA1.git
```

- install packages in [requirements](./requirements.txt)

## CA1 pyramidal cell models
Two compartmental models of CA1 pyramidal cells are used:
- CA1 pyramidal neuron: Dendritic Na+ spikes are required for LTP at distal synapses ​[(Kim et al. (2015)](https://modeldb.science/184054)
- Global and multiplexed dendritic computations under in vivo-like conditions [(Ujfalussy et al 2018)](https://modeldb.science/265511)

# Synaptic plasticity experiments
There are two defined experiments on synaptic plasticity. In all experiments, the voltage-based ETDP synaptic plasticity rule described
in [Benuskova and Abraham (2007)](https://link.springer.com/article/10.1007/s10827-006-0002-x) was employed.
- **Dist_tuf_LTP_CA1** implements four theta-burst stimulation protocols as in [Kim et al. (2015)](https://doi.org/10.7554/eLife.06414).
- **Suthresold_LTP_CA1** implements four low-frequency stimulation protocols as in [Magó et al. (2020)](https://doi.org/10.1523/jneurosci.2071-19.2020).

# Code organization

`CA1_plasticity`

This directory contains the classes and files needed to work with models, run simulations, save records and plot figures.

`Dist_tuft_LTP_CA1`

The directory includes files and directories from [ModelDB](https://modeldb.science/184054) necessary for running the model, along with additional resources:
- `libcell.py`: This file contains the `CA1` class, representing the model.
- `main.py`: This script is used to run simulations and plot figures.
- `figures` subdirectory: It contains saved figures used in the paper.
- `recordings` subdirectory: This contains saved records (not included).
- `settings` subdirectory: Here, you'll find `setting.json` and `synapses.json` files required for configuring the simulation and adding synapses to the model.

`Subthreshold_LTP_CA1`

The directory includes files and directories from [ModelDB](https://modeldb.science/265511) necessary for running the model, along with additional resources:
- `libcell.py`: This file contains the `CA1` class, representing the model.
- `main.py`: This script is used to run simulations and plot figures.
- `figures` subdirectory: It contains saved figures used in the paper.
- `recordings` subdirectory: This contains saved records (not included).
- `settings` subdirectory: Here, you'll find `setting.json` and `synapses.json` files required for configuring the simulation and adding synapses to the model.

# Typical workflow

A typical workflow consists of four main stages:

1. configuration
2. model instantiation
3. simulation
4. saving and plot figures
