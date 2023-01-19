# Fast Autonomous Scanning Toolkit (FAST)

This repository contains the code for the Fast Autonomous Scanning Toolkit (FAST) reported 
in the manuscript: 
"Demonstration of an AI-driven workflow for autonomous high-resolution scanning microscopy"


The preprint of the manuscript is hosted at https://arxiv.org/abs/2301.05286

### Code details:

At the backend, FAST adapts the SLADS-Net [^1] and SLADS [^2] algorithms from the papers and from the
repositories of the paper authors. In addition, FAST also contains code specific to
the adoption at the Sector 26 beamline of APS. The code breakdown is as follows:

- The `fast/core` submodule contains most of the SLADS-Net algorithm and the neural network training procedure. This submodule can be used for simulations without requiring any of the experiment-specific code.
- The `fast/s26_runners` submodule contains the code required to run the module at the S26 beamline. For this woek, we use tkinter and Epics to manage the communication between an edge device and a beamline computer. It is not too difficult to replace this with other modes of comunication (e.g. monitoring a file). Please contact the authors if help is needed for this.
- The `fast/s26_analysis` submodule contains routines for simulation and analysis corresponding to S26. 
- The `input_params.py` file contains default input settings for simulation, experiment, training, etc.

[^1]: Zhang, Yan, G. M. Dilshan Godaliyadda, Nicola Ferrier, Emine B. Gulsoy, Charles A. Bouman, and Charudatta Phatak. “SLADS-Net: Supervised Learning Approach for Dynamic Sampling Using Deep Neural Networks.” Electronic Imaging 30, no. 15 (January 28, 2018): 131-1–1316. https://doi.org/10.2352/ISSN.2470-1173.2018.15.COIMG-131.  
[^2]: Godaliyadda, G. M. Dilshan P., Dong Hye Ye, Michael D. Uchic, Michael A. Groeber, Gregery T. Buzzard, and Charles A. Bouman. “A Framework for Dynamic Image Sampling Based on Supervised Learning.” IEEE Transactions on Computational Imaging 4, no. 1 (March 2018): 1–16. https://doi.org/10.1109/TCI.2017.2777482.


### Code examples
Examples for training and numerical simulation using the FAST code are located in `work_directory`. 
Specifically:

- `work_directory/training/training_cameraman.ipynb` contains the code used for training the NN using the cameraman image. While this jupyter notebook shows the procedure used to optimize the SLADS hyperparameter $c$, only the optimal model (corresponding to $c=2$) is stored in this repository. The training data is regenerated within this notebook.
- `work_directory/test/simulate_fast.ipynb` contains the code for the FAST results in the numerical simulated presented in the manuscript. 
- The files `work_directory/test/simulate_full.ipynb` respectively contain the full simulation code --- including the raster grid and random sampling simulations --- and the analysis code used in the manuscript. This can be fairly computationally expensive and require a large amount of memory and storage.