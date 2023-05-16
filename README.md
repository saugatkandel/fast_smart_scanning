# Fast Autonomous Scanning Toolkit (FAST)

This repository contains the code for the Fast Autonomous Scanning Toolkit (FAST) reported 
in the manuscript: 
"Demonstration of an AI-driven workflow for autonomous high-resolution scanning microscopy"


The preprint of the manuscript is hosted at https://arxiv.org/abs/2301.05286

---
- [Fast Autonomous Scanning Toolkit (FAST)](#fast-autonomous-scanning-toolkit-fast)
    - [Code details](#code-details)
      - [Added 04/02/23](#added-040223)
    - [Code examples](#code-examples)
      - [Added 04/02/23:](#added-040223-1)
    - [Installation](#installation)
      - [Added 04/02/23:](#added-040223-2)
    - [Use](#use)
    - [Paper data](#paper-data)
    - [GUI Prototype (Added 05/05/23):](#gui-prototype-added-050523)

---

### Code details

At the backend, FAST adapts the SLADS-Net [^1] and SLADS [^2] algorithms from the papers and from the
repositories of the paper authors. In addition, FAST also contains code specific to
the adoption at the Sector 26 beamline of APS. The code breakdown is as follows:

- The `fast/core` submodule contains most of the SLADS-Net algorithm and the neural network training procedure. This submodule can be used for simulations without requiring any of the experiment-specific code.
- The `fast/s26_runners` submodule contains the code required to run the module at the S26 beamline. For this woek, we use tkinter and Epics to manage the communication between an edge device and a beamline computer. It is not too difficult to replace this with other modes of comunication (e.g. monitoring a file). Please contact the authors if help is needed for this.
- The `fast/s26_analysis` submodule contains routines for simulation and analysis corresponding to S26. 
- The `input_params.py` file contains default input settings for simulation, experiment, training, etc.

#### Added 04/02/23

- The new version of the code has been modified to use the Sobol, Halton, or Latin-hypercube initializations using the Scipy package instead of the Hammersly initialization using the Scikit-optimize package. This is because the Scikit-optimize package has not been updated to support `numpy>=1.24`. If the Hammersly initialization is desired, then the Scikit-image package needs to be modified manually using the instructions here: https://github.com/scikit-optimize/scikit-optimize/issues/1147.
- The new version of the code also contains an option to generate the scan pattern based on expected feature size, to ensure that `>99%` of image patches of the provided size are sampled during the initial scan. This function, which is in `fast/utils/generate_scan_pattern.py`, has not been included in any of the example notebooks just yet. 

[^1]: Zhang, Yan, G. M. Dilshan Godaliyadda, Nicola Ferrier, Emine B. Gulsoy, Charles A. Bouman, and Charudatta Phatak. “SLADS-Net: Supervised Learning Approach for Dynamic Sampling Using Deep Neural Networks.” Electronic Imaging 30, no. 15 (January 28, 2018): 131-1–1316. https://doi.org/10.2352/ISSN.2470-1173.2018.15.COIMG-131.  
[^2]: Godaliyadda, G. M. Dilshan P., Dong Hye Ye, Michael D. Uchic, Michael A. Groeber, Gregery T. Buzzard, and Charles A. Bouman. “A Framework for Dynamic Image Sampling Based on Supervised Learning.” IEEE Transactions on Computational Imaging 4, no. 1 (March 2018): 1–16. https://doi.org/10.1109/TCI.2017.2777482.


---

### Code examples
Examples for training and numerical simulation using the FAST code are located in `work_directory`. 
Specifically:

- `work_directory/training/training_cameraman.ipynb` contains the code used for training the NN using the cameraman image. While this jupyter notebook shows the procedure used to optimize the SLADS hyperparameter $c$, only the optimal model (corresponding to $c=2$) is stored in this repository. The training data is regenerated within this notebook.
- `work_directory/other_examples/simulate_generic_image.ipynb` contains a simulation for a generic coffee cup image. 
- `work_directory/sims_paper/simulate_fast.ipynb` contains the code for the FAST results in the numerical simulated presented in the manuscript. 
- The files `work_directory/sims_paper/simulate_full.ipynb`and  `work_directory/test/comparisons_full.ipynb`respectively contain the full simulation code --- including the raster grid and random sampling simulations --- and the analysis code used in the manuscript. This can be fairly computationally expensive and require a large amount of memory and storage.

#### Added 04/02/23:
- `work_directory/training_usc_sipi/` contains an notebooks that generate a trained model with a number of miscellaneous images from the USC-SIPI and Scikit-image databases. It also contains notebooks to analyze the statistics of these images, and to test their performance on the numerical simulation of the WSe2 flake used in the manuscript. 
-  `work_directory/training_usc_sipi/` and `work_directory/shepp_logan/` now contains some experiments with the Shepp-Logan phantom as well. The Shepp-Logan phantom is generated on demand using the `odl` (https://odlgroup.github.io/odl/odl.html) package. These examples require either the `odl` package installed separrately or using the command in the `Installation` section below.

The new examples have not been documented properly just yet.

### Installation

We recommend creating a new environment (through conda, pipenv, etc) to try out FAST. The conda command might look like:
```shell
conda create -n fast_test python=3.10
conda activate fast_test
conda config --prepend channels conda-forge
conda install pip
```
We recommend using Python 3.10 because PyPi does not currently contain a scikit-image wheel for Python 3.11 for M1 Macs. One alternative is to install scikit-image through conda instead. We have not tried the current version of the code in other operating systems, but we do not expect any issues.

The next step would be to download the code (and unzip it if necessary) to the desired location.  Then, enter the directory and run:
```shell
pip install .
```
or, if we want an "editable" installation where one can modify the source code and continue using the package without any reinstall. 
```shell
pip install -e .
```
This should install all the dependencies required for the project. Optionally, expert users can install the requirements listed in the `pyproject.toml` file directly from conda or elsewhere.

Jupyter notebook or jupyter lab is also required to run the `ipynb` notebooks supplied as examples. To install jupyter lab:
```shell
conda install jupyterlab
```
To run jupyterlab
```shell
jupyter lab
```
Then navigate to the appropriate notebook.


#### Added 04/02/23:

To try the Shepp-Logan phantom examples,  install the package using the `odl` optional dependency:
```shell
pip install -e .[odl]
``` 

### Use

The jupyter notebooks in `work_directory` contain the example codes. To use the API within a python script, 
we can follow the same code pattern demonstrated in the jupyter notebooks.

### Paper data

The data used in the paper wil be added to the respository in the near future. In the meantime, it can be provided upon reasonable request. 


### GUI Prototype (Added 05/05/23):

To try the prototype UI, check out the `gui_prototype` branch and install the package using the `gui_protoype` optional dependency:
```shell
pip install -e .[gui_prototype]
```

Then run 
```shell
python work_directory/plotly_test.py
```

This should generate an output like:\
`Dash is running on http://127.0.0.1:8050/`

Copy the url generated to the browser.