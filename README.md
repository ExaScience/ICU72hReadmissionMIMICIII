# Machine Learning in the ICU: Blocking factors and quantified needs
- - -  

We are releasing this repository to make it possible to replicate our work, and in case it is useful for further work in this area. If you are using any part of this code repository that we have added to, we would appreciate if you cite our paper: (reference to follow afer review)

This repository is released under an MIT license. It builds on a previous repository that was also released under an MIT license which can be found [here](https://github.com/apakbin94/ICU72hReadmissionMIMICIII). If you use any of the code from the underlying repository, you should also reference the paper associated with that repository (see the file *original_README.md*).

## System Requirements

### Dependencies
This repository requires conda to manage and install the many Python package dependencies required to run the experiments. Installing the dependencies using the conda environment file in the repository is detailed below.

In theory this repo is (mostly) platform agnostic through its use of Python and conda. In practice, it has only been tested on a relatively recent Linux system (5.4.0-96-generic #109-Ubuntu SMP), using conda 4.12.0. There is no requirement for non-standard hardware.

## Installation

### Dataset
A small demo data set for testing the code is available here:
https://doi.org/10.13026/C2HM2Q

The main MIMIC-3 dataset is accessible after passing the necessary data handling training. More information can be found here:
https://mimic.mit.edu/docs/iii/

### Code and Data preparation

1. Establish an appropriate conda environment (and then activate it):

       conda create --name some_name --file conda/my_conda_env.yml
       conda activate some_name

    Expected install time is < 10 minutes on a reasonably modern system.

2. Prepare *df_MASTER_DATA.csv*:

    Follow the steps for Phase 1 from *original_README.md* and apply to the demo or main dataset


## Running experiments


The top level script and configuration file in this repository are:

        models1/experiments.py
        models1/experiments.yml

1. Go to the *models1/* directory. First fill in the *experiments.yml* file appropriately and then launch the experiments with:

        python experiments.py

    Expected run time varies a large amount depending on the system being used and the experiments specified. The longer running experiments can easily take > 10 hours.


2. (Optional) post process the experiments on percentage data splits by running:

        python icu72hra_collectFigs.py experiments.yml


### Expected output

The main expected output (on either the demo or main dataset) is a file called AUC_STATS.txt that gives information about the achieved ROCAUCs on the test sets for each sub-model. The optional post-processing file produces a plot to show how the performance varies as the data set training percentage varies.

Auxilliary output on either the demo or main dataset is a set of directories, one per model type. Within these directories, there are various files relating to data for and performance of models in individual folds.
