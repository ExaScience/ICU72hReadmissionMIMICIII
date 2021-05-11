# Machine Learning in the ICU: Blocking factors and quantified needs
- - -  

We are releasing this repository to make it possible to replicate our work, and in case it is useful for further work in this area. If you are using any part of this code repository that we have added to, we would appreciate if you cite our paper: (reference to follow afer review)

This repository is released under an MIT license. It builds on a previous repository that was also released under an MIT license which can be found [here](https://github.com/apakbin94/ICU72hReadmissionMIMICIII). If you use any of the code from the underlying repository, you should also reference the paper associated with that repository (see the file *original_README.md*).


## Running experiments

1. Establish an appropriate conda environment (and then activate it):

       conda create --name some_name --file conda/my_conda_env.yml
       conda activate some_name


2. Prepare *df_MASTER_DATA.csv*:

Follow the steps for Phase 1 from *original_README.md*


3. Run the experiments from this repository:

The top level script and configuration file in this repository are:

       models1/experiments.py
       models1/experiments.yml

Go to the *models1/* directory. First fill in the *experiments.yml* file appropriately and then launch the experiments with:

       python experiments.py


