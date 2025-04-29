# SLEAP Optuna environment setup

SLEAP is made to work with python 3.7, but Optuna works with python 3.8, causing dependency issues. People have however successfully installed and used SLEAP with python 3.8, and the instructions to get this working on the HPC are as follows:
1. Create SLEAP python 3.8 environment:

`mamba create --name sleap_optuna pip python=3.8 cudatoolkit=11.3 cudnn=8.2`

`pip install 'sleap[pypi]==1.3.3'`

2. You may need to modify my LD_LIBRARY_PATH so that tensorflow can find `libcudart.so.11.0` dynamic library (might not be necessary for everyone, unsure). Run the following command:

`mkdir -p ~/.conda/envs/sleap_optuna/etc/conda/activate.d`

Navigate to this dir and make a env_vars.sh file. Put the following lines in it:
```
#!/bin/bash  
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH
```
3. Install Optuna and the other packages necessary for the parameter sweeping:

`pip install optuna`

`pip install datasette`

`pip install submitit`

4. Navigate to your conda environment and then `lib/python3.8/site-packages/sleap/nn/training.py`. Comment out line 953 self.`evaluate()`. This is not strictly necessary but will speed up your SLEAP optuna trials and avoid unwanted errors.