# Multi-annotator Deep Learning: A Probabilistic Framework for Classification

Authors: Marek Herde, Denis Huseljic, and Bernhard Sick

## Project Structure
- [`evaluation`](/evaluation): collection of Python and Bash scripts required to perform experimental evaluation
- [`lfma`](/lfma): Python package consisting of several sub-packages
    - [`classifiers`](/lfma/classifiers): implementation of multi-annotator supervised learning techniques according to
      scikit-learn interfaces
    - [`modules`](/lfma/modules): implementation of multi-annotator supervised learning techniques as 
      [`pytorch_lightning`](https://www.pytorchlightning.ai/) modules, [`pytorch`](https://pytorch.org/) data sets, 
      and special layers
    - [`utils`](/lfma/utils): helper functions
- [`notebooks`](/notebooks):
  - [`annotator_simulation.ipynb`](/notebooks/annotator_simulation.ipynb): simulation of annotator sets for data sets 
    without annotations from real-world annotators
  - [`classification.ipynb`](/notebooks/classification.ipynb): visualization of MaDL's properties regarding the three 
    research questions in the accompanied article
  - [`data_set_creation_download.ipynb`](/notebooks/data_set_creation_download.ipynb): creation of artificial data sets
    and download of real-world data sets
  - [`evaluation.ipynb`](/notebooks/evaluation.ipynb): loading and presentation of experimental results
- [`requirements.txt`](requirements.txt): list of Python packages required to reproduce experiments 

## How to execute experiments?
To accommodate a large number of experiments, we utilized a computer cluster equipped with multiple V100 and A100 GPUs,
allowing us to execute many experiments simultaneously. Without access to such a cluster, reproducing all results 
presented in the accompanying article would take much more time.

In the following, we describe step-by-step how to execute all experiments presented in the accompanied article. 
As a prerequisite, we assume to have a Linux distribution as operating system and 
[`conda`](https://docs.conda.io/en/latest/) installed on your machine.

1. _Setup Python environment:_
```bash
projectpath$ conda create --name madl python=3.9
projectpath$ conda activate madl
```
First, we need to install `torch` with the build (1.13.1). For this purpose, we refer to 
[`pytorch`](https://pytorch.org/). An exemplary command for a Linux operating system would be:
```bash
projectpath$ pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```
Subsequently, we install the remaining requirements:
```bash
projectpath$ pip install -r requirements.txt
```
2. _Create and download data sets:_ Start jupyter-notebook and follow the instructions in the jupyter-notebook file
[`notebooks/data_set_creation_download.ipynb`](/notebooks/data_set_creation_download.ipynb).
```bash
projectpath$ conda activate madl
projectpath$ jupyter-notebook
```
3. _Simulate annotators:_ Start jupyter-notebook and follow the instructions in the jupyter-notebook file 
[`notebooks/annotator_simulation.ipynb`](/notebooks/annotator_simulation.ipynb).
```bash
projectpath$ conda activate madl
projectpath$ jupyter-notebook
```
4. _Create experiment scripts:_ We create a bash script for each triple of data set, annotator set, and multi-annotator 
supervised learning technique. Such a file contains the complete configuration of the respective experimental setup 
including hyperparameters. Depending on your machine or even compute cluster, you can optionally define your own SLURM 
setup. For this purpose, you need to define the corresponding variables in the 
[`evaluation/create_experiment_scripts.py`](evaluation/create_experiment_scripts.py) accordingly. For example, you can adjust the 
amount of RAM allocated for a run of an experiment. Once, you have adjusted the variables in this script, you may 
execute it via:
```bash
projectpath$ conda activate madl
projectpath$ python evaluation/create_experiment_scripts.py
```
5. _Execute experiment scripts:_ After the creation of the experiment scripts, there will be a folder for each
multi-annotator supervised learning technique we want to evaluate. For example, the file 
`evaluation/madl/madl_cifar10_none.py` corresponds to evaluating MaDL on CIFAR10 with independent annotators. Such a 
file consists of multiple commands executing the file [`evaluation/run_experiment.py`](evaluation/run_experiment.py) with 
different configurations. For a better understanding of these possible configurations, we refer to the explanations in 
the file [`evaluation/run_experiment.py`](evaluation/run_experiment.py). If you  have disabled the use of 
[`slurm`](https://slurm.schedmd.com/documentation.html), you can now execute such a `bash` script via:
```bash
projectpath$ conda activate madl
projectpath$ ./evaluation/madl/madl_cifar10_none.sh
```
Otherwise, you need to use the `sbatch` command:
```bash
projectpath$ conda activate madl
projectpath$ sbatch ./evaluation/madl/madl_cifar10_none.sh
```

## How to investigate the experimental results?
Once, an experiment is completed, its associated results are saved as a `.csv` file at the directory specified by 
`evaluation.run_experiment.RESULT_PATH`. For getting a tabular and summarized presentation of these results, you need 
to start jupyter-notebook and follow the instructions in the jupyter-notebook file 
[`notebooks/evaluation.ipynb`](notebooks/evaluation.ipynb).
```bash
projectpath$ conda activate madl
projectpath$ jupyter-notebook
```

## How to reproduce the visualizations?
Start jupyter-notebook and follow the instructions in the jupyter-notebook file 
[`notebooks/classification.ipynb`](notebooks/classification.ipynb).
```bash
projectpath$ conda activate madl
projectpath$ jupyter-notebook
```

## Citing
If you use ``multi-annotator-deep-learning`` in one of your research projects or would like to reference the 
accompanied article, please use the following:

```
@article{
    herde2023multiannotator,
    title={Multi-annotator Deep Learning: A Probabilistic Framework for Classification},
    author={Marek Herde and Denis Huseljic and Bernhard Sick},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2023},
    url={https://openreview.net/forum?id=MgdoxzImlK},
}
```
