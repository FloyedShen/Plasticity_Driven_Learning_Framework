# Code for Learning the Plasticity: Plasticity-Driven Learning Framework in Spiking Neural Networks

## Requirements

For working memory and reinforcement learning experiments:

* jax >= 0.49
* tensorboard 
* timm >= 0.6.12

In addition, you will need to install the lite versions of ```evojax``` and ```brax``` in this path, 
some of which are required to support the changes made to the base components in the experiments in this manuscript:

```bash
     cd Dependencies 
     pip install evojax 
     pip install brax 
```

For reproducing the figures in the manuscript:

* jax >= 0.49
* numpy 
* matplotlib
* seaborn
* mediapy 

## Usage

### Working Memory Experiment

* MetaPlasticity: 

```bash 
    cd examples 
    python repeated_seq_learning.py --policy BatchedGruMetaStdpMLPPolicy
```

Direct training weights:

```bash 
    cd examples 
    python repeated_seq_learning.py --policy BatchedGruMLPPolicy
```

### Reinforcement Learning Experiment

```bash 
    cd examples
    python meta_learning.py --env {env} --policy {policy} --num-tasks 8 --seed 42
```

The options available for ```env``` are ```{ant_dir, swimmer_dir, halfcheetah_vel, hopper_vel, fetch, ur5e}```.


## Directory Structure

* ```logs/```: This directory contains log files generated during the execution.

* ```Dependencies/```: This directory contains information about the dependencies required to run the project.

* ```checkpoints/```: This directory contains checkpoints from model training.

* ```examples/```: This directory contains example Python scripts, demonstrating how to use this project or its models.

* ```figure/```: This directory contains figures related to the manuscript, as well as Jupyter notebooks used to generate these figures.


All the figures in the manuscript containing the experiments and their generated scripts can be found under the ```figure``` path
