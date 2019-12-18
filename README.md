# Towards a Time-efficient CNN Design with Cartesian Genetic Programming
KAIST CS454, AI based Software Engineering, project originated from [cgp-cnn-PyTorch](https://github.com/sg-nm/cgp-cnn-PyTorch)

## Requirements

* Ubuntu 16.04.6 LTS
* Python version         3.6.2
* PyTorch version        0.4.1
* tensorflow             1.9.0
* CUDA version           10.x (Any)
* scikit-image           0.13.0
* pandas                 0.20.3


## Usage

```groovy
exp_main.py [-h] [--gpu_ids GPU_IDS [GPU_IDS ...]]
                   [--population POPULATION] [--lam LAM]
                   [--log_folder LOG_FOLDER] [--max_gen MAX_GEN]
                   [--snm {normal,strong}] [--seed SEED]
                   [--epoch EPOCH]

optional arguments:
  -h, --help            show this help message and exit
  --gpu_ids GPU_IDS [GPU_IDS ...], -g GPU_IDS [GPU_IDS ...]
                        List of GPU ids
  --population POPULATION, -p POPULATION
                        Num. of Population (Num. of Parents)
  --lam LAM, -l LAM     Num. of offsprings
  --log_folder LOG_FOLDER
                        Log folder name
  --max_gen MAX_GEN, -max MAX_GEN
                        Num. of max evaluations
  --snm {normal,strong}, -snm {normal,strong}
                        Strong Neutral Mutation
  --seed SEED, -s SEED  Numpy random seed
  --epoch EPOCH, -e EPOCH
                        Training epoch
```

## Designing Convolutional Neural Network Architectures Based on Cartegian Genetic Programming

This repository contains the code for the following paper:

Masanori Suganuma, Shinichi Shirakawa, and Tomoharu Nagao, "A Genetic Programming Approach to Designing Convolutional Neural Network Architectures," 
Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '17, Best paper award), pp. 497-504 (2017) [[paper]](https://doi.org/10.1145/3071178.3071229) [[arXiv]](https://arxiv.org/abs/1704.00764)
