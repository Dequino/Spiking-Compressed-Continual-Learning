# Spiking Compressed Continual Learning
This is the official Pytorch implementation of **\[ISVLSI 2024\]** - [Spiking Compressed Continual Learning](https://arxiv.org/abs/2407.03111).

**Spiking Neural Networks (SNNs)** are **bio-inspired**, **power efficient** neuron models that specialize in efficiently processing **time series**.

<div align="center">
    <img width="50%" src="https://github.com/Dequino/Spiking-Compressed-Continual-Learning/blob/main/figure/snnn.png?raw=true"/>
</div>

We experiment **Continual Learning** strategies on this family of models, an approach relatively unexplored in the Spiking-related research field, to make them adapt to evolving environments without forgetting past knowledge.

On a **progressive class learning** task in which we continously train our pre-trained model on a different language, we demonstrate that our model generalizes to the new classes, achieving **+88.2%** top-1 accuracy on each new class, with only **2.2%** accuracy loss on the older ones, while learning on **compressed data** (compression ratio 1:2, **50% training memory saving**).

## Highlights
All experiments were done on the **[Heidelberg](https://zenkelab.org/datasets/) SHD Dataset**.

<div align="center">
    <img width="50%" src="https://github.com/Dequino/Spiking-Compressed-Continual-Learning/blob/main/figure/Figure_5.png?raw=true"/>
</div>

- **Sample incremental** - up to **92.46%** top-1 total accuracy when learning the same classes spoken by a new speaker. With compression and learning on a more shallow layer, we achieved **88.79%** top-1 accuracy using only **160 KB** of memory.
- **Class incremental** - up to **92.05%** top-1 total accuracy when learning a new class. With compression and learning on a more shallow layer, we achieved **85.53%** top-1 accuracy using only **160 KB** of memory.

<div align="center">
    <img width="50%" src="https://github.com/Dequino/Spiking-Compressed-Continual-Learning/blob/main/figure/Figure_6.png?raw=true"/>
</div>
  
- **Progressive class learning** - pretrain on 10 classes in english, progressively add 10 german classes. Compression ratio 1:2. Final accuracy on the full test set: **78.4%**.

## Features

<div align="center">
    <img width="50%" src="https://github.com/Dequino/Spiking-Compressed-Continual-Learning/blob/main/figure/LatentReplaysSNN.png?raw=true"/>
</div>

1. **Latent Replays (LRs) in Spiking Neural Networks** - On a pretrained network, when adding new data we first **freeze** first N layers and train only the last ones. We **replay past latent activations** (spike sequences) to avoid forgetting. On narrow layers, we have **memory saving** compared to raw rehearsal.

<div align="center">
    <img width="50%" src="https://github.com/Dequino/Spiking-Compressed-Continual-Learning/blob/main/figure/compression.png?raw=true"/>
</div>

2. **Compressed Latent Replays** - Because we need to store the full spike sequence for each past sample, we use a **lossy time compression** (1). The sub-sampled spike sequences can have different compression ratios (in the image, a 1:4 compression ratio). When replaying past data, we do a **run-time un-compression** (2) to respect time constants.

3. **Reproducibility** - This repository contains the scripts used to generate the sperimental results of our work. It is possible to reproduce said results by following the steps provided in the next sections. Here is a list describing all the different components of this repository:
    - **heidelberg_unite_datasets** - a handy script that unites the original train and test sets of Heidelberg SHD dataset.
    - **Statedicts** - Folders containing the scripts to pretrain the SNN models and save their statedicts. You need to run this scripts to generate the pre-trained models required for experimenting with the different CL strategies provided. There are 3 different pre-training modes: **statedicts** is the base one for sample incremental experiments, **statedicts_64_elements** pretrains using 64-elements batches (sample incremental), and finally **statedicts_class_incremental** contains the script to pretrain the network for the single class-incremental task and the progressive multi-class learning task.   
    - **sample_incremental** folders contain the script to run various sample-incremental experiments. 3 different strategies have been explored: **naive**, in which the new samples are used to train the network without using any kind of rehearsal strategy, **reharsal** in which a full training data reharsal is used to avoid forgetting, and **LR** in which the Latent Replay strategy is used to avoid forgetting, while using much less memory than the reharsal method.
    - **class_incremental** forlders are parallel to the sample incremental ones, and follow a similar structure. **naive**, **reharsal** and **LR** are present. Also, a fourth folder, **class_incremental_multiclasses**, contains the scripts required to run the progressive class learning experiment.

## Getting started

*The versions listed in this section have been tested on a Ubuntu 22.04 machine. Different versions may also work, but have not been tested. Certain scripts assume a linux-based filesystem is being used, if working on a windows machine please edit the scripts accordingly. The experiments were run on NVIDIA RTX A5000 and GeForce GTX 1080 Ti*

Installing a [Conda](https://docs.conda.io/en/latest/) environment with **python=3.11.5** to meet the requirements is strongly suggested.

The packages required can be installed by running:

`pip install h5py matplotlib seaborn torch torchvision IPython`

Download the **shd_test.h5.gz** and **shd_train.h5.gz** dataset files from the [Heidelberg](https://zenkelab.org/datasets/) official repository, and extract them.

The scripts of this repository assume that the files you downloaded are stored under the `~/data/hdspikes/` folder.

After you downloaded and extracted the dataset files, run the merging script:

`python heidelberg_unite_datasets.py`

This will merge the original test and train datasets into a single dataset, **shd_merged.h5**.

This file is used by the scripts found in the **statedicts** folders. Based on the task selected, each one of these scripts does the following:

1. Separate the merged dataset into a **pretrain** and a **"removed"** dataset.
2. Separate the new **pretrain** dataset into a new **train** and **test** set.
3. Pre-train the model.
4. Save **pre-trained model's statedict**, **pretrain train & test datasets** and the **removed data** in 4 different pickle files.

In the sample incremental statedicts, run the script with the  `--speaker==X` option, where X is the index of the speaker you want to remove from the training (and will be used for the sample incremental task). In the class incremental task, run the script with the `--removed-class=X` option.

Each **statedict** folder contains a **testsuite.py** script. This script automatically runs the command for you, once for each speaker/class. A "test.conf" file is generated to check which step the script has already reached.

Once you have generated all the necessary files for a test, keep them in the corresponding **statedict** folder to be accessible by the experiment scripts.

You can now navigate to the test folder you desire to run, and run the corresponding script, with the correct `--speaker==X` or `--removed-class=X` used to generate the statedicts earlier. Different scripts have been provided for each CL strategy, including **subsample** scripts that work with data subsampled by a lossy compression strategy. **testsuite.py** scripts have also been provided to run multiple tests sequentially. Test results are returned in **.csv** and **.txt** files.

## Citation
If this repository has been useful for you, please do cite our work, thank you.

`
@inproceedings{dequino2024compressed,
  title={Compressed Latent Replays for Lightweight Continual Learning on Spiking Neural Networks},
  author={Dequino, Alberto and Carpegna, Alessio and Nadalini, Davide and Savino, Alessandro and Benini, Luca and Di Carlo, Stefano and Conti, Francesco},
  booktitle={2024 IEEE Computer Society Annual Symposium on VLSI (ISVLSI)},
  pages={240--245},
  year={2024},
  organization={IEEE}
}
`

### Acknowledgements
This work received support from Swiss National Science Foundation Project 207913 "TinyTrainer: On-chip Training for TinyML devices"




 


