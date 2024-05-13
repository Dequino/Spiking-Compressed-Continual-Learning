'''
Copyright (C) 2022-2024 Politecnico di Torino and University of Bologna

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

'''
Authors: Alberto Dequino, Davide Nadalini, Alessio Carpegna
'''

import os
import h5py
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import argparse

import torch
import torch.nn as nn
import torchvision
from torch.utils import data

from utils import get_shd_dataset

from IPython.display import clear_output

from os.path import exists

# Create file if non existing
if exists("test_sums.conf"):
    pass
else:
    f = open("test_sums.conf", 'w')
    f.write('0')
    f.close()


def generate_values(n):
    values = []
    values.append(0)
    for i in range(n):
        x = 10 ** (-(n-1) + i)
        values.append(x)
    return values

valuelist=generate_values(12)


f = open("test_sums.conf", 'r')
print(valuelist)
start = int(f.readline())
f.close()
count = 0

values = [40]


for i in range(12):
    if count >= start:
        for j in range(4):
            for k in values:
                cmd = "python heidelberg_latent_subsample_sums.py --speaker=%d --latent_depth=%d --nb_epochs=50 --latent_batches=%d --latent_subsample=1" % (count, j, k)
                print(cmd)
                os.system(cmd)
        f = open("test_sums.conf", 'w')
        f.write("%d" % (count+1))
        f.close()
    count = count + 1
    