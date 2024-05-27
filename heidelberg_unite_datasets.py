import os
import h5py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn
import torchvision
from torch.utils import data

from utils import get_shd_dataset

from IPython.display import clear_output

# The coarse network structure and the time steps are dicated by the SHD dataset. 

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")

print(device)

# Here we load the Dataset
cache_dir = os.path.expanduser("~/data")
cache_subdir = "hdspikes"
get_shd_dataset(cache_dir, cache_subdir)

train_file = h5py.File(os.path.join(cache_dir, cache_subdir, 'shd_train.h5'), 'r')
test_file = h5py.File(os.path.join(cache_dir, cache_subdir, 'shd_test.h5'), 'r')
new_data = h5py.File(os.path.join(cache_dir, cache_subdir, 'shd_merged.h5'), 'w')

print("init done")

def getdatasets(key,archive):
    if key[-1] != '/': 
        key += '/'
    out = []

    for name in archive[key]:
        path = key + name
        
        if isinstance(archive[path], h5py.Dataset):
            out += [path]
        else:
            out += getdatasets(path,archive)
    return out

# read as much datasets as possible from the old HDF5-file
datasets = getdatasets('/', train_file)
print("Datasets: ")
print(datasets)

# get the group-names from the lists of datasets
groups = list(set([i[::-1].split('/',1)[1][::-1] for i in datasets]))
groups = [i for i in groups if len(i)>0]

# sort groups based on depth
idx    = np.argsort(np.array([len(i.split('/')) for i in groups]))
groups = [groups[i] for i in idx]

print("Groups: ")
print(groups)

# create all groups that contain dataset that will be copied
for group in groups:
    new_data.create_group(group)
    
# copy datasets
for path in datasets:
    # - get group name
    group = path[::-1].split('/',1)[1][::-1]

    # - minimum group name
    if len(group) == 0: group = '/'

    # - copy data
    train_file.copy(path, new_data[group])
    for element in test_file[path]:
        new_data[path].resize(new_data[path].shape[0]+1, axis=0)
        new_data[path][-1] = element
    print(path)
    print(new_data[path].shape)



# test new datasets
x_train = new_data['spikes']
y_train = new_data['labels']

e_train = new_data['extra']

speakers_train = e_train['speaker']

m=0
for s in speakers_train:
    if s > m:
        m = s

speaker_count = [None] * (m + 1)

for s in speakers_train:
    if speaker_count[s] is not None:
        speaker_count[s] = speaker_count[s] + 1
    else:
        speaker_count[s] = 1

print("Checking in train set...")
print(speaker_count)