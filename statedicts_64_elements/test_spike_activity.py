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

import pickle

# The coarse network structure and the time steps are dicated by the SHD dataset, but may have been slightly modified to accomodate our use case.

#Set the OS to use the GPU on socket 0 (used for GPU server)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


#Arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--nb_inputs', type=int, default=700)       #Dimension of inputs 
parser.add_argument('--nb_hidden', type=int, default=200)       #Hidden dimension 
parser.add_argument('--nb_outputs', type=int, default=20)       #Number of possible outputs   
parser.add_argument('--time_step', type=float, default=1e-3)    #Duration of each timestep
parser.add_argument('--nb_steps', type=int, default=100)        #Number of timesteps
parser.add_argument('--max_time', type=float, default=1.4)      #Maximum duration of voice sample
parser.add_argument('--batch_size', type=int, default=64)       #Batch size
parser.add_argument('--nb_epochs', type=int, default=200)       #Number of pre-training epochs
parser.add_argument('--tau_mem', type=float, default=10e-3)     #Constant used to generate membrane threshold
parser.add_argument('--tau_syn', type=float, default=5e-3)      #Constant used to generate synapsis threshold
parser.add_argument('--weight_scale', type=float, default=0.2)  #Scaling factor for initial weight generation
parser.add_argument('--reg_1', type=float, default=0)           #First Regolarization factor Leave to zero for best result         
parser.add_argument('--reg_2', type=float, default=0)           #Second Regolarization factor Leave to zero for best result
parser.add_argument('--speaker', type=int, default=0)           #Speaker ID to be removed for sample-incremental experiment
args = parser.parse_args()


nb_inputs   = args.nb_inputs
nb_hidden_1 = (args.nb_hidden)
nb_hidden_2 = nb_hidden_1//2
nb_hidden_3 = nb_hidden_2//2
nb_outputs  = args.nb_outputs

time_step = args.time_step
nb_steps = args.nb_steps
max_time = args.max_time

batch_size = args.batch_size

nb_epochs = args.nb_epochs

dtype = torch.float

# Check whether a GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")     
else:
    device = torch.device("cpu")

print(device)

tau_mem = args.tau_mem
tau_syn = args.tau_syn

#Generating the aplpha and beta constant for managing the spiking thresholds
alpha   = float(np.exp(-time_step/tau_syn))
beta    = float(np.exp(-time_step/tau_mem))

weight_scale = args.weight_scale

#Generating weight matrices
w1 = torch.empty((nb_inputs, nb_hidden_1),  device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w1, mean=0.0, std=weight_scale/np.sqrt(nb_inputs))

w2 = torch.empty((nb_hidden_1, nb_hidden_2),  device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w2, mean=0.0, std=weight_scale/np.sqrt(nb_hidden_1))

w3 = torch.empty((nb_hidden_2, nb_hidden_3),  device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w3, mean=0.0, std=weight_scale/np.sqrt(nb_hidden_2))

v1 = torch.empty((nb_hidden_1, nb_hidden_1), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(v1, mean=0.0, std=weight_scale/np.sqrt(nb_hidden_1))

v2 = torch.empty((nb_hidden_2, nb_hidden_2), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(v2, mean=0.0, std=weight_scale/np.sqrt(nb_hidden_2))

v3 = torch.empty((nb_hidden_3, nb_hidden_3), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(v3, mean=0.0, std=weight_scale/np.sqrt(nb_hidden_3))


w_out = torch.empty((nb_hidden_3, nb_outputs), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w_out, mean=0.0, std=weight_scale/np.sqrt(nb_hidden_3))

reg_1 = args.reg_1 
reg_2 = args.reg_2 

print("\ninit done\n")

#Leaving the original comments in triple brackets
class SurrGradSpike(torch.autograd.Function):
    """
    Here we implement our spiking nonlinearity which also implements 
    the surrogate gradient. By subclassing torch.autograd.Function, 
    we will be able to use all of PyTorch's autograd functionality.
    Here we use the normalized negative part of a fast sigmoid 
    as this was done in Zenke & Ganguli (2018).
    """
    
    scale = 100.0 # controls steepness of surrogate gradient

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we compute a step function of the input Tensor
        and return it. ctx is a context object that we use to stash information which 
        we need to later backpropagate our error signals. To achieve this we use the 
        ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor we need to compute the 
        surrogate gradient of the loss with respect to the input. 
        Here we use the normalized negative part of a fast sigmoid 
        as this was done in Zenke & Ganguli (2018).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input/(SurrGradSpike.scale*torch.abs(input)+1.0)**2
        return grad
    
# here we overwrite our naive spike function by the "SurrGradSpike" nonlinearity which implements a surrogate gradient
spike_fn  = SurrGradSpike.apply

#Here we define the SNN structure, which is a shallow net composed by 1 linear layer, 1 hidden "spiking" layer, and one spiking classification layer.
#inputs is the matrix of spikes, generated automatically using the to_dense() method on firing_times and units_fired composing the sparse input samples representation.
def run_snn_2(inputs):
    #Initializing the synapsis and membrane tensors used by the hidden layers
    syn1 = torch.zeros((batch_size,nb_hidden_1), device=device, dtype=dtype)
    mem1 = torch.zeros((batch_size,nb_hidden_1), device=device, dtype=dtype)
    
    syn2 = torch.zeros((batch_size,nb_hidden_2), device=device, dtype=dtype)
    mem2 = torch.zeros((batch_size,nb_hidden_2), device=device, dtype=dtype)
    
    syn3 = torch.zeros((batch_size,nb_hidden_3), device=device, dtype=dtype)
    mem3 = torch.zeros((batch_size,nb_hidden_3), device=device, dtype=dtype)


    mem_rec = []
    spk_rec = []

    # Compute hidden layer activity
    #Initializing the output tensor
    out1 = torch.zeros((batch_size, nb_hidden_1), device=device, dtype=dtype)
    out2 = torch.zeros((batch_size, nb_hidden_2), device=device, dtype=dtype)
    out3 = torch.zeros((batch_size, nb_hidden_3), device=device, dtype=dtype)
    
    #torch.einsum is a very fancy way to declare a matrix multiplication. abc and cd indicates the dimensions of the input matrices (inputs and w1), 
    #while abd is the dimension of the expected output.
    #torch figures the dimensions against which the abc and cd matrices have to multiplied in order to get the desired abd output. It's as shrimple as that.
    #Anyway, this is the first linear multiplication between the inputs and the weight matrix
    h1_from_input = torch.einsum("abc,cd->abd", (inputs, w1)) # a=batch_size b=nb_steps c=nb_units d=nb_hidden
    
    #Spiking function
    #Cycle over the number of timesteps
    for t in range(nb_steps):
        #a=batch_size, b=nb_hidden_1, c=nb_hidden_1
        #Sums the current output multiplied by a weight matrix (v1) with the activation values of the current timestep
        h1 = h1_from_input[:,t] + torch.einsum("ab,bc->ac", (out1, v1))
        #The membrane potential of the previous timestep is substracted by 1
        mthr1 = mem1-1.0
        #Spikes are generated wherever mthr is still positive
        out1 = spike_fn(mthr1)
        rst1 = out1.detach() # We do not want to backprop through the reset, so we make a clone of it detached to the gradient

        #Current timestep h1 is used to stimulate the synapsis, the synapsis then stimulates the membrane
        #If previous output had spiked, the membrane is reset, which may ignore part of the current synapsis contribution
        new_syn1 = alpha*syn1 + h1
        new_mem1 =(beta*mem1 +syn1)*(1.0-rst1)
        
        #Multiply out1 with the weights for next layer
        h2_from_out1 = torch.einsum("ab,bc->ac", (out1, w2))
        
        #Repeat process for second layer
        h2 = h2_from_out1 + torch.einsum("ab,bc->ac", (out2, v2))
        mthr2 = mem2-1.0
        out2 = spike_fn(mthr2)
        rst2 = out2.detach()
        new_syn2 = alpha*syn2 + h2
        new_mem2 =(beta*mem2 +syn2)*(1.0-rst2)
        h3_from_out2 = torch.einsum("ab,bc->ac", (out2, w3))
        
        #Repeat process for third layer
        h3 = h3_from_out2 + torch.einsum("ab,bc->ac", (out3, v3))
        mthr3 = mem3-1.0
        out3 = spike_fn(mthr3)
        rst3 = out3.detach()
        new_syn3 = alpha*syn3 + h3
        new_mem3 =(beta*mem3 +syn3)*(1.0-rst3)
        '''
        #Repeat process for fourth layer
        h4 = h4_from_out3 + torch.einsum("ab,bc->ac", (out4, v4))
        mthr4 = mem4-1.0
        out4 = spike_fn(mthr4)
        rst4 = out4.detach()
        new_syn4 = alpha*syn4 + h4
        new_mem4 =(beta*mem4 +syn4)*(1.0-rst4)
        '''
        #Both the current membrane potential and current output are saved in "histories"
        mem_rec.append(mem3)
        spk_rec.append(out3)
        
        mem1 = new_mem1
        mem2 = new_mem2
        mem3 = new_mem3
        
        syn1 = new_syn1
        syn2 = new_syn2
        syn3 = new_syn3
        

    mem_rec = torch.stack(mem_rec,dim=1)
    spk_rec = torch.stack(spk_rec,dim=1)

    # Readout layer
    # a=batch_size b=nb_steps c=nb_hidden d=nb_outputs
    #The history of output spikes is multiplied by a weight matrix (w_out)
    h4 = torch.einsum("abc,cd->abd", (spk_rec, w_out))
    flt = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    out = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    out_rec = [out]
    for t in range(nb_steps):
        #Outputs are treated as "spikes" over the time dimension, using the same synapsis and membrane constants as filters. No resets are used
        new_flt = alpha*flt +h4[:,t]
        new_out = beta*out +flt

        flt = new_flt
        out = new_out

        out_rec.append(out)
        
    out_rec = torch.stack(out_rec,dim=1)
    other_recs = [mem_rec, spk_rec]
    return out_rec, other_recs

#Here we define the SNN structure, which is a shallow net composed by 1 linear layer, 1 hidden "spiking" layer, and one spiking classification layer.
#inputs is the matrix of spikes, generated automatically using the to_dense() method on firing_times and units_fired composing the sparse input samples representation.
def run_snn(inputs):
    #Initializing the synapsis and membrane tensors used by the hidden layers
    syn1 = torch.zeros((batch_size,nb_hidden_1), device=device, dtype=dtype)
    mem1 = torch.zeros((batch_size,nb_hidden_1), device=device, dtype=dtype)
    
    syn2 = torch.zeros((batch_size,nb_hidden_2), device=device, dtype=dtype)
    mem2 = torch.zeros((batch_size,nb_hidden_2), device=device, dtype=dtype)
    
    syn3 = torch.zeros((batch_size,nb_hidden_3), device=device, dtype=dtype)
    mem3 = torch.zeros((batch_size,nb_hidden_3), device=device, dtype=dtype)


    mem_rec = []
    spk_rec = []

    # Compute hidden layer activity
    #Initializing the output tensor
    out1 = torch.zeros((batch_size, nb_hidden_1), device=device, dtype=dtype)
    out2 = torch.zeros((batch_size, nb_hidden_2), device=device, dtype=dtype)
    out3 = torch.zeros((batch_size, nb_hidden_3), device=device, dtype=dtype)
    
    #torch.einsum is a very fancy way to declare a matrix multiplication. abc and cd indicates the dimensions of the input matrices (inputs and w1), 
    #while abd is the dimension of the expected output.
    #torch figures the dimensions against which the abc and cd matrices have to multiplied in order to get the desired abd output. It's as shrimple as that.
    #Anyway, this is the first linear multiplication between the inputs and the weight matrix
    h1_from_input = torch.einsum("abc,cd->abd", (inputs, w1)) # a=batch_size b=nb_steps c=nb_units d=nb_hidden
    
    #Spiking function
    #Cycle over the number of timesteps
    spikes_1 = 0
    spikes_2 = 0
    spikes_3 = 0
    for t in range(nb_steps):
        #a=batch_size, b=nb_hidden_1, c=nb_hidden_1
        #Sums the current output multiplied by a weight matrix (v1) with the activation values of the current timestep
        h1 = h1_from_input[:,t] + torch.einsum("ab,bc->ac", (out1, v1))
        #The membrane potential of the previous timestep is substracted by 1
        mthr1 = mem1-1.0
        #Spikes are generated wherever mthr is still positive
        out1 = spike_fn(mthr1)
        spikes_1 = spikes_1 + out1.sum().item()    
 
        rst1 = out1.detach() # We do not want to backprop through the reset, so we make a clone of it detached to the gradient

        #Current timestep h1 is used to stimulate the synapsis, the synapsis then stimulates the membrane
        #If previous output had spiked, the membrane is reset, which may ignore part of the current synapsis contribution
        new_syn1 = alpha*syn1 + h1
        new_mem1 =(beta*mem1 +syn1)*(1.0-rst1)
        
        #Multiply out1 with the weights for next layer
        h2_from_out1 = torch.einsum("ab,bc->ac", (out1, w2))
        
        #Repeat process for second layer
        h2 = h2_from_out1 + torch.einsum("ab,bc->ac", (out2, v2))
        mthr2 = mem2-1.0
        out2 = spike_fn(mthr2)
        spikes_2 = spikes_2 + out2.sum().item() 
        rst2 = out2.detach()
        new_syn2 = alpha*syn2 + h2
        new_mem2 =(beta*mem2 +syn2)*(1.0-rst2)
        h3_from_out2 = torch.einsum("ab,bc->ac", (out2, w3))
        
        #Repeat process for third layer
        h3 = h3_from_out2 + torch.einsum("ab,bc->ac", (out3, v3))
        mthr3 = mem3-1.0
        out3 = spike_fn(mthr3)
        spikes_3 = spikes_3 + out3.sum().item() 
        rst3 = out3.detach()
        new_syn3 = alpha*syn3 + h3
        new_mem3 =(beta*mem3 +syn3)*(1.0-rst3)

        #Both the current membrane potential and current output are saved in "histories"
        mem_rec.append(mem3)
        spk_rec.append(out3)
        
        mem1 = new_mem1
        mem2 = new_mem2
        mem3 = new_mem3
        
        syn1 = new_syn1
        syn2 = new_syn2
        syn3 = new_syn3
        

    print("Layer 1 spike sparsity: %.3f" % (spikes_1 / (batch_size*nb_hidden_1*nb_steps)))
    print("Layer 2 spike sparsity: %.3f" % (spikes_2 / (batch_size*nb_hidden_2*nb_steps)))
    print("Layer 3 spike sparsity: %.3f" % (spikes_3 / (batch_size*nb_hidden_3*nb_steps)))
    mem_rec = torch.stack(mem_rec,dim=1)
    spk_rec = torch.stack(spk_rec,dim=1)

    # Readout layer
    # a=batch_size b=nb_steps c=nb_hidden d=nb_outputs
    #The history of output spikes is multiplied by a weight matrix (w_out)
    h4 = torch.einsum("abc,cd->abd", (spk_rec, w_out))
    flt = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    out = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
    out_rec = [out]
    for t in range(nb_steps):
        #Outputs are treated as "spikes" over the time dimension, using the same synapsis and membrane constants as filters. No resets are used
        new_flt = alpha*flt +h4[:,t]
        new_out = beta*out +flt

        flt = new_flt
        out = new_out

        out_rec.append(out)
        
    out_rec = torch.stack(out_rec,dim=1)
    other_recs = [mem_rec, spk_rec]
    return out_rec, other_recs

#Pre-Train function
#Arguments:
#           x_data (list): list of torch.Tensor containing sparse input spiked data 
#           y_data (list): list of torch.Tensor containing the labels associated with the samples
#           file (file): object pointing to the .json file for saving weights post-train
#           lr (float): learning rate
#           nb_epochs (int): number of pre-training epochs
#Returns:
#           loss_hist (list): history of all the loss values, not used for profiling atm 
def train(x_data, y_data, lr=1e-3, nb_epochs=10):
    #The weight matrices are used for both pre-train and incremental train
    f = open("state_dict.pkl", "wb")
    params = [w1,w2,w3,w_out,v1,v2,v3]
    optimizer = torch.optim.Adamax(params, lr=lr, betas=(0.9,0.999))

    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()
    
    loss_hist = []
    for e in range(nb_epochs):
        local_loss = []
        for x_local, y_local in zip(x_data, y_data):
            x_local.to(device)
            y_local.to(device)
            #Run the SNN here
            output,recs = run_snn(x_local.to_dense())
            y_local = y_local.type(torch.LongTensor).to(device)   # casting to long
            _,spks=recs
            #The maximum values of output is collected over the time dimension, a softmax is used for making the prediction
            m,_=torch.max(output,1)
            log_p_y = log_softmax_fn(m)
        
            # Here we set up our regularizer loss
            # The strength paramters here are merely a guess and there should be ample room for improvement by
            # tuning these paramters.
            reg_loss = reg_1*torch.sum(spks) # L1 loss on total number of spikes
            reg_loss += reg_2*torch.mean(torch.sum(torch.sum(spks,dim=0),dim=0)**2) # L2 loss on spikes per neuron
        
            # Here we combine supervised loss and the regularizer
            loss_val = loss_fn(log_p_y, y_local) + reg_loss

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            local_loss.append(loss_val.item())
        mean_loss = np.mean(local_loss)
        loss_hist.append(mean_loss)
 
        print("\nEpoch %i: loss=%.5f\n"%(e+1,mean_loss))
        
    pickle.dump(params, f)
    f.close()
    
    return loss_hist
    
#Computes accuracy of the model
#Arguments:
#           x_data (list): list of torch.Tensor containing the dataset on which to experiment accuracy. Usually a test set 
#           y_data (list): list of torch.Tensor containing the labels associated to x_data
#Returns:
#           np.mean(accs) (float): average of the accuracies 
def compute_classification_accuracy(x_data, y_data):
    """ Computes classification accuracy on supplied data in batches. """
    accs = []
    for x_local, y_local in zip(x_data, y_data):
        x_local.to(device)
        y_local.to(device)
        print(x_local.shape)
        output,_ = run_snn_2(x_local.to_dense())
        m,_= torch.max(output,1) # max over time
        _,am=torch.max(m,1)      # argmax over output units
        tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
        accs.append(tmp)
    return np.mean(accs)

#Computes sparsity of the model
#Arguments:
#           x_data (list): list of torch.Tensor containing the dataset on which to experiment accuracy. Usually a test set 
#           y_data (list): list of torch.Tensor containing the labels associated to x_data
#Returns:
#           np.mean(accs) (float): average of the accuracies 
def compute_classification_sparsity(x_data, y_data):
    """ Computes sparsity on supplied data in batches. """
    tot = 0
    num = 0
    for x_local, y_local in zip(x_data, y_data):
        x_local.to(device)
        y_local.to(device)
        x_dense = x_local.to_dense()
        sum = x_dense.sum().item()
        x_sparsity = (sum / torch.numel(x_dense))
        tot = tot + x_sparsity
        num = num + 1
        output,_ = run_snn(x_local.to_dense())
    average = (tot / num)
    return average


    

#Function that generates the data batches to be used in the treaining and testing functions, filtering the speaker selected in command line arguments.
#Arguments:
#           X (h5py._hl.group.Group): dataset group containing the firing times + units fired  datasets
#           y (h5py._hl.dataset.Dataset): dataset object containing the labels
#           batch_size (int): size of the batches, selected by command line argument
#           nb_steps (int): number of time steps each sample is divided into
#           nb_units (int): number of units involved in each sample
#           max_time (float): maximum time duration for voice samples
#           shuffle (bool): if True, the dataset batches are randomly reordered
#Returns:
#           x_batches (list): list of torch.Tensor, containing the batches of the firing times + units fired couple samples
#           y_batches (list): list of torch.Tensor, contianing the batches of the labels of the samples
#           speaker_batches (list): list of torch.Tensor, containing the batches of the speaker information of the samples
#The returned lists are shuffled in the same way, to not lose the positional association
def sparse_data_generator_from_hdf5_spikes(X, y, batch_size, nb_steps, nb_units, max_time, shuffle=True):
    
    #Compute discrete firing times
    firing_times = X['times']
    units_fired = X['units']
    labels_ = np.array(y,dtype=int)
    
    #Convert lists in numpy arrays
    labels_ = np.array(labels_, dtype=int)
    firing_times = np.array(firing_times)
    units_fired = np.array(units_fired)
    
    #Calculate number of batches and initialize lists
    number_of_batches = len(labels_)//batch_size
    sample_index = np.arange(len(labels_))
    
    x_batches = []
    y_batches = []
    
    #Linear space listing the possible/accepted time values, from 0 to max_time, divided by nb_steps equal levels  
    time_bins = np.linspace(0, max_time, num=nb_steps)

    #If shuffle==True, shuffles the list of indexes generated earlier, so that the lists aren't explored in a contiguous fashion
    if shuffle:
        np.random.shuffle(sample_index)

    #Cycle to build the batches
    counter = 0
    while counter<number_of_batches:
        #Select indexes to insert in current batch. If shuffle==True, the indexes selected will be random
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]

        #Placeholder structure to contain the spike info
        coo = [ [] for i in range(3) ]
        for bc,idx in enumerate(batch_index):
            #Approximates the firing times to the closest values available in the time_bins list
            times = np.digitize(firing_times[idx], time_bins)
            units = units_fired[idx]
            batch = [bc for _ in range(len(times))]
            
            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)
    
        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size,nb_steps,nb_units])).to(device)
        y_batch = torch.tensor(labels_[batch_index],device=device)

        x_batches.append(X_batch.to(device=device))
        y_batches.append(y_batch.to(device=device))

        counter += 1
        
    
    return x_batches, y_batches


#MAIN CODE HERE

#Reload the train and test sets used to create the statedict
speaker = args.speaker
print("\nEntering for speaker %d\n" % speaker)


#Reload the train and test sets used to create the statedict
f = open("pretrain_x_speaker_%d.pkl" % speaker, "rb")
x_batches_pretrain = pickle.load(f)
f.close()

f = open("pretrain_y_speaker_%d.pkl" % speaker, "rb")
y_batches_pretrain = pickle.load(f)
f.close()

f = open("pretrain_test_x_speaker_%d.pkl" % speaker, "rb")
x_batches_pretrain_test = pickle.load(f)
f.close()

f = open("pretrain_test_y_speaker_%d.pkl" % speaker, "rb")
y_batches_pretrain_test = pickle.load(f)
f.close()

f = open("removed_x_speaker_%d.pkl" % speaker, "rb")
x_batches_continual = pickle.load(f)
f.close()

f = open("removed_y_speaker_%d.pkl" % speaker, "rb")
y_batches_continual = pickle.load(f)
f.close()

#Load the state dict
f = open("../statedicts_64_elements/state_dict_speaker_%d.pkl" % speaker, "rb")

params = pickle.load(f)
[w1,w2,w3,w_out,v1,v2,v3]
w1 = params[0]
w2 = params[1]
w3 = params[2]
w_out = params[3]
v1 = params[4]
v2 = params[5]
v3 = params[6]

f.close()


print("\nInput sparsity: %.3f\n"%(compute_classification_sparsity(x_batches_pretrain_test,y_batches_pretrain_test)))
