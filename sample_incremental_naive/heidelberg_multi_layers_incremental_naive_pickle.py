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

# Here we load the Dataset
cache_dir = os.path.expanduser("~/data")
cache_subdir = "hdspikes"
get_shd_dataset(cache_dir, cache_subdir)

#SHD merged is the entire dataset, both train and test set merged together from the original version available at https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/
dataset_file = h5py.File(os.path.join(cache_dir, cache_subdir, 'shd_merged.h5'), 'r')

#Division of the dataset in spikes and labels
x_train = dataset_file['spikes']
y_train = dataset_file['labels']

#Extra information (including speaker ID, generality etc..) included in the "extra" dataset branch
e_train = dataset_file['extra']
speakers_train = e_train['speaker']

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

speaker = args.speaker

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
def run_snn(inputs):
    #Initializing the synapsis and membrane tensors used by the hidden layers
    syn1 = torch.zeros((batch_size,nb_hidden_1), device=device, dtype=dtype)
    mem1 = torch.zeros((batch_size,nb_hidden_1), device=device, dtype=dtype)
    
    syn2 = torch.zeros((batch_size,nb_hidden_2), device=device, dtype=dtype)
    mem2 = torch.zeros((batch_size,nb_hidden_2), device=device, dtype=dtype)
    
    syn3 = torch.zeros((batch_size,nb_hidden_3), device=device, dtype=dtype)
    mem3 = torch.zeros((batch_size,nb_hidden_3), device=device, dtype=dtype)
    
    #syn4 = torch.zeros((batch_size,nb_hidden_4), device=device, dtype=dtype)
    #mem4 = torch.zeros((batch_size,nb_hidden_4), device=device, dtype=dtype)

    mem_rec = []
    spk_rec = []

    # Compute hidden layer activity
    #Initializing the output tensor
    out1 = torch.zeros((batch_size, nb_hidden_1), device=device, dtype=dtype)
    out2 = torch.zeros((batch_size, nb_hidden_2), device=device, dtype=dtype)
    out3 = torch.zeros((batch_size, nb_hidden_3), device=device, dtype=dtype)
    #out4 = torch.zeros((batch_size, nb_hidden_4), device=device, dtype=dtype)
    
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
        #h4_from_out3 = torch.einsum("ab,bc->ac", (out3, w4))
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
        #mem4 = new_mem4
        syn1 = new_syn1
        syn2 = new_syn2
        syn3 = new_syn3
        #syn4 = new_syn4

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
    
#Train function used to incrementally train the model
#This method is very similar to the previous one, but uses different lists for training and testing.
#Arguments:
#           x_data_pretrain (list): list of torch.Tensor containing the TEST set of the unfiltered speakers which were used for pretraining
#           y_data_pretrain (list): labels associated to x_data_pretrain
#           x_data_continual (list): list of torch.Tensor containing the TRAIN set of the speaker that is to be added
#           y_data_continual (list): labels associated to x_data_continual
#           x_data_test (list): list of torch.Tensor containing the TEST set of the speaker that is to be added
#           y_data_test (list): labels associated to x_data_test
#           file (file): file object used for printing output
#           csv (file): csv file object used for printing output 
#           lr (float): learning rate   
#           nb_epochs (int): number of training epochs
#Returns:
#           loss_hist (list): history of loss values (unused atm)       
def naive_sample_incremental_train(x_data_pretrain, y_data_pretrain, x_data_continual, y_data_continual, x_data_test, y_data_test, file, csv, lr=1e-3, nb_epochs=10):
    
    params = [w1,w2,w3,w_out,v1,v2,v3]
    optimizer = torch.optim.Adamax(params, lr=lr, betas=(0.9,0.999))

    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()
    
    loss_hist = []
    for e in range(nb_epochs):
        local_loss = []
        for x_local, y_local in zip(x_data_continual, y_data_continual):
            x_local.to(device)
            y_local.to(device)
            output,recs = run_snn(x_local.to_dense())
            y_local = y_local.type(torch.LongTensor).to(device)   # casting to long
            _,spks=recs
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
        #live_plot(loss_hist)
        file.write("\nEpoch %i: loss=%.5f\n"%(e+1,mean_loss))
        print("\nEpoch %i: loss=%.5f\n"%(e+1,mean_loss))
        
        #Printing the model's accuracy over the various speakers every epoch
        f.write("\nEpoch %i: Pretrain test set accuracy on the 11 speakers: %.3f\n"%(e+1, compute_classification_accuracy(x_data_pretrain,y_data_pretrain)))
        print("\nEpoch %i: Pretrain test set accuracy on the 11 speakers: %.3f\n"%(e+1, compute_classification_accuracy(x_data_pretrain,y_data_pretrain)))
        f.write("\nEpoch %i: Test accuracy: %.3f\n"%(e+1, compute_classification_accuracy(x_data_test,y_data_test)))
        print("\nEpoch %i: Test accuracy: %.3f\n"%(e+1, compute_classification_accuracy(x_data_test,y_data_test)))
        csv.write("%d, %i, %.3f, %.3f\n"%(speaker, e+1, compute_classification_accuracy(x_data_pretrain,y_data_pretrain), compute_classification_accuracy(x_data_test,y_data_test)))
        
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
        output,_ = run_snn(x_local.to_dense())
        m,_= torch.max(output,1) # max over time
        _,am=torch.max(m,1)      # argmax over output units
        tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
        accs.append(tmp)
    return np.mean(accs)
    
#The same function as before, except that it calculates the accuracy on a granural level, over the various speakers
#Arguments:
#           x_data (list): list of torch.Tensor containing the dataset on which to experiment accuracy. Usually a test set 
#           y_data (list): list of torch.Tensor containing the labels associated to x_data
#           speaker_data (list): list of torch.Tensor containing the speaker data associated to x_data
#Returns:
#           per_speaker_acc (list): list of floats containing the average accuracy calculated when testing on each different speaker (the first element is calculated on test set of the first speaker, and so on)
def compute_classification_accuracy_speaker(x_data, y_data, speaker_data):
    """ Computes classification accuracy on supplied data in batches. """
    accs = []
    per_speaker_acc = [0] * 12
    per_speaker_count = [0] * 12
    for x_local, y_local, speaker_local in zip(x_data, y_data, speaker_data):
        x_local.to(device)
        y_local.to(device)
        speaker_local.to(device)
        output,_ = run_snn(x_local.to_dense())
        m,_= torch.max(output,1) # max over time
        _,am=torch.max(m,1)      # argmax over output units
        corr = (y_local==am)
        for i, c in enumerate(corr):
            per_speaker_acc[speaker_local[i]] += c
            per_speaker_count[speaker_local[i]] += 1
            
    for i, c in enumerate(per_speaker_count):
        if c != 0:
            per_speaker_acc[i] = per_speaker_acc[i] / c;
    return per_speaker_acc





#MAIN CODE HERE


print("\nEntering train function for speaker %d\n" % speaker)


#Reload the train and test sets used to create the statedict
f = open("../statedicts_64_elements/pretrain_x_speaker_%d.pkl" % speaker, "rb")
x_batches_pretrain = pickle.load(f)
f.close()

f = open("../statedicts_64_elements/pretrain_y_speaker_%d.pkl" % speaker, "rb")
y_batches_pretrain = pickle.load(f)
f.close()

f = open("../statedicts_64_elements/pretrain_test_x_speaker_%d.pkl" % speaker, "rb")
x_batches_pretrain_test = pickle.load(f)
f.close()

f = open("../statedicts_64_elements/pretrain_test_y_speaker_%d.pkl" % speaker, "rb")
y_batches_pretrain_test = pickle.load(f)
f.close()

f = open("../statedicts_64_elements/removed_x_speaker_%d.pkl" % speaker, "rb")
x_batches_continual = pickle.load(f)
f.close()

f = open("../statedicts_64_elements/removed_y_speaker_%d.pkl" % speaker, "rb")
y_batches_continual = pickle.load(f)
f.close()

#Initializing the batch lists for reharsal training and test on the new data
x_batches_continual_train = []
x_batches_continual_test = []
y_batches_continual_train = []
y_batches_continual_test = []

#Separating the incremental set in the same fashion as the pretrain set: 1/5th of the batches are used as test set
n_batches = len(x_batches_continual)
n_batches_div = n_batches//5

print("\nNumber of batches: %d\n" % n_batches)

start = 0
stop = n_batches_div

if stop > 1:
	x_batches_continual_test = x_batches_continual[start:stop-1]
	y_batches_continual_test = y_batches_continual[start:stop-1]
else:
	x_batches_continual_test = x_batches_continual[0:1]
	y_batches_continual_test = y_batches_continual[0:1]


for j in range(n_batches):
    if j < start or j >= stop:
        x_batches_continual_train.append(x_batches_continual[j])
        y_batches_continual_train.append(y_batches_continual[j])
    
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

f = open("speaker_%d_naive_sample_inc_train.txt" % speaker, 'w')
csv = open("naive_incremental_csv.csv", "a")

f.write("\nPre-Training accuracy on the 11 speakers: %.3f\n"%(compute_classification_accuracy(x_batches_pretrain_test,y_batches_pretrain_test)))
print("\nPre-Training accuracy on the 11 speakers: %.3f\n"%(compute_classification_accuracy(x_batches_pretrain_test,y_batches_pretrain_test)))
f.write("\nPre-Training Test accuracy on speaker %d: %.3f\n"%(speaker, compute_classification_accuracy(x_batches_continual_test,y_batches_continual_test)))
print("\nPre-Training Test accuracy on speaker %d: %.3f\n"%(speaker, compute_classification_accuracy(x_batches_continual_test,y_batches_continual_test)))

#Function call to the incremental train function, see above for more info
loss_hist = naive_sample_incremental_train(x_batches_pretrain_test, y_batches_pretrain_test, x_batches_continual_train, y_batches_continual_train, x_batches_continual_test, y_batches_continual_test, file=f, csv=csv, lr=2e-4, nb_epochs=50)

f.write("\nPost-Incremental Training Test accuracy: %.3f\n"%(compute_classification_accuracy(x_batches_continual_test+x_batches_pretrain_test,y_batches_continual_test+y_batches_pretrain_test)))
print("\nPost-Incremental Training Test accuracy: %.3f\n"%(compute_classification_accuracy(x_batches_continual_test+x_batches_pretrain_test,y_batches_continual_test+y_batches_pretrain_test)))

f.close()
csv.close()