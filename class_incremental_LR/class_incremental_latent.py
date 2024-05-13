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

import copy

# The coarse network structure and the time steps are dicated by the SHD dataset, but may have been slightly modified to accomodate our use case.

#Set the OS to use the GPU on socket 0 (used for GPU server)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
parser.add_argument('--removed_class', type=int, default=0)     #Speaker ID to be removed for sample-incremental experiment
parser.add_argument('--latent_depth', type=int, default=0)      #Depth of the latent replay layer
parser.add_argument('--latent_batches', type=int, default=10)   #Number of activation batches used as latent memory
parser.add_argument('--lr', type=float, default=2e-4)		#Learning Rate
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

lr = args.lr

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

latent_depth = args.latent_depth
latent_batches = args.latent_batches

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
def run_snn(inputs, depth, l):
    torch.set_grad_enabled(False)
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
    #out4 = torch.zeros((batch_size, nb_hidden_4), device=device, dtype=dtype)
    
    #torch.einsum is a very fancy way to declare a matrix multiplication. abc and cd indicates the dimensions of the input matrices (inputs and w1), 
    #while abd is the dimension of the expected output.
    #torch figures the dimensions against which the abc and cd matrices have to multiplied in order to get the desired abd output. It's as shrimple as that.
    #Anyway, this is the first linear multiplication between the inputs and the weight matrix
    if(l == False):
        h1_from_input = torch.einsum("abc,cd->abd", (inputs, w1)) # a=batch_size b=nb_steps c=nb_units d=nb_hidden
    
    if(depth == 0):
            torch.set_grad_enabled(True)
            if(l == True):
                h1_from_input = inputs
    
    #Spiking function
    #Cycle over the number of timesteps
    for t in range(nb_steps):
        if(l == False or depth == 0):
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
            mem1 = new_mem1
            syn1 = new_syn1
        
        if(depth == 1):
            torch.set_grad_enabled(True)
            if(l == True):
                out1 = inputs[:,t]
        
        if(l == False or depth < 2):
            #Multiply out1 with the weights for next layer
            h2_from_out1 = torch.einsum("ab,bc->ac", (out1, w2))
            
            #Repeat process for second layer
            h2 = h2_from_out1 + torch.einsum("ab,bc->ac", (out2, v2))
            mthr2 = mem2-1.0
            out2 = spike_fn(mthr2)
            rst2 = out2.detach()
            
            new_syn2 = alpha*syn2 + h2
            new_mem2 =(beta*mem2 +syn2)*(1.0-rst2)
            mem2 = new_mem2
            syn2 = new_syn2
        
        if(depth == 2):
            torch.set_grad_enabled(True)
            if(l == True):
                out2 = inputs[:,t]
        
        
        if(l == False or depth < 3):
            h3_from_out2 = torch.einsum("ab,bc->ac", (out2, w3))
            
            #Repeat process for third layer
            h3 = h3_from_out2 + torch.einsum("ab,bc->ac", (out3, v3))
            mthr3 = mem3-1.0
            out3 = spike_fn(mthr3)
            rst3 = out3.detach()
            
            new_syn3 = alpha*syn3 + h3
            new_mem3 =(beta*mem3 +syn3)*(1.0-rst3)
            mem_rec.append(mem3)
            mem3 = new_mem3
            syn3 = new_syn3
            
        if(depth == 3):
            torch.set_grad_enabled(True)
            if(l == True):
                out3 = inputs[:,t]
            
        #Both the current membrane potential and current output are saved in "histories"
        spk_rec.append(out3)
        
    if(l == False or depth < 3):    
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
def run_snn_generate_latent(inputs, depth):
    with torch.no_grad():
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
        #out4 = torch.zeros((batch_size, nb_hidden_4), device=device, dtype=dtype)
        
        #torch.einsum is a very fancy way to declare a matrix multiplication. abc and cd indicates the dimensions of the input matrices (inputs and w1), 
        #while abd is the dimension of the expected output.
        #torch figures the dimensions against which the abc and cd matrices have to multiplied in order to get the desired abd output. It's as shrimple as that.
        #Anyway, this is the first linear multiplication between the inputs and the weight matrix
        h1_from_input = torch.einsum("abc,cd->abd", (inputs, w1)) # a=batch_size b=nb_steps c=nb_units d=nb_hidden
        if(depth == 0):
            latent_activations = h1_from_input
        
        latent_rec = []
            
        
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
            
            if(depth == 1):
                latent_rec.append(rst1)

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
            
            if(depth == 2):
                latent_rec.append(rst2)
            
            new_syn2 = alpha*syn2 + h2
            new_mem2 =(beta*mem2 +syn2)*(1.0-rst2)
            h3_from_out2 = torch.einsum("ab,bc->ac", (out2, w3))
            
            
            
            #Repeat process for third layer
            h3 = h3_from_out2 + torch.einsum("ab,bc->ac", (out3, v3))
            mthr3 = mem3-1.0
            out3 = spike_fn(mthr3)
            rst3 = out3.detach()
            
            if(depth == 3):
                latent_rec.append(rst3)
            
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
        
        if(depth != 0):
           latent_activations = torch.stack(latent_rec, dim=1) 
           
        out_rec = torch.stack(out_rec,dim=1)
        other_recs = [mem_rec, spk_rec]
        return latent_activations, out_rec, other_recs
        
#Train function
#Arguments:
#           x_data (list): list of torch.Tensor containing sparse input spiked data 
#           y_data (list): list of torch.Tensor containing the labels associated with the samples
#           lr (float): learning rate
#           nb_epochs (int): number of pre-training epochs
#Returns:
#           loss_hist (list): history of all the loss values, not used for profiling atm 
def train_latent(x_data, y_data, x_batches_pretrain_test, y_batches_pretrain_test, x_batches_continual_test, y_batches_continual_test, f, csv, depth, latent_data_x, latent_data_y, lr=1e-3, nb_epochs=10):
    #The weight matrices are used for both pre-train and incremental train
    if(depth==0):
        params = [w2,w3,w_out,v1,v2,v3]
    if(depth==1):
        params = [w2,w3,w_out,v2,v3]
    if(depth==2):
        params = [w3,w_out,v3]
    if(depth==3):
        params = [w_out]
    optimizer = torch.optim.Adamax(params, lr=lr, betas=(0.9,0.999))

    log_softmax_fn = nn.LogSoftmax(dim=1)
    loss_fn = nn.NLLLoss()
    
    loss_hist = []
    for e in range(nb_epochs):
        local_loss = []
        #Train on new data
        for x_local, y_local in zip(x_data, y_data):
            x_local.to(device)
            y_local.to(device)
            #Run the SNN here
            output,recs = run_snn(x_local.to_dense(), depth, l=False)
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
        #Train on latent data    
        for latent_local_x, latent_local_y in zip(latent_data_x, latent_data_y):
            latent_local_x.to(device)
            latent_local_y.to(device)
            #Run the SNN here
            output,recs = run_snn(latent_local_x, depth, l=True)
            latent_local_y = latent_local_y.type(torch.LongTensor).to(device)   # casting to long
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
            loss_val = loss_fn(log_p_y, latent_local_y) + reg_loss

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            local_loss.append(loss_val.item())
        
        mean_loss = np.mean(local_loss)
        loss_hist.append(mean_loss)
        
        pretrain_acc = compute_classification_accuracy(x_batches_pretrain_test,y_batches_pretrain_test)
        newclass_acc = compute_classification_accuracy(x_batches_continual_test,y_batches_continual_test)
       
 
        print("\nEpoch %i: loss=%.5f\n"%(e+1,mean_loss))
        print("\nEpoch %i: Post-Latent accuracy on the 19 classes: %.3f\n"%(e+1, pretrain_acc))
        print("\nEpoch %i: Post-Latent accuracy on the removed class %d: %.3f\n" % (e+1, c, (newclass_acc)))
        f.write("\nEpoch %i: Post-Latent accuracy: %.3f %.3f\n"%(e+1, pretrain_acc, newclass_acc))
        csv.write("%d, %d, 38, %f, %i, %.3f, %.3f\n"%(c, depth, lr, e+1, pretrain_acc, newclass_acc))
    
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
        output,_ = run_snn(x_local.to_dense(), 4, l=False)
        m,_= torch.max(output,1) # max over time
        _,am=torch.max(m,1)      # argmax over output units
        tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
        accs.append(tmp)
    return np.mean(accs)
    
def generate_latent(x_batches_pretrain, y_batches_pretrain, latent_depth, latent_batches):
    latent_activations_rec = []
    outputs_rec = []
    
    for n, (x, y) in enumerate(zip(x_batches_pretrain, y_batches_pretrain)):
        x.to(device)
        y.to(device)
        latent_activations, outputs, _ = run_snn_generate_latent(x.to_dense(), latent_depth)
        latent_activations_rec.append(latent_activations)
        outputs_rec.append(y)
        if(n == latent_batches-1):
            break
 
    return torch.stack(latent_activations_rec, dim=1), torch.stack(outputs_rec)
    




#MAIN CODE HERE

c = args.removed_class
print("\nEntering for class %d\n" % c)

#Reload the train and test sets used to create the statedict
f = open("../statedicts_class_incremental/pretrain_x_class_%d.pkl" % c, "rb")
x_batches_pretrain = pickle.load(f)
f.close()

f = open("../statedicts_class_incremental/pretrain_y_class_%d.pkl" % c, "rb")
y_batches_pretrain = pickle.load(f)
f.close()

f = open("../statedicts_class_incremental/pretrain_class_x_class_%d.pkl" % c, "rb")
x_batches_pretrain_test = pickle.load(f)
f.close()

f = open("../statedicts_class_incremental/pretrain_class_y_class_%d.pkl" % c, "rb")
y_batches_pretrain_test = pickle.load(f)
f.close()

f = open("../statedicts_class_incremental/removed_x_class_%d.pkl" % c, "rb")
x_batches_continual = pickle.load(f)
f.close()

f = open("../statedicts_class_incremental/removed_y_class_%d.pkl" % c, "rb")
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
	x_batches_continual_test.append(x_batches_continual[0])
	y_batches_continual_test.append(y_batches_continual[0])


for j in range(n_batches):
    if j < start or j >= stop:
        x_batches_continual_train.append(x_batches_continual[j])
        y_batches_continual_train.append(y_batches_continual[j])
        
#Load the state dict
f = open("../statedicts_class_incremental/state_dict_class_%d.pkl" % c, "rb")

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


with torch.no_grad():
    w_new = torch.empty(w_out[:, c].shape)
    w_new = torch.nn.init.normal_(w_new, mean=0.003, std=0.003)
    w_out[:, c] = w_new

#Generate the latent activation values
print("\nGenrating latent replay values...\n")

latent_data_x_list = []
latent_data_y_list = []
for cl in range(20):
    if (cl != c):
        f = open("../statedicts_class_incremental/removed_x_class_%d.pkl" % cl, "rb")
        x_batches_continual_pain = pickle.load(f)
        f.close()

        f = open("../statedicts_class_incremental/removed_y_class_%d.pkl" % cl, "rb")
        y_batches_continual_pain = pickle.load(f)
        f.close()
        
        latent_data_x, latent_data_y = generate_latent(x_batches_continual_pain, y_batches_continual_pain, latent_depth, 2)
        
        latent_data_x_list.append(latent_data_x[:,0,:,:].squeeze())
        latent_data_y_list.append(latent_data_y[0,:].squeeze())

f = open("latent_class_%d_depth_%d.txt" % (c, latent_depth), "w")
csv = open("latent_class_incremental_csv.csv", "a")

pretrain_acc = compute_classification_accuracy(x_batches_pretrain_test,y_batches_pretrain_test)
newclass_acc = compute_classification_accuracy(x_batches_continual_test,y_batches_continual_test)

print("\nPre-Training accuracy on the 19 classes: %.3f\n"%(pretrain_acc))
print("\nPre-Training accuracy on the removed class %d: %.3f\n" % (c, (newclass_acc)))
f.write("\nPre-Training accuracy on the 19 classes: %.3f\n"%(pretrain_acc))
f.write("\nPre-Training accuracy on the removed class %d: %.3f\n" % (c, (newclass_acc)))

loss_hist = train_latent(x_batches_continual_train, y_batches_continual_train, x_batches_pretrain_test, y_batches_pretrain_test, x_batches_continual_test, y_batches_continual_test, f, csv, latent_depth, latent_data_x_list, latent_data_y_list, lr=lr, nb_epochs=nb_epochs)

f.close()
csv.close()