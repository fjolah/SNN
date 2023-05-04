#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Tested with Python 3.9.7 / Brian2 / macOS 13.2.1
"""
Created on Wed Apr 26 13:17:51 2023

@author: fjolahyseni
"""

from brian2 import *
from brian2tools import *
prefs.codegen.target = 'numpy'
seed = np.random.randint(300)
print(seed)

# Parameters
C = 281 * pF #membrane capacitance
gL = 30 * nS #resting leak conductance
tau = C / gL #membrane time constant
taus = 5*ms #synaptic time constant
EL = -70.6 * mV #resting potential
VT = -50.4 * mV #spike threshold
DeltaT = 2 * mV #slope factor
Vcut = VT + 5 * DeltaT
Vr = VT + 5*mV #reset threshold

#Adaptation parameters
tauw=100*ms #time constant
a = -0.5*nS #adaptation intensity
b = 0.5*nA 

#Noise parameters
noise_mean = 0
noise_std = 0.2*mV
noise_tau = 10*ms

#Simulation
defaultclock.dt = 0.01* ms #timestep
layer_number = 120 #number of layers
layer_size = 25 #neurons present in each layer

eqs = """
dvm/dt = ((EL - vm) + DeltaT*exp((vm - VT)/DeltaT) + I/gL - w/gL + stimul/gL+g)/tau + noise_std*sqrt(1/noise_tau)* xi : volt #AdEx
dw/dt = (a*(vm - EL) - w)/tauw : amp #Adaptation
dg/dt = -g/taus : volt #PSP
stimul : amp (constant)
I : amp
"""

neuron = NeuronGroup(layer_size*layer_number,  model=eqs, threshold='vm>Vcut', 
                     reset="vm=Vr; w += b",  method='euler')
neuron.vm = EL 

#Uncomment the following  line, if you want to test inhibition. Also
#comment the line inhibition = 0.
# inhibition =  -0.1*15*nA * np.random.binomial(n=1, p=0.05, size=[3000])
inhibition  = 0
neuron.stimul = inhibition

S = Synapses(neuron, neuron, model='w_syn : volt',on_pre='g += w_syn')
#feedforward all-to-all connectivity between layers
S.connect(j='k for k in range((int(i/layer_size)+1)*layer_size, (int(i/layer_size)+2)*layer_size) '
            'if i<N_pre-layer_size') #Diesmann_et_al_1999

#Uncomment the following line, if you want to test noise in the weights. Also
#comment the line distribution = 0.
# weight_noise = - 0.3*1* np.random.binomial(n=1, p=0.1, size=[74375])
weight_noise = 0
S.w_syn[:] = 1.4 *mV + weight_noise * mV

trace = StateMonitor(neuron, 'vm', record=True)
spikes = SpikeMonitor(neuron)

"""Starting the simulation"""  
neuron.I = 0
run(1* ms)
neuron.I[:layer_size] = 12*nA #input given to the first layer
run(1*ms)
neuron.I =0
run(349 * ms)

plot_raster(spikes.i, spikes.t)
plt.show()

#Save the data as np arrays so you can plot them later.
# np.save('sch_spikes.npy', spikes.i)
# np.save('sch_spiketimes.npy', spikes.t)
# np.save('sch_tracev.npy', trace.vm)
# np.save('sch_traces.npy', trace.t)
