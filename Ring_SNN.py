#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Tested with Python 3.9.7 / Brian2 / macOS 13.2.1
"""
Created on Tue May  2 15:58:30 2023

@author: fjolahyseni
"""

from brian2 import *
from brian2tools import *
import matplotlib.pyplot as plt
import numpy as np
prefs.codegen.target = 'numpy'
seed = np.random.randint(300)

# Parameters
C = 281 * pF  # membrane capacitance
gL = 30 * nS  # resting leak conductance
tau = C / gL  # membrane time constant
taus = 5*ms  # synaptic time constant
EL = -70.6 * mV  # resting potential
VT = -50.4 * mV  # spike threshold
DeltaT = 3 * mV  # slope factor
Vcut = VT + 5 * DeltaT
Vr = VT+5*mV

#Synaptic weight parameters
J0 = -9.8/20  # global inhibition
J2 = 2.1*7/20  # excitation factor

#Adaptation parameters
tauw = 100*ms  # time constant
a = 0.001*nS  # adaptation intensity
b = 0.001*nA

#Noise parameters
noise_mean = 0
noise_std = 2*mV
noise_tau = 10*ms

#Simulation
defaultclock.dt = 0.1 * ms  # timestep
n = 3000  # number of neurons
sim_duration = 350 * ms  # duration of the simulation
n_iterations = int(round(sim_duration/defaultclock.dt))

eqs = """
dvm/dt=(gL*(EL-vm)+gL*DeltaT*exp((vm-VT)/DeltaT)+I - w + stimul + g*gL)/C + noise_std*sqrt(1/noise_tau)* xi : volt #AdEx
dw/dt = (a*(vm - EL) - w)/tauw : amp #adaptation
dg/dt = -g/taus : volt #PSP
I : amp
stimul : amp (constant)

"""
neuron = NeuronGroup(n,  model=eqs , threshold='vm>Vcut',reset = "vm=Vr;w += b"\
                     , method='euler')
neuron.vm, neuron.w = EL, a * (neuron.vm - EL)

#Uncomment the following 2 lines, if you want to test inhibition. Also
#comment the line inhibition = 0.
# inhibition =  - 0.1*1.9*gL/nS*nA * np.random.binomial(n=1, p=0.5, size=[3000])
inhibition =  0
neuron.stimul = inhibition

"""Normalized feature space and building the connectivity matrix"""
trajectory, start, end = 1, 0, 1  
width = trajectory/n*80 #sigma of the connectivity matrix's Gaussian profile
bias = trajectory/n*100 #bias in the connectivity matrix
X = np.linspace(start,end, n, endpoint=False).reshape(1,n)
D = np.minimum(np.abs(X - X.T + bias), 1-(np.abs(X - X.T + bias))) 
J = J0 + J2* np.exp(-D**2/ (2 *width*width))
J *= (1-np.eye(n)) #self connections are zero

"""Connectivity pattern among neurons"""
S = Synapses(neuron, neuron, model='w_syn : volt', on_pre = 'g += w_syn')
S.connect()

#Uncomment the following line, if you want to test noise in the weights. Also
#comment the line distribution = 0.
#weight_noise = -.3*np.max(J)* np.random.binomial(n=1, p=1, size=[9000000])
weight_noise = 0
S.w_syn = weight =  (J.T).flatten()* mV + weight_noise*mV 

trace = StateMonitor(neuron, variables = True,  record=True) 
spikes = SpikeMonitor(neuron)

"""Starting the simulation"""  
neuron.I[0] = 1.9*nA 
neuron.I[1:] = 0.7* nA #neurons' constant input
run(15 * ms)
neuron.I[:] = 0.7* nA #neurons' constant input
run(sim_duration-15*ms)

plot_raster(spikes.i,spikes.t)
plt.show()

#Save the data as np arrays so you can plot the two figures in the paper.
# np.save('snn_tracev.npy', trace.vm)
# np.save('snn_traces.npy', trace.t)
#np.save('snn_win_spikes.npy', spikes.i)
#np.save('snn_win_spiketimes.npy', spikes.t)

