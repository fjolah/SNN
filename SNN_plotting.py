#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Tested with Python 3.9.7 / Brian2 / macOS 13.2.1
"""
Created on Tue May  2 19:21:14 2023

@author: fjolahyseni
"""

from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from brian2 import *

#Load the data from all the simulations run in the 3 scenarios.
sch_normals = np.load('sch_spikes.npy')
sch_wins = np.load('sch_win_spikes.npy')
sch_ins = np.load('sch_in_spikes.npy')
snn_normals = np.load('snn_spikes.npy')
snn_wins = np.load('snn_win_spikes.npy')
snn_ins = np.load('snn_in_spikes.npy')
    
sch_normalt = np.load('sch_spiketimes.npy')
sch_wint = np.load('sch_win_spiketimes.npy')
sch_int = np.load('sch_in_spiketimes.npy')
snn_normalt = np.load('snn_spiketimes.npy')
snn_wint = np.load('snn_win_spiketimes.npy')
snn_int = np.load('snn_in_spiketimes.npy')

#Code to plot Figure 1 in the paper. 
fig =plt.figure(tight_layout ='True', figsize = (10,4))
gs = gridspec.GridSpec(2,2)

ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(sch_wint,sch_wins, s= 1e-1,  color = 'firebrick')
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.set_ylim(0,500)
ax1.set_xlim(0,0.1)
ax1.set_ylabel('Neuron index')
ax1.set_title('A. 30% decrease, 10% of the synaptic weights', loc = 'center')

ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(sch_int,sch_ins, s= 1e-1,  color = 'darkred')
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_title('B. 10% of input in inhbition, 5% of neurons')
ax2.set_ylim(0,500)
ax2.set_xlim(0,0.1)

ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(snn_wint,snn_wins, s= 1e-1,  color = 'firebrick')
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.set_xlabel('Time in seconds')
ax3.set_title('C. 30% decrease, 100% of the synaptic weights')
ax3.set_ylabel('Neuron index')

ax4 = fig.add_subplot(gs[1,1])
ax4.scatter(snn_int,snn_ins, s= 1e-1,  color = 'darkred')
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.set_xlabel('Time in seconds')
ax4.set_title('D. 10% of input in inhbition, 50% of neurons')

plt.savefig("esannfig.eps", format='eps', bbox_inches='tight')
#%%
#Load the data from the simulations run on cases without inhibition or 'weight noise'.
snn_normalv = np.load('snn_tracev.npy')
snn_normaltrace = np.load('snn_traces.npy')
sch_normalv = np.load('sch_tracev.npy')
sch_normaltrace = np.load('sch_traces.npy')

#Code to plot Figure 2 in the paper. 
fig = plt.figure(figsize=(10,4))
gs = GridSpec(2,2, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
for i in range(10):
    ax1.plot(sch_normaltrace[:] / ms, sch_normalv[i*50+400,:] / mV, label = i+10, alpha = 0.3)
ax1.plot(sch_normaltrace[:]/ ms, sch_normalv[550,:] / mV, color = 'maroon')
ax1.set_ylabel('Membrane Potential (mV)')
ax1.set_ylim(-60,-34)
ax1.set_xlim(60,110)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.set_title(' Synfire Chain', loc = 'center')

ax2 = fig.add_subplot(gs[0, 1])
for i in range(10):
    ax2.plot(snn_normaltrace[:] / ms, snn_normalv[i*50+600,:] / mV, label = i+10, alpha = 0.3)
ax2.plot(snn_normaltrace[:]/ ms, snn_normalv[800,:] / mV, color = 'maroon')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.set_ylim(-60,-34)
ax2.set_xlim(60,110)
ax2.set_title('Ring Model', loc = 'center')

ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(sch_normalt*1000,sch_normals, s= 1,  color = 'black')
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.set_xlabel('Time (ms)')
ax3.set_ylim(370,750)
ax3.set_xlim(60,110)
ax3.set_ylabel('Neuron Index', y = 0.5)

ax4 = fig.add_subplot(gs[1, 1])
ax4.scatter(snn_normalt*1000,snn_normals, s= 1e-1,  color = 'black')
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.set_xlabel('Time (ms)')
ax4.set_ylim(570,1000)
ax4.set_xlim(60,110)

plt.tight_layout()
ax1.set_rasterized(True)
ax2.set_rasterized(True)
ax3.set_rasterized(True)
ax4.set_rasterized(True)
plt.savefig("ringvssch.eps",format = 'eps', bbox_inches='tight')
plt.show()

