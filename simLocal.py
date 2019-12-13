# -*- coding: utf-8 -*-
"""
Local Metropolis
"""
import numpy as np
from numpy.random import rand
import pandas as pd
import pickle, os
import matplotlib.pyplot as plt


def initialstate(N):   
    ''' generates a random spin configuration for initial condition'''
    state = 2*np.random.randint(2, size=(N,N))-1
    return state

def mcmove(config, beta, *args):
    '''Monte Carlo move using Metropolis algorithm '''
    ene, mag = args[0], args[1]
    for i in range(N):
        for j in range(N):
                a = np.random.randint(0, N)
                b = np.random.randint(0, N)
                s =  config[a, b]
                nb = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]
                cost = - 2*s*nb * J  # (C_new-C_old)*J
                if cost < 0:
                    s *= -1
                    ndupdate = True
                elif rand() < np.exp(-cost*beta):
                    s *= -1
                    ndupdate = True
                else:
                    ndupdate = False
                if ndupdate:
                    ene += cost
                    mag += s - config[a,b]
                    
                config[a, b] = s

    return config, ene, mag


def calcEnergy(config):
    '''Energy of a given configuration'''
    energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i,j]
            nb = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
            energy += -nb*S
    return energy/4.

# change in energy is 8J
#    d          d               u          u        
# d  d  d => d  u  d   or    u  u  u => u  d  u  
#    d          d               u          u 
#     
# change in energy is 4J
#    d          d               u          u        
# d  d  u => d  u  u   or    u  u  d => u  d  d   
#    d          d               u          u 
#     
# Here u and d are used for up and down configuration of the spins

def calcMag(config):
    '''Magnetization of a given configuration'''
    mag = np.sum(config)
    return mag

def storePath(l,tem):
    import os
    dirpath = os.path.join(os.getcwd(),'L%i'%(l))
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    filepath = os.path.join(dirpath,'T%.4fconfig.pkl'%(tem))
    f = open(filepath, 'ab')
    pickle.dump('starting measurement', f)
    return f

def storeConfig(step, config, confdic):
    confdic['step %i'%step] = np.copy(config)
    return confdic

def readConfigFile(l,tem):
    dirpath = os.path.join(os.getcwd(),'L%i'%(l))
    filepath = os.path.join(dirpath,'T%.4fconfig.pkl'%(tem))
    f = open(filepath, 'rb')
    print(pickle.load(f))
    return pickle.load(f)


## change these parameters for a smaller (faster) simulation 
nt      = 31         #  number of temperature points
N       = 40         #  size of the lattice, N x N
eqSteps = int( 2048 * N/20 )      #  number of MC sweeps for equilibration
mcSteps = int( 1024 * N/20 )       #  number of MC sweeps for calculation

J = -1 # -1: ferro, +1: anti-ferro
T       = np.linspace(1.50, 3.30, nt);

#nt=1; T = [2.46]
print(eqSteps, mcSteps)
print(T)
E,M,C,X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)
n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N)
# divide by number of samples, and by system size to get intensive values

#----------------------------------------------------------------------
#  MAIN PART OF THE CODE
#----------------------------------------------------------------------

for tt in range(nt):

    E1 = M1 = E2 = M2 = 0
    config = initialstate(N)
    iT=1.0/T[tt]; iT2=iT*iT;
    if T[tt]==2.46: continue
    fileobj = storePath(N, T[tt])    # store configuration
    confdic = {}
    
    # figure
    E1ep = []
    fig, ax = plt.subplots(figsize=(4,3))
    
    print('start T=', T[tt])
    Ene = calcEnergy(config); Mag = calcMag(config)  # initial measures
    for i in range(eqSteps):         # equilibrate
        config, Ene, Mag = mcmove(config, iT, *(Ene, Mag))           # Monte Carlo moves
        E1ep.append(Ene)
    for i in range(eqSteps, eqSteps+mcSteps):         # measure start
        config, Ene, Mag = mcmove(config, iT, *(Ene, Mag))           # Monte Carlo moves
        
        E1 = E1 + Ene
        M1 = M1 + Mag
        M2 = M2 + Mag*Mag 
        E2 = E2 + Ene*Ene
        
        confdic = storeConfig(i, config, confdic)    # store configuration

        ##
        E1ep.append(Ene)

    curveE, = ax.plot(E1ep)
    plt.show()
    E[tt] = n1*E1
    M[tt] = n1*M1
    C[tt] = (n1*E2 - n2*E1*E1)*iT2
    X[tt] = (n1*M2 - n2*M1*M1)*iT

    confdic['E'] = E[tt]
    confdic['M'] = M[tt]
    confdic['C'] = C[tt]
    confdic['X'] = X[tt]
    pickle.dump(confdic, fileobj)

print("stop")

