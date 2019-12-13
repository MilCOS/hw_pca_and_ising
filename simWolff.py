# -*- coding: utf-8 -*-
"""
Wolff algorithm
@author: xuhan
"""
import numpy as np
import random
import pandas as pd


def initialize(nlist,L,N):       
    """get neighbors of each sites
    """             
    for i in range(N):
        nlist[i][0] = (i-1)%N
        nlist[i][1] = (i-L)%N
        nlist[i][2] = (i+1)%N
        nlist[i][3] = (i+L)%N   

def wolff_step(spin, nlist, N, Padd):
    
    stack = np.zeros(N, dtype=np.int)   # duzelt
    js = random.randrange(0,N) ## pickup one site
    stack[0] = js
    sp=1
    
    oldspin=spin[js]
    newspin = -1*spin[js] # spin flip
    spin[js] = newspin # flip the first one
    
    cluster_size=0
    cluster_idx = [js]
    while sp:
        sp = sp - 1
        current = stack[sp] # backward access --stack
        for i in range(4):
            if spin[nlist[current][i]] == oldspin: # same spin
                if random.uniform(0, 1) < Padd : # whether to put into the cluster
                    stack[sp] = nlist[current][i]
                    sp = sp + 1
                    spin[nlist[current][i]] = newspin # spin flip (then we won't access it again)
                    cluster_size = cluster_size + 1
                    cluster_idx.append( nlist[current][i] ) # store cluster idx

    return cluster_size, cluster_idx

##
def measure(J,spin,nlist,N,Nobs):
    
    observable=np.zeros(Nobs, dtype=np.float64)
    
    observable[0] = np.sum(spin) / N    # magnetization 
    #for i in range(N):
    #    tmp = J * spin[i] * (spin[nlist[i][0]] + spin[nlist[i][1]]) 
    #    observable[1] = observable[1] + tmp
    #observable[1] = observable[1] / N # energy
    observable[2] = observable[0]**2    # m^2
    #observable[3] = observable[1]**2    # e^2

    return observable

def ising2dmc(J,T,L,N,Nobs,mc_warmup=1024,mc_measure=1024,
              spin_read_file=False):
    if spin_read_file:
        spin = np.loadtxt('Wolff/L%i/backupspin_T%.4f.txt'%(L,T)) 
    else:
        spin = np.random.randint(0,2, size=N, dtype=np.int)*2-1
    dfconf = pd.DataFrame({-1: spin}, dtype=np.float64)
    nlist = np.zeros([N,4], dtype=np.int64)
    observable=measure(J,spin,nlist,N,Nobs)
    dfobsr = pd.DataFrame({-1: observable})

    Padd = 1.0 - np.exp(-2.0*abs(J)/T) ## Wolff algorithm (ferromagnetic)
    
    initialize(nlist,L,N)
        
    cluster = 0

    mcfname = 'Wolff/L%i/spinconf_T%.4f.csv'%(L,T)
    obsfname = 'Wolff/L%i/observation_T%.4f.csv'%(L,T)
    ## warmup steps
    print('T:',T)
    print('== warmup ==')
    for i in range(mc_warmup):
        for k in range(1,N+1):
            wolff_step(spin,nlist,N,Padd)
        dfconf[i] = spin
    ## measure steps
    print('== measure ==')
    for j in range(mc_warmup, mc_warmup+mc_measure):
        for k in range(1,N+1):
            s, _ = wolff_step(spin,nlist,N,Padd)
            cluster = cluster + s
            observable=measure(J,spin,nlist,N,Nobs)
        dfconf[j] = spin
        dfobsr[j] = observable

    mean_cluster = 1.0 * cluster / (mc_measure*N*N)
    np.savetxt('Wolff/L%i/backupspin_T%.4f.txt'%(L,T), spin)
    ## store
    dfconf.to_csv(mcfname)
    dfobsr.to_csv(obsfname)

    return mean_cluster, dfobsr

if __name__ == "__main__":
    import os
    ##parameters
    L = 40

    if os.path.exists('Wolff/L%i'%L):
        None
    else:
        os.makedirs('Wolff/L%i'%L)
    N = L**2
    J = -1 # ferro
    nt = 31
    Ts = np.linspace(1.50, 3.30, nt);
    wm = 512//4
    mm = 1024//8
    Nobs = 4
    ## main program
    for T in Ts:
        mcluster_size, obsr = ising2dmc(J,T,L,N,Nobs,wm,mm)