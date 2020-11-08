# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools


def generate(transition, emission, init, N):
    seq = np.zeros(N,dtype='object') #sequence of size n
    st = np.zeros(N+1,dtype='object') #state sequence of size n
    st[0] = np.random.choice(list(init.columns), 1, p=list(init.iloc[0]))[0]
    for i in range(N):
        seq[i] = np.random.choice(list(emission.columns), 1, p=list(emission.loc[st[i]]))[0] #emission
        st[i+1] = np.random.choice(list(transition.columns), 1, p=list(transition.loc[st[i]]))[0] #transition
    st = np.delete(st, -1)
    return np.sum(st), np.sum(seq)


def transition_Durbin_CpG(delta,tau):
    #import Durbin et al.'s proto transition matrix
    proto_transition = pd.read_csv("cpg_hmm_2/proto_transition.csv",index_col=0)
    transition = proto_transition.copy()
    #derive transition matrix
    transition.iloc[:4,:4] = proto_transition.iloc[:4,:4]*(1-tau)
    transition.iloc[4:,4:] = proto_transition.iloc[4:,4:]*(1-delta)
    transition.iloc[4:,:4] = proto_transition.iloc[4:,:4]*delta
    transition.iloc[:4,4:] = proto_transition.iloc[:4,4:]*tau
    return transition


def viterbi(seq, transition, emission, init, disp_v = False, disp_alignment=False):
    
    #convert sequence to array
    seq = np.array(list(seq), dtype='object')
    N = np.size(seq)
    
    #constructing matrices
    k = np.shape(emission)[0]
    V = np.zeros([k,N]) #viterbi matrix
    tracer = np.zeros([k,N],dtype='object')
    
    #init
    V[:,0] = np.log(init*emission[seq[0]])
    
    #fill up
    for j in range(1,N):
        for i in range(0,k):
            possibles = V[:,j-1]+np.log(transition.iloc[:,i])+np.log(emission[seq[j]][i])
            V[i,j] = np.max(possibles)
            tracer[i,j-1] = (np.array(transition.columns[np.where(possibles==np.max(possibles))])[0])
    V = np.exp(V)
    
    #last column of tracer
    tracer[k-1,N-1] = np.array(transition.columns[np.where(V[:,N-1]==np.max(V[:,N-1]))])[0] #final traceback
    tracer[tracer==0] = '/' #ignoring all elements of the last column except the maximum one
    
    #traceback
    st = np.zeros(N,dtype='object')
    st[-1] = tracer[-1,-1]
    for i in range(2,N+1):
        st[-i] = tracer[(np.where(np.array(transition.columns)==st[-i+1])[0][0]),-i]
    
    
    if disp_v is True:
        plt.figure(figsize=(15,1.5))
        sns.heatmap(np.log(V),linewidth = 1,cmap='Blues',yticklabels=list(transition.columns))
        plt.title("$\log(V)$")
    
    if disp_alignment is True:
        print(np.sum(st))
        print(np.sum(seq))
    
    return np.sum(st)


def forward(seq, transition, emission, init, disp_f = False):

    #convert sequence to array
    seq = np.array(list(seq), dtype='object')
    N = np.size(seq)
    
    #constructing matrices
    k = np.shape(emission)[0]
    F = np.zeros([k,N]) #forward matrix
    F[:,0] = init*emission[seq[0]] #initialize

    #fill up
    for j in range(1,N):
        for i in range(0,k):
            possibles = F[:,j-1]*(transition.iloc[:,i])*(emission[seq[j]][i])
            F[i,j] = np.sum(possibles)
            
    if disp_f is True:
        plt.figure(figsize=(15,1.5))
        sns.heatmap(np.log(F),linewidth = 1,cmap='Blues',yticklabels=list(transition.columns))
        plt.title("$\log(F)$")
    
    #calculate probability of sequence
    P = np.sum(F[:,N-1])
    
    return P,F


def backward(seq, transition, emission, init, disp_b = False):

    #convert sequence to array
    seq = np.array(list(seq), dtype='object')
    N = np.size(seq)
    
    #constructing matrices
    k = np.shape(emission)[0]
    B = np.zeros([k,N]) #forward matrix
    B[:,-1] = 1 #initialize

    #fill up
    for j in range(N-2,-1,-1):
        for i in range(0,k):
            possibles = B[:,j+1]*(transition.T.iloc[:,i])*(emission[seq[j+1]])
            B[i,j] = np.sum(possibles)
            
    if disp_b is True:
        plt.figure(figsize=(15,1.5))
        sns.heatmap(np.log(B),linewidth = 1,cmap='Blues',yticklabels=list(transition.columns))
        plt.title("$\log(B)$")
    
    #calculate probability of sequence
    P = np.sum(np.array(emission[seq[0]]*B[:,0]*init))
    
    return P,B


def forward_backward(seq, transition, emission, init, disp_f = False, disp_b = False, disp_pn = False, disp_pe = False):
    _,F = forward(seq,transition,emission,init,disp_f=disp_f)
    P,B = backward(seq,transition,emission,init,disp_b=disp_b)
    pi_node = B*F/P
    
    #constructing matrices
    seq = np.array(list(seq), dtype='object')
    k = np.shape(emission)[0]
    N = np.size(seq)
    pi_edge = np.zeros([k**2,N-1]) #pi_edge matrix

    #fill up step
    for n in range(0,N-1):
        for m,x in enumerate(itertools.product(range(k),
                                   range(k))):
            pi_edge[m,n] = F[x[0],n]*B[x[1],n+1]*transition.iloc[x[0],x[1]]*emission[seq[n+1]][x[1]]
    pi_edge = pi_edge/P
    
    if disp_pn:
        plt.figure(figsize=(8,2))
        sns.heatmap(pi_node,linewidth = 1,
                    cmap='Blues',yticklabels=list(transition.columns))
        plt.title("$\Pi^*$");
    
    if disp_pe:
        plt.figure(figsize=(2,10))
        #transitions
        transits = np.array([i for i in itertools.product(list(transition.columns),
                                                          list(transition.columns))],dtype='object')
        transits[:,0] = "$"+transits[:,0]+r" \rightarrow "; transits[:,1] = transits[:,1]+"$"
        transits = np.sum(transits,1)
        
        sns.set(font_scale=0.8)
        sns.heatmap(pi_edge,linewidth = 1,
                    yticklabels=list(transits),cbar_kws={"aspect":50},cmap='Blues')
        plt.title("$\Pi^{**}$");
        sns.set()
    
    return pi_node, pi_edge, P


def baum_welch(seq, transition_0, emission_0, init):
    
    p = 1
    P = 0.1
    
    transition_0 = transition_0.copy(); emission_0 = emission_0.copy(); init = init.copy()
    k = np.shape(emission_0)[0]

    #convergence criterion
    while np.abs(np.log(p)-np.log(P))>1e-5:
        p = P
        
        #perform forward-backward
        pi_node, pi_edge, P = forward_backward(seq, transition_0, emission_0, init)
        #print(pi_node, pi_edge)
    
        #calculate transition matrix
        transition_1 = pd.DataFrame(np.sum(pi_edge,1).reshape(k,k), columns = 
                                      transition_0.columns, index = transition_0.index)
        #sum over the rows to get values, one for each type of transition
        
        #row normalize
        transition_1 = transition_1.div(transition_1.sum(axis=1), axis=0)
        
        
        #calculate emission matrix
        emission_1 = emission_0.copy(); emission_0.iloc[:,:] = 0
        
        for i,r in enumerate(emission_0.index):
            for j,c in enumerate(emission_0.columns):
                emission_1.loc[r,c] = np.sum(pi_node[i][np.array(list(seq),dtype='object')==c])
        #sum over the rows when the seq contains a particular letter
        
        #row normalize
        emission_1 = emission_1.div(emission_1.sum(axis=1), axis=0)
       
        #print probability of sequence
        print("P(x|theta):",P)
        
        #set new HMM
        transition_0 = transition_1.copy(); emission_0 = emission_1.copy()
        
    return transition_1, emission_1
