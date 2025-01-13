# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 11:51:50 2022
WLC scattering functions based on the MATLAB code modified by Guan-Rong Huang Oct. 16 2019
@author: CHTUNG
"""

import numpy as np
import f_Sk
# from numba import jit
from scipy.special import sici as sici


def Sk(q,L,b):
    p1 = 4.12
    p2 = 4.42
    p1short = 5.36
    p2short = 5.62
    q0 = 3.1
    
    #
    len_q = q.shape[0]
    i=np.arange(len_q)
    q0short = np.zeros(len_q)
    q0short[i] = np.max((1.9*b/np.sqrt(f_Sk.Rgsquareshort(q[i],L,b)),3))
    
    #
    if L/b > 10:
        C = 3.06/(L/b)**0.44
    else:
        C = 1

    #%% S
    S = np.zeros(len_q)
    for i in range(len_q):
        if L > 4*b: # Longer Chains
            if q[i]*b <= 3.1:
                # Modified by Yun on Oct. 15,
                Sexvmodify = f_Sk.Sexvnew(q, L, b)
                
                S[i] = (Sexvmodify[i] + C*(4/15 + 7/(15*f_Sk.u(q[i],L,b)) - 
                                           (11/15 + 7/(15*f_Sk.u(q[i],L,b)))*np.exp(-f_Sk.u(q[i],L,b)))*(b/L))
                # End of modification
                
            else: #q[i]*b > 3.1
                S[i] = (f_Sk.a1long(q[i], L, b, p1, p2, q0)/((q[i]*b)**(p1)) + 
                        f_Sk.a2long(q[i], L, b, p1, p2, q0)/((q[i]*b)**(p2)) +
                        np.pi/(q[i]*L))
                
        else: # L <= 4*b Shorter Chains
        # Modified by Guan-Rong Huang Oct. 16 2019
            if q[i]*b <= max((1.9*b/np.sqrt(f_Sk.Rgsquareshort(q[i],L,b)),3)):
                if q[i]*b<=0.01:
                    S[i] = 1 - f_Sk.Rgsquareshort(q[i],L,b)*(q[i]**(2))/3
                else:
                    S[i] = f_Sk.Sdebye1(q[i],L,b)
                    
            else: #q[i]*b > max((1.9*b/np.sqrt(f_Sk.Rgsquareshort(q[i],L,b)),3))
                S[i] = (f_Sk.a1short(q[i],L,b,p1short,p2short,q0short[i])/((q[i]*b)**(p1short)) + 
                        f_Sk.a2short(q[i],L,b,p1short,p2short,q0short[i])/((q[i]*b)**(p2short)) +
                        np.pi/(q[i]*L))
                
    return S


def S_rod(q,L):
    y = 2*sici(q*L)[0]/(q*L) - 4*np.sin(q*L/2)**2/(q*L)**2

    return y                