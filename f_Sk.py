# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 12:14:33 2022
WLC scattering functions based on the MATLAB code modified by Guan-Rong Huang Oct. 16 2019
@author: CHTUNG
"""

import numpy as np

def AlphaSquare(x):
    if x > 10:
        y = (1 + (x/3.12)**2 + (x/8.67)**3)**(0.176/3)
    else:
        y = (1 + (x/3.12)**2 + (x/8.67)**3)**(0.170/3);
        
    return y

def Rgsquarezero(q,L,b):
    y = ( L*b/6 ) * ( 1 - 1.5*(b/L) + 1.5*(b/L)**2 - 0.75*(b/L)**3*( 1 - np.exp(-2*(L/b) ) ) );
    return y

def Rgsquareshort(q,L,b):
    y = AlphaSquare(L/b)*Rgsquarezero(q,L,b)
    
    return y

def Rgsquare(q,L,b):
    y = AlphaSquare(L/b)*L*b/6
    
    return y

def u(q,L,b):
    y = Rgsquare(q,L,b)*q**(2)
    
    return y

def u1(q,L,b):
    y = Rgsquareshort(q,L,b)*q**(2)
    
    return y

def Sdebye(q,L,b):
    y = 2*(np.exp(-u(q,L,b)) + u(q,L,b) -1) / ( (u(q,L,b))**2 )
    
    return y

def Sdebye1(q,L,b):
    y = 2*(np.exp(-u1(q,L,b)) + u1(q,L,b) -1)/((u1(q,L,b))**2)
    
    return y

def w(x):
    y = 0.5*( 1 + np.tanh( (x - 1.523)/0.1477 ) )
    
    return y

def Sexv(q,L,b):
    C1=1.22
    C2=0.4288
    C3=-1.651
    miu = 0.585

    y = ((1 - w(q*np.sqrt(Rgsquare(q,L,b))))*Sdebye(q,L,b) + 
         w(q*np.sqrt(Rgsquare(q,L,b)))*(C1*(q*np.sqrt(Rgsquare(q,L,b)))**(-1/miu) + 
                                       C2*(q*np.sqrt(Rgsquare(q,L,b)))**(-2/miu) +
                                       C3*(q*np.sqrt(Rgsquare(q,L,b)))**(-3/miu)))
    
    return y

def Sexvnew(q,L,b):
    C1=1.22
    C2=0.4288
    C3=-1.651
    miu = 0.585
    C_star2 = np.zeros(q.shape[0])
    # Modified by Guan-Rong Huang Oct. 17 2019
    for i in range(q.shape[0]-1):
        if Sexv(q[i+1],L,b)>=Sexv(q[i],L,b):
            C_star2[i] = 0
        else:
            C_star2[i] = 1

    C_star2[q.shape[0]-1] = C_star2[q.shape[0]-2]
    
    y = ((1 - w(q*np.sqrt(Rgsquare(q,L,b))))*Sdebye(q,L,b) + 
         C_star2*w(q*np.sqrt(Rgsquare(q,L,b)))*(C1*(q*np.sqrt(Rgsquare(q,L,b)))**(-1/miu) + 
                                                C2*(q*np.sqrt(Rgsquare(q,L,b)))**(-2/miu) + 
                                                C3*(q*np.sqrt(Rgsquare(q,L,b)))**(-3/miu)))
    
    return y

def a1long(q, L, b, p1, p2, q0):
    if L/b > 10:
        C = 3.06/(L/b)**0.44
    else:
        C = 1
        
    E = np.exp(1)
    C1 = 1.22
    C2 = 0.4288
    C3 = -1.651
    C4 = 1.523
    C5 = 0.1477
    miu = 0.585
    m1 = -1/miu
    m2 = -2/miu
    m3 = -3/miu
    q02 = q0**2
    q03 = q0**3
    q04 = q0**4
    q05 = q0**5
    b2 = b**2
    b3 = b**3
    b4 = b**4
    b5 = b**5
    Rgs = Rgsquare(q,L,b)
    z0 = (q02*Rgs)/b2
    z1 = q02*Rgs
    z = np.exp(-z0)
    Rg = np.sqrt(Rgs)
    q0Rg = Rg*q0
    qRb = q0Rg/b
    qRb45 = (qRb - C4)/C5
    
    y = (q0**p1*(-b*np.pi/(L*q0) + 
                 (b*C*(4/15 - z*(11/15 + 7*b2/(15*z1)) + 7*b2/(15*z1)))/L + 
                 (2*b4*(z + z0 - 1)*(1 - 0.5*(1 + np.tanh(qRb45))))/(q04*Rgs**2) + 
                 0.5*(C3*qRb**m3 + C2*qRb**m2 + C1*qRb**m1)*(1 + np.tanh(qRb45))) + 
         (1/(b*p1*q0**(-1 - p1 - p2) - b*p2*q0**(-1 - p1 - p2)))*
         q0**(p1 - p2)*(-q0**(-p1)*(b2*np.pi/(L*q02) + 
                                    (b*C*(-14*b3/(15*q03*Rgs) + 14*b3*z/(15*q03*Rgs) + (2*z*q0*(11/15 + 7*b2/(15*q02*Rgs))*Rgs)/b))/L + 
                                    (Rg*(C3*qRb**m3 + C2*qRb**m2 + C1*qRb**m1)*1/np.cosh(qRb45)**2)/(2*C5) - 
                                    (b4*Rg*(z + z0 - 1)*1/np.cosh(qRb45)**2)/(C5*q04*Rgs**2) + 
                                    (2*b4*(2*q0*Rgs/b - (2*z*q0*Rgs)/b)*(1 - 0.5*(1 + np.tanh(qRb45))))/(q04*Rgs**2) - 
                                    (8*b5*(-1 + z + z1/b2)*(1 + 0.5*((-1 - np.tanh(qRb45)))))/(q05*Rgs**2) + 
                                    0.5*(-3*C3*Rg*qRb**(-1 + m3)/miu - 2*C2*Rg*qRb**(-1 + m2)/miu - C1*Rg*qRb**(-1 + m1)/miu)*(1 + np.tanh(qRb45))) - 
                        b*p1*q0**(-1 - p1)*(-b*np.pi/(L*q0) + 
                                            (b*C*(4/15 - z*(11/15 + 7*b2/(15*z1)) + 7*b2/(15*z1)))/L + 
                                            (2*b4*(z + z0 - 1)*(1 + 0.5*(-1 - np.tanh(qRb45))))/(q04*Rgs**2) + 
                                            0.5*(C3*qRb**m3 + C2*qRb**m2 + C1*qRb**m1)*(1 + np.tanh(qRb45)))))
    
    return y

def a2long(q, L, b, p1, p2, q0):
    if L/b > 10:
        C = 3.06/(L/b)**0.44
    else:
        C = 1
    
    C1 = 1.22
    C2 = 0.4288
    C3 = -1.651
    C4 = 1.523
    C5 = 0.1477
    miu = 0.585
    m1 = -1/miu
    m2 = -2/miu
    m3 = -3/miu
    q02 = q0**2
    q03 = q0**3
    q04 = q0**4
    q05 = q0**5
    b2 = b**2
    b3 = b**3
    b4 = b**4
    b5 = b**5
    Rgs = Rgsquare(q,L,b)
    z0 = (q02*Rgs)/b2
    z1 = q02*Rgs
    z = np.exp(-z0)
    Rg = np.sqrt(Rgs)
    q0Rg = Rg*q0
    qRb = q0Rg/b
    qRb45 = (qRb - C4)/C5
    
    y = (-q0**(-p1)*(b2*np.pi/(L*q02) + 
                     (b*C*(-14*b3/(15*q03*Rgs) + (14*b3*z)/(15*q03*Rgs) + (2*z*q0*(11/15 + (7*b2)/(15*z1))*Rgs)/b))/L + 
                     (Rg*(C3*qRb**m3 + C2*qRb**m2 + C1*qRb**m1)*1/np.cosh(qRb45)**2)/(2*C5) - 
                     (b4*Rg*(z + z0 -1)*1/np.cosh(qRb45)**2)/(C5*q04*Rgs**2) + 
                     (2*b4*(2*(1-z)*q0*Rgs/b)*(1 - 0.5*(1 + np.tanh(qRb45 ))))/(q04*Rgs**2) - 
                     (8*b5*(z + z0 - 1)*(1 - 0.5*(1 + np.tanh(qRb45))))/(q05*Rgs**2) + 
                     0.5*(-(3*C3*Rg*qRb**(- 1 + m3))/miu - (2*C2*Rg*qRb**(- 1 + m2))/miu - (C1*Rg*qRb**(-1 + m1))/miu)*(1 + np.tanh(qRb45))) - 
         b*p1*q0**(- 1 - p1)*(-b*np.pi/(L*q0) + 
                              (b*C*(4/15 - z*(11/15 + 7*b2/(15*z1)) + (7*b2)/(15*q02* Rgs)))/L + 
                              (2*b4*(z + z0 - 1)*(1 - 0.5*(1 + np.tanh(qRb45))))/(q04* Rgs**2) + 
                              0.5*(C3*qRb**m3 + C2* qRb**m2 + C1*qRb**m1)*(1 + np.tanh(qRb45))))
    
    y = -1/(b*(p1 - p2)*q0**(-1 - p1 - p2))*y; 
    
    return y

def a1short(q, L, b, p1, p2, q0):
    Rgs = Rgsquareshort(q,L,b)
    q02 = q0**2
    q03 = q0**3
    b2 = b**2
    b3 = b**3
    z = np.exp(q02*Rgs/b2)
    
    y = (8*b3*L - 8*b3*z*L - 2*b3*L*p2 + 
        2*b3*z*L*p2 + 4*b*L*q02*Rgs + 
        4*b*z*L*q02*Rgs - 2*b*z*L*p2*q02*Rgs - 
        z*np.pi*q03*Rgs**2 + z*p2*np.pi*q03*Rgs**2)
    
    y = y/(L*(p1 - p2)*Rgs**2)*b/z*q0**(p1 - 4)
    
    return y

def a2short(q, L, b, p1, p2, q0):
    Rgs = Rgsquareshort(q,L,b)
    q02 = q0**2
    q03 = q0**3
    b2 = b**2
    b3 = b**3
    z = np.exp(q02*Rgs/b2)
    
    y = (8*b3*L - 8*b3*z*L - 2*b3*L*p1 + 2*b3*z*L*p1 + 
         4*b*L*q02*Rgs + 4*b*z*L*q02*Rgs - 2*b*z*L*p1*q02*Rgs - 
         z*np.pi*q03*Rgs**2 + z*p1*np.pi*q03*Rgs**2)

    y = -b/z*q0**(p2 - 4)*y/(L*(p1 - p2)*Rgs**2)
    
    return y