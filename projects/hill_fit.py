# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 11:42:41 2019

@author: jliv
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

data = pd.read_csv("path/to/file/company_data.csv")

data['c1_cume'] = data['c1'].cumsum()
data['c2_cume'] = data['c2'].cumsum()


def HILL(x,k,s,c):
    h = c/(1+(x/k)**(-s))
    return h


def normalize(d):
    d = np.array(d)
    dnorm = (d - min(d))/(max(d)-min(d))
    min_norm = min(d)
    max_norm = max(d)
    return(dnorm,min_norm,max_norm)

def de_normalize(dnorm,min_norm,max_norm):
    dnorm = np.array(dnorm)
    d = dnorm*(max_norm - min_norm)+min_norm
    return(d)

#This fit function normalizes the target Y between 0 and 1 (ynorm)
#Calculates the Hill function of the initial k,s,c
#Evaluates the error of Hill from ynorm
#Updates estimates of k,s,c using learning rate a and the partial derivative of k,s,c

def HILL_fit(d,epoch = 200, a = .1,init = 0, s = 1.9, k_perc = 2, c_init = 1.3):
    y = d
    k = len(d)/k_perc
    s = s
    c = c_init
    ynorm,min_norm,max_norm = normalize(y)
    for j in range(epoch):
        for i in range(1+init,len(ynorm)+1+init):
            h = HILL(i,k,s,c)
            grad_k = c*((1/((1+(i/k)**(-s))**(2)))*(s)*(i/k)**((-s-1)))*(-i)/(k**2)
            grad_s = c*(1/((1+(i/k)**(-s))**2))*math.log(i/k)*(i/k)**(-s)
            grad_c = 1/((1+(i/k)**(-s)))
            e = ynorm[i-1-init] - h
            k = k + a*3*grad_k*e
            s = s + a*grad_s*e
            c = c + a*grad_c*e
    
    fit = []
    for i in range(1+init,len(d)+1+init):
        h_pred = HILL(i,k,s,c)
        fit.append(h_pred)
    
    return fit,k,s,c,min_norm,max_norm


def hill_proj(k,s,p, c,init = 50):
    k = k
    s = s
    projection = []
    for i in p:
        h = HILL(i+init+1,k,s,c)
        projection.append(h)
    return(projection)

hillfit,k,s,c,min_norm,max_norm = HILL_fit(d = data['c2_cume'],epoch = 2000,init = 25, s = 1.3,k_perc = 1)
plt.plot(de_normalize(hillfit,min_norm,max_norm))
plt.plot(data['c2_cume'])


hillfit,k,s,c,min_norm,max_norm = HILL_fit(d = data['c1_cume'],epoch = 2000,init = 25, s = 1.3,k_perc = 1)
plt.plot(de_normalize(hillfit,min_norm,max_norm))
plt.plot(data['c1_cume'])



proj_hil = de_normalize(hill_proj(k,s,[25,50,75,100,125,150,170,200,225,250,275,300],init = 25, c=c),min_norm,max_norm)
plt.scatter([25,50,75,100,125,150,170,200,225,250,275,300],proj_hil)
plt.plot(data['c1_cume'])