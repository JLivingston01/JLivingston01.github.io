
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

data = pd.read_csv("path/to/data/company_data.csv")

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
    y,k,s,c = d,len(d)/k_perc,s,c_init
    ynorm,min_norm,max_norm = normalize(y)
    for j in range(epoch):
        for i in range(1+init,len(ynorm)+1+init):
            k,s,c = k + a*(c*((1/((1+(i/k)**(-s))**(2)))*(s)*(i/k)**((-s-1)))*(-i)/(k**2))*(ynorm[i-1-init] - HILL(i,k,s,c)),\
                    s + a*(c*(1/((1+(i/k)**(-s))**2))*math.log(i/k)*(i/k)**(-s))*(ynorm[i-1-init] - HILL(i,k,s,c)),\
                    c + a*(1/((1+(i/k)**(-s))))*(ynorm[i-1-init] - HILL(i,k,s,c))

    fit = []
    for i in range(1+init,len(d)+1+init):
        h_pred = HILL(i,k,s,c)
        fit.append(h_pred)
    
    return fit,k,s,c,min_norm,max_norm


def hill_proj(k,s,p, c,init = 0):
    k = k
    s = s
    projection = []
    for i in p:
        h = HILL(i+init+1,k,s,c)
        projection.append(h)
    return(projection)

hillfit,k,s,c,min_norm,max_norm = HILL_fit(d = data['c2_cume'],epoch = 12000,init = 0, s = 10,k_perc = 3)
plt.plot(de_normalize(hillfit,min_norm,max_norm))
plt.plot(data['c2_cume'])


hillfit,k,s,c,min_norm,max_norm = HILL_fit(d = data['c1_cume'],epoch = 12000,a = 1,init = 0, s = 17,k_perc = 1)
plt.figure(figsize = (8,6))
plt.plot(de_normalize(hillfit,min_norm,max_norm), label='Hill Fit')
plt.plot(data['c1_cume'], label='Cumulative KPI')
plt.title("The Fit of Hill Closely Resembles the KPI Data")
plt.legend(loc = 2)
plt.ylabel("Cumulative KPI")
plt.xlabel("Time (Months)")
plt.show()

#plt.plot((de_normalize(hillfit,min_norm,max_norm) - data['c1_cume'])/data['c1_cume'])
mag_error = (de_normalize(hillfit,min_norm,max_norm) - data['c1_cume'])
plt.figure(figsize = (8,6))
plt.plot(mag_error)
plt.ylabel("Magnitude of Model Error")
plt.xlabel("Time (Months)")
plt.title("Model Error Shows Normal Periodicity, Well Centered Near 0")
plt.show()

              
proj_hil = de_normalize(hill_proj(k,s,[25,50,75,100,125,150,170,200,225,250,275,300],init = 0, c=c),min_norm,max_norm)
plt.figure(figsize = (8,6))
plt.scatter([25,50,75,100,125,150,170,200,225,250,275,300],proj_hil, label = 'Projection using Hill')
plt.plot(data['c1_cume'], label = 'Cumulative KPI 1')
plt.ylabel("Cumulative KPI")
plt.xlabel("Time (Months)")
plt.legend(loc = 2)
plt.title("This Hill Projection May Indicate the Potential for Further Scalability")
plt.show()


hillfit,k,s,c,min_norm,max_norm = HILL_fit(d = data['c2_cume'],a = 1,epoch = 12000,init = 0, s = 1,k_perc = 3)
plt.plot(de_normalize(hillfit,min_norm,max_norm))
plt.plot(data['c2_cume'])
plt.show()

proj_hil = de_normalize(hill_proj(k,s,[25,50,75,100,125,150,170,200,225,250,275,300],init = 0, c=c),min_norm,max_norm)
plt.figure(figsize = (8,6))
plt.scatter([25,50,75,100,125,150,170,200,225,250,275,300],proj_hil, label = 'Projection using Hill')
plt.plot(data['c2_cume'], label = 'Cumulative KPI 2')
plt.ylabel("Cumulative KPI")
plt.xlabel("Time (Months)")
plt.legend(loc = 2)
plt.title("This Hill Projection May Indicate Further Scalability is Difficult")
plt.show()
