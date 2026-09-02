#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 10:14:56 2026

@author: linnpreussjelvez
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize



df = pd.read_csv("pollution_cleaneddata.csv")

df = df.sort_values(by="POOR")

poor=df["POOR"]
mort=df["MORT"]

#print(poor)

def regression(x0,k=10):
    
    distances=[]
    for x in poor:
        distance = np.abs(x0-x)
        distances.append(distance)
        
    idx=np.argsort(distances)[:k]
    used_distances=np.array(distances)[idx]
    used_poor=np.array(poor)[idx]
    used_mort=np.array(mort)[idx]
    
     #tricube for weight
    weight= (1-(used_distances/used_distances.max())**3)**3
    
    #print(used_distances)
    #print(used_poor)
    
    def minimization(x,y,weight):
        def beta(beta):
            beta0,beta1=beta
            residuals=y-beta0-beta1*x
            return np.sum(weight*residuals**2)
         
        result=minimize( beta, x0=[0,0])
        beta0,beta1=result.x
        return beta0, beta1
    beta0,beta1=minimization(used_poor,used_mort,weight)
    y_hat=beta0+beta1*x0
    
    y_fitted=beta0+beta1*x0
    residuals=used_mort-y_fitted
    
    sse = np.sum(weight*residuals**2)
    s=np.sqrt(sse/(k-2))
    x_bar=np.mean(used_poor)
    se= s*np.sqrt((1/k)+ (x0-x_bar)**2/np.sum((used_poor-x_bar)**2))
    return y_hat, se
    
    


for x0 in [10,18,25]:
    y_hat, se= regression(x0)
    print(f"POOR = {x0}%: predicted MORT = {y_hat:.2f}, SE = {se:.2f}")
    
x_grid=np.linspace(poor.min(), poor.max(),200)
y_grid=[]
for x0 in x_grid:
    y_hat,se = regression(x0)
    y_grid.append(y_hat)
    
plt.scatter(poor,mort,label="data", color="lightblue")
plt.plot(x_grid,y_grid,label="local regression", color="black")
plt.xlabel("POOR (%)")
plt.ylabel("MORT")
plt.legend()
plt.ylim(700,1300)
plt.show()
    
    