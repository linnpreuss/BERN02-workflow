#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 09:08:54 2026

@author: linnpreussjelvez
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson


df = pd.read_csv("bird_count.csv")

count=df["count"]
year=df["yr"]

year_div=year/np.max(year) #because overflow
#print(year_div)
y=count
x= np.column_stack([np.ones(len(year)), year_div])


def neg_loglikelihood(beta,x,y):
    
    lam= np.exp(x @ beta)
    return -np.sum(-lam + y*np.log(lam))

beta_init=np.zeros(2)
result = minimize(neg_loglikelihood, beta_init, args=(x, y))

beta_hat=result.x
print(beta_hat)

def predict(x0, beta_hat):
    lam_hat= np.exp(beta_hat[0] + beta_hat[1]*x0)
    #print(beta_hat[0], beta_hat[1])
    #print(lam_hat)
    prediction=poisson.rvs(mu=lam_hat, size=3)
    print(prediction)
    mean_prediction= np.mean(prediction) #for plotting
    #print(mean_prediction)
    #y_hat=predict(x0, beta_hat)
    return prediction


x0_grid = np.linspace(year_div.min(), year_div.max(), 100)
x0_grid = year_div   
y_hat_grid = [predict(x0, beta_hat) for x0 in x0_grid]


plt.scatter(year,count, label="data", color="lightblue")
plt.plot(x0_grid*np.max(year), y_hat_grid, label="prediction", color="black" )
plt.legend()
plt.xlabel("year")
plt.ylabel("count")
plt.show()


results = []

for x0, yr, c in zip(year_div, year, count):
    prediction = predict(x0, beta_hat)

    results.append({
        "year": yr,
        "prediction_1": prediction[0],
        "prediction_2": prediction[1],
        "prediction_3": prediction[2]
    })

results = pd.DataFrame(results)

results.to_csv("BERN02_exercise2.csv", index=False)