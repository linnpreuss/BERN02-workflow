#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 01:36:36 2026

@author: linnpreussjelvez
"""

import numpy as np
import matplotlib.pyplot as plt

class TwoSiteParameters:
    def __init__(self, J=1.0, V1=0.0, V2=0.0, dt=0.01, T=10.0):
        self.J = J
        self.V1 = V1
        self.V2 = V2
        self.dt = dt
        self.T = T
        self.Nx = 2
        
        # for the grid
        self.x = np.array([0, 1])
        self.dx = 1.0
        self.L = 1.0
    
    def psiZero(self):
        return np.array([1.0, 0.0], dtype=complex)

class TwoSiteCrankNicolson:
    def __init__(self, parameters):
        self.p = parameters
        self.psi = parameters.psiZero()
        self.t = 0
        
        H = np.array([[self.p.V1, -self.p.J], 
                      [-self.p.J, 0]], dtype=complex)
        
        I = np.eye(2, dtype=complex)
        
        self.A = I + (0.5j * self.p.dt * H)
        self.B = I - (0.5j * self.p.dt * H)
        self.A_inv = np.linalg.inv(self.A)
        self.M = self.A_inv @ self.B
    
    def step(self):
        self.psi = self.M @ self.psi
        self.t += self.p.dt
    
    def getPsi(self):
        return self.psi.copy()
        
    def getPsi2(self):
        return np.abs(self.psi)**2
    
    def probability(self):
        return self.getPsi2()
    
    def trans_refl(self):
        # transmission = site 2, reflection = site 1
        psi2 = self.getPsi2()
        return psi2[1], psi2[0]
    
    def normalize(self):
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.p.dx)
        if norm != 0:
            self.psi /= norm
        return self

parameters = TwoSiteParameters(J=1.0,V1=2.0, V2=0.0, dt=0.1, T=10.0)
solver = TwoSiteCrankNicolson(parameters)

times = []
probability = []

while solver.t < parameters.T:
    times.append(solver.t)
    probability.append(solver.getPsi2())
    solver.step()

times = np.array(times)
probability = np.array(probability)

#%%  probabilities
plt.figure(figsize=(10, 6))
plt.plot(times, probability[:, 0], color='blue', label='Site 1 (x=0)', linewidth=2)
plt.plot(times, probability[:, 1], color='red', label='Site 2 (x=1)', linewidth=2)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.ylim(0, 1)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.title(f'Time evolution')
plt.tight_layout()
plt.show()


#%% 2 sites - probability as function of x at different t
plt.figure(figsize=(12, 8))

time_indices = [0, 10, 20, 50, 100]
#time_indices = [0, 1, 5,10, 15]
time_labels = [f't={times[i]:.1f}' for i in time_indices]

for i, (idx, label) in enumerate(zip(time_indices, time_labels)):
    plt.subplot(2, 3, i+1)
    
    x = parameters.x
    psi2 = probability[idx]
    
    bars = plt.bar(x, psi2, width=0.4, color=['blue', 'red'], alpha=0.7, edgecolor='black')
    
    for bar, prob in zip(bars, psi2):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                f'{prob:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.ylim(0, 1.0)
    plt.ylabel('probability')
    plt.title(f'{label}')
    plt.grid(True, alpha=0.3, axis='y')
    

plt.suptitle(f'2-site probability', 
             fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
#%% different potentials

potentials=[0,1,2,3,4,5,6,7,8]

x_positions=[0,1]

plt.figure(figsize=(10, 7))

for V in potentials:
    parameters = TwoSiteParameters(V1=V)
    solver = TwoSiteCrankNicolson(parameters)
    
    while solver.t < parameters.T:
        solver.step()
    
    psi2 = solver.getPsi2()
    V=np.array(V)
    
    plt.bar(V,psi2, width=0.15, color='black', alpha=0.7, label=f'V={V}',edgecolor='black')
    plt.xlabel('potential V1')
    plt.ylabel('probability')






