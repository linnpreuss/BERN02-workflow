#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 15:43:04 2026

@author: linnpreussjelvez
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import factorized

class TwoSiteParameters:
    def __init__(self, J=1.0, V1=0.0, V2=0.0, dt=0.01, T=10.0):
        self.J = J
        self.V1 = V1
        self.V2 = V2
        self.dt = dt
        self.T = T
        self.Nx = 5
        
        # for the grid
        self.x = np.array([0, 1])
        self.dx = 1.0
    
    def psiZero(self):
        return np.array([1.0, 0.0, 0.0,0.0,0.0], dtype=complex)

class TwoSiteCrankNicolson:
    def __init__(self, parameters):
        self.p = parameters
        self.psi = parameters.psiZero()
        self.t = 0
        
        
        diagonalA = 1 + 1j * self.p.dt / self.p.dx**2 - 1j * self.p.dt * self.p.V1
        diagonalB = 1 - 1j * self.p.dt / self.p.dx**2 + 1j * self.p.dt * self.p.V1
        offDiagonalA = -0.5j * self.p.dt / self.p.dx**2
        offDiagonalB = 0.5j * self.p.dt / self.p.dx**2

        A = diags([offDiagonalA, diagonalA, offDiagonalA], offsets=[-1, 0, 1], 
                  shape=(self.p.Nx, self.p.Nx), format='csr')
        B = diags([offDiagonalB, diagonalB, offDiagonalB], offsets=[-1, 0, 1], 
                  shape=(self.p.Nx, self.p.Nx), format='csr')
        
        self.solveA = factorized(A)
        self.B = B
    
    def step(self):
        if self.t >= self.p.T:
            return False

        rhs = self.B @ self.psi
        self.psi = self.solveA(rhs)
        self.t += self.p.dt
        return True
    
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

parameters = TwoSiteParameters(J=5.0,V1=2.0, V2=0.0, dt=0.1, T=10.0)
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
plt.plot(times, probability[:, 0], color='blue', label='Site 1', linewidth=2)
plt.plot(times, probability[:, 1], color='red', label='Site 2 ', linewidth=2)
plt.plot(times, probability[:, 2], color='orange', label='Site 3', linewidth=2)
plt.plot(times, probability[:, 3], color='pink', label='Site 4', linewidth=2)
plt.plot(times, probability[:, 4], color='green', label='Site 5', linewidth=2)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Time evolution')
plt.tight_layout()
plt.show()


#%% 2 sites - probability as function of x at different t
'''
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
'''
#%% different potentials

potentials=[15,16,17,18,19,20]

x_positions=[0,1]

plt.figure(figsize=(10, 7))

for V in potentials:
    parameters = TwoSiteParameters(V1=V)
    solver = TwoSiteCrankNicolson(parameters)
    
    while solver.t < parameters.T:
        solver.step()
    
    psi2 = solver.getPsi2()
    V=np.array(V)
    
    plt.bar(V,psi2, width=0.15, color='black', alpha=0.6, label=f'V={V}',edgecolor='black')
    plt.xlabel('potential V1')
    plt.ylabel('probability')






