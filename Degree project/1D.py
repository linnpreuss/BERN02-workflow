#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 13:52:05 2026

@author: linnpreussjelvez
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import factorized

class TwoSiteParameters:
    def __init__(self, J=1.0, V1=0.0, V2=0.0, dt=0.01, T=20.0, Nx=5):
        self.J = J
        self.V1 = V1
        self.V2 = V2
        self.dt = dt
        self.T = T
        self.Nx=Nx
        
        # for the grid
        self.x = np.array([0, 1])
        self.dx = 1.0
    
    def psiZero(self):
        zeros=np.zeros(self.Nx)
        zeros[0]=1
        return zeros

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

parameters = TwoSiteParameters()
solver = TwoSiteCrankNicolson(parameters)

times = []
probability = []
position=[]

while solver.t < parameters.T:
    times.append(solver.t)
    probability.append(solver.getPsi2())
    solver.step()
    position.append(parameters.x)

times = np.array(times)
probability = np.array(probability)
positions=np.array(position)


plt.figure(figsize=(10, 6))
plt.plot(position, probability[:,0])

for i in range(parameters.Nx):
    plt.plot(times, probability[:,i], label=(f'site= {i+1}'))
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.ylim(0, 1.5)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f'Time evolution - length {i+1}')
plt.show()


'''
# Plot probability vs position at specific time points
plt.figure(figsize=(10, 6))

# Select specific time indices to plot
time_indices = [0, len(times)//4, len(times)//2, 3*len(times)//4, -1]
colors = ['b', 'g', 'r', 'c', 'm']
labels = ['t=0', f't={times[len(times)//4]:.2f}', f't={times[len(times)//2]:.2f}', 
          f't={times[3*len(times)//4]:.2f}', f't={times[-1]:.2f}']

for idx, color, label in zip(time_indices, colors, labels):
    plt.plot(parameters.x, probability[idx, :], 'o-', color=color, label=label, markersize=8)

plt.xlabel('Position (site index)', fontsize=12)
plt.ylabel('Probability', fontsize=12)
plt.ylim(0, 1.1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Probability distribution at different times')
plt.show()
'''