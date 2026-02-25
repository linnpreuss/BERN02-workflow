#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 13:42:18 2026

@author: linnpreussjelvez
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import factorized

# I only give it L

class TwoSiteParameters:
    def __init__(self, J=1.0, V=0.0, dt=0.01, T=20.0, Nx=1000, L=30):
        self.J = J
        self.dt = dt
        self.T = T
        self.Nx=Nx
        self.L=L
        self.V=np.zeros(Nx)
        
        # for the grid
        self.x = np.linspace(-L/2, L/2, Nx)
        self.dx = self.x[1]-self.x[0]
        
    
    def psiZero(self):
        sigma = 0.5
        k0 = 5.0     
        x0 = -2.0   
        psi = np.exp(-(self.x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * self.x)
        #norm= (np.pi*sigma**2)**(1/4)
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
        return psi 
    
    
    def barrier(self, center=0.0, width=1.0, height=1.0):
        self.V=np.zeros(self.Nx)
        self.width=width
        self.center=center
        self.height=height
        for i in range(self.Nx):
            if abs(self.x[i] -center) <width/2:
                self.V[i] = height
        return self.V
    

        

class TwoSiteCrankNicolson:
    def __init__(self, parameters):
        self.p = parameters
        self.psi = parameters.psiZero()
        self.t = 0
        
        diagonalA =[]
        diagonalB = []
        for i in range(self.p.Nx):
            diag_A=1 + 1j * self.p.dt / self.p.dx**2 - 1j * self.p.dt * self.p.V[i]
            diag_B=1 - 1j * self.p.dt / self.p.dx**2 + 1j * self.p.dt * self.p.V[i]
            diagonalA.append(diag_A)
            diagonalB.append(diag_B)
        
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

parameters = TwoSiteParameters()
    
parameters.barrier()

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
position=np.array(position)



# wavefunction at diff times
plt.figure(figsize=(10, 6))
times_parts= [0, 50, 100, 150]
for i in times_parts:  
    plt.plot(parameters.x, probability[i,:], label=f't= {i}')
    
#plt.plot(parameters.x, parameters.V, color='red') 
V_scaled=parameters.V / np.max(parameters.V)
plt.fill_between(parameters.x, 0, V_scaled, 
                 alpha=0.2, color='red')

plt.xlabel('Position x')
plt.ylabel('Probability Density ')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Wavefunction ')
plt.show()


'''
for i in range(parameters.x):
    plt.plot(times, probability[:,i], label=(f'site= {i+1}'))
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.ylim(0, 1)
    #plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title(f'Time evolution - length {i+1}')
plt.show()
'''


