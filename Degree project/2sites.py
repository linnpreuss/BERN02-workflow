#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 01:36:36 2026

@author: linnpreussjelvez
"""

import numpy as np
import matplotlib.pyplot as plt

class TwoSiteParameters:
    def __init__(self, J=1.0, V=0.0, dt=0.01, T=10.0):
        self.J = J
        self.V=V
        self.dt=dt
        self.T=T
        self.Nx=2
        
        #for the grid
        self.x=np.array([0,1])
        self.dx=1.0
        self.L=1.0
    
    def psiZero(self):
        return np.array([1.0,1.0],dtype=complex) /np.sqrt(2)
    
    def potential(self):
        return np.array([0.0,self.V], dtype=complex)
    
    
    

class TwoSiteCrankNicolson:
    def __init__(self,parameters):
        self.p=parameters
        self.psi=np.array([1.0,1.0],dtype=complex) /np.sqrt(2)
        self.t=0
        
        H= np.array([[self.p.V,-self.p.J],[-self.p.J,0]],dtype=complex)
        
        I = np.eye(2,dtype=complex)
        
        self.A= I +(0.5j * self.p.dt * H)
        self.B= I- (0.5j*self.p.dt *H)
        self.A_inv=np.linalg.inv(self.A)
        self.M=self.A_inv @ self.B
        
    def step(self):
        self.psi = self.M @ self.psi
        self.t+=self.p.dt
    
    def getPsi(self):
        return self.psi.copy()
        
    def getPsi2(self):
        return np.abs(self.psi)**2
    
    def  probability(self):
        return self.getPsi2()
    
    def trans_refl(self):
        #transmission= site 2, reflection=site 1
        psi2= self.getPsi2()
        return psi2[1],psi2[0]
    
    def normalize(self):
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.p.dx)
        if norm != 0:
            self.psi /= norm
    
    
    
parameters=TwoSiteParameters(J=1.0,V=5.0,dt=0.1,T=20.0)
solver = TwoSiteCrankNicolson(parameters)

times = []
#psi=[]
probability=[]

while solver.t<parameters.T:
    times.append(solver.t)
   # psi.append(solver.getPsi())
    probability.append(solver.getPsi2())
    solver.step()
    
times = np.array(times)
#psi=np.array(psi)
probability=np.array(probability)

#%% probability in each site

plt.plot(times, probability[:,0], color='blue', label='site 1')
plt.plot(times, probability[:,1], color='red', label='site 2')
plt.xlabel('time')
plt.ylabel('probability')
plt.ylim(0,1)
plt.legend()
plt.grid(True, alpha=0.5)

#%%

plt.figure(figsize=(8,6))


points = [0, len(times)//4, len(times)//2, 3*len(times)//4, -1]
colors = ['blue', 'green', 'red', 'purple', 'orange']

for idx, color in zip(points, colors):
    x = parameters.x
    psi2 = probability[idx]
    t = times[idx]
    
    plt.plot(x, psi2, 'o-', color=color, linewidth=2, markersize=8, label=f't={t:.1f}')

plt.xlabel('x')
plt.show()

#%% probability over x, different t
import matplotlib.patches as patches


fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

times_in_steps = [0, 10, 25, 50, 100]

for idx, t_steps in enumerate(times_in_steps[:4]):  
    solver.psi = parameters.psiZero()
    solver.psi = solver.normalize()
    solver.t = 0
    
    for i in range(t_steps):
        solver.step()
    
    ax = axes[idx]
    
    x_pos = parameters.x
    psi2 = solver.getPsi2()
    
    bars = ax.bar(x_pos, psi2, width=0.3, color='blue', alpha=0.7)
    
    for bar, prob in zip(bars, psi2):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{prob:.3f}', ha='center', va='bottom', fontsize=10)
    
    if parameters.V > 0:
        ax.axvspan(0.85, 1.15, alpha=0.2, color='red', label=f'V={parameters.V}')
        
        
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel('x')
    ax.set_ylabel('probability')
    ax.set_title(f'{t_steps} steps (t = {solver.t:.2f})')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Site 1 (x=0)', 'Site 2 (x=1)'])
    ax.grid(True, alpha=0.3, axis='y')
    
    trans, refl = solver.computeTransRef()
    ax.text(0.02, 0.95, f'T: {trans:.4f}\nR: {refl:.4f}', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle(f'Two-site system evolution: J={parameters.J}, V={parameters.V}, dt={parameters.dt}', 
             fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

    