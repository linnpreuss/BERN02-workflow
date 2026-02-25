#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 14:22:42 2026

@author: linnpreussjelvez
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import factorized


            
class CrankNicolson:
    def __init__(self, Nx=500, L=30.0, dt=0.001, T=2000.0, k0=5.0):
        self.Nx=Nx
        self.L=L
        self.dt=dt
        self.T=T
        
        self.x = np.linspace(0, L, self.Nx)
        self.dx = self.x[1]-self.x[0]
        
        
        self.probability_list =[]
        self.time_list = []
        self.psi_list=[]
        
        self.t=0.0
        
        self.V = np.zeros(Nx)
        
        self.k0=k0
    
    
    def initial_psi(self, k0):
        x0=200
        sigma= 20.0
        psi=np.exp((-(self.x-x0)**2)/(2*sigma**2) ) * np.exp( 1j*k0*self.x)
        norm= (np.pi*sigma**2)**(1/4) 
        return psi /norm
         
    def barrier(self, center=200.0, width=20.0, height=5.0):
        self.width=width
        self.center=center
        self.height=height
        self.V=np.zeros(self.Nx)
        for i in range(self.Nx):
            if abs(self.x[i] -center) <width/2:
                self.V[i] = height
                
                
    def calc_a(self):
        a= np.zeros(self.Nx, dtype=complex)
        a[0]= 2*( 1+ self.dx**2 *self.V[0] - (1j*2*self.dx**2 /self.dt ))
        for k in range(1,self.Nx-1):
            a[k]=2* ( 1+ self.dx**2 *self.V[k] - 1j*2*self.dx**2/self.dt) - (1/a[k-1])
        return a
        
    def calc_omega(self, psi):
        omega=np.zeros(self.Nx, dtype=complex)
        self.psi=psi
        for k in range(1, self.Nx-1):
            omega[k]= - psi[k-1] + 2*( 1j*2*self.dx**2/self.dt +1 + self.dx**2 *self.V[k])*psi[k]-psi[k+1]
        omega[0] = 0
        omega[-1] = 0
        return omega
        
    def calc_b(self, omega, a):
        b=np.zeros(self.Nx, dtype=complex)
        self.omega=omega
        self.a=a
        b[0]=self.omega[0]
        for k in range(1,self.Nx-1):
            b[k]= b[k-1]/a[k-1] + omega[k]
        return b 
        
    def calc_psi(self, psi,a ,b):
        self.b=b
        self.a=a
        self.psi=psi
        psi_new= np.zeros(self.Nx, dtype=complex)
        for k in range(self.Nx-2,0,-1):
            psi_new[k]= (1/a[k]) *(psi_new[k+1] - b[k])
        return psi_new
    
    def step(self):
        
        if self.t >= self.T:
            return False
        
        a = self.calc_a()
        omega = self.calc_omega(self.psi)
        b = self.calc_b(omega, a)
        
        self.psi = self.calc_psi(self.psi, a, b)
        
        self.t += self.dt 
        
        return True
        
        
        
    def total_calc(self, k0):
        self.psi=self.initial_psi(k0)
        self.probability_list.append(np.abs(self.psi)**2)
        self.psi_list.append(self.psi)
        self.time_list.append(self.t)
        
        while self.t < self.T: #time limit
            if not self.step():
                break
            self.probability_list.append(np.abs(self.psi)**2)
            self.psi_list.append(self.psi)
            self.time_list.append(self.t)
            



    def plotting(self, k0):
        plt.figure(figsize=(12, 8))
    
        #time_indices = [0, 500, 1000, 1500] 
        #time_indices=[0]
        time_indices=[0,500]
        
        area_stay_final=None
        
        for time in time_indices:
            y=self.probability_list[time]
            plt.plot(self.x, self.probability_list[time],linewidth=2, label=f't = {self.time_list[time]:.2f} (n={time})')
            
            a=190
            b=210
            i_a=np.searchsorted(self.x,a)
            i_b=np.searchsorted(self.x,b)
            y_a= np.interp(a, self.x,y)
            y_b= np.interp(b, self.x,y)
            
            x_back= np.concatenate((self.x[:i_a], [a]))
            y_back= np.concatenate((y[:i_a], [y_a]))
            
            x_stay=np.concatenate((self.x[i_a:i_b], [b]))
            y_stay=np.concatenate((y[i_a:i_b], [y_b]))
            
            x_forward= np.concatenate(([b], self.x[i_b:]))
            y_forward=np.concatenate(([y_b], y[i_b:]))
            
            area_back=np.trapezoid(y_back, x_back)
            area_stay= np.trapezoid(y_stay, x_stay)
            area_forward=np.trapezoid(y_forward, x_forward)
            
            
            # GOOD ENOUGH ??
            
            area_tot= area_back+ area_stay+area_forward
            area_stay_final=area_stay
            #print(f'area in potential = {area_stay}')
            
            
            print('=========')
            print(f'analyzing k0={k0}')
            print(f'area before potential = {area_back}')
            print(f'area in potential = {area_stay}')
            print(f'area after potential = {area_forward}')
            print(f'total area =  {area_tot}')
            
            
        V_scaled = self.V / np.max(self.V) * np.max(self.probability_list[0]) 
        plt.fill_between(self.x, 0, V_scaled, alpha=0.2, 
                         color='gray', label='Potential')
        plt.xlabel('position x', fontsize=12)
        plt.ylabel('Probability ', fontsize=12)
        plt.ylim(-0.005,0.05)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title(f'probability vs x for different times, k0={k0}, V={V}', fontsize=14)

        plt.show()
        return area_stay_final
        
        
        
        
        
if __name__ == "__main__":
    
    k0_list = [0.4,0.16]
    V_list=[0.4,0.16]
    
    #k0_list = [np.sqrt(0.4),np.sqrt(0.16)]
    #V_list=[0.4,0.8]
    
    #k0_list = [5.0,8.0]
   # V_list=[1.0,1.0]

    plt.figure(figsize=(12,8))
    area_list=[]
    ratio_list=[]

    for k0,V in zip(k0_list, V_list):
        ratio=k0/V
        
        print(f'ratio= {k0/V}')
        solver = CrankNicolson(L=400.0, dt=0.05, Nx=200)
        solver.barrier(height=V)
        solver.total_calc(k0)
        area=solver.plotting(k0)
        area_list.append(area)
        ratio_list.append(ratio)






        