#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 14:22:42 2026

@author: linnpreussjelvez
"""
import numpy as np
import matplotlib.pyplot as plt


            
class CrankNicolson:
    def __init__(self, Nx=500, Ny=500, Lx=30.0, Ly=30, dt=0.001, T=2000.0, kx=5.0,ky=5.0):
        self.Nx=Nx
        self.Ny=Ny
        self.Lx=Lx
        self.Ly=Ly
        self.dt=dt
        self.T=T
        
        self.x = np.linspace(-Lx/2, Lx/2, self.Nx)
        self.dx = self.x[1]-self.x[0]
        self.y = np.linspace(-Ly/2, Ly/2, self.Ny)
        self.dy = self.y[1]-self.y[0]
        
        
        self.probability_list =[]
        self.time_list = []
        self.psi_list=[]
        
        self.t=0.0
        
        self.V = np.zeros((Nx, Ny))
        
        self.kx=kx
        self.ky=ky
        
        #grid 
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
    
    
    def initial_psi(self, kx,ky):
        x0=2.5
        y0=2.5
        sigma= 10.0
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        psi=np.exp(((-(X-x0)**2)-((Y-y0)**2))/(2*sigma**2) ) * np.exp( 1j*(kx*X + ky*Y))
        norm= (np.pi*sigma**2)**(1/4) 
        return psi /norm
         
        
    def barrier(self, center_x=0.0, center_y=0.0, width=5.0, height=5.0):
        self.width = width
        self.center_x = center_x
        self.center_y = center_y
        self.height = height
        self.V = np.zeros((self.Nx, self.Ny))
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
    
        self.V=height* np.exp(( -(X-center_x)**2-(Y-center_y)**2) / (2*width**2) )
                
                
    def calc_a_x(self, j):
        a = np.zeros(self.Nx, dtype=complex)
        a[0] = 2 * (1 + self.dx**2 * self.V[0, j] - (1j * 2 * self.dx**2 / self.dt))
        for i in range(1, self.Nx-1):
            a[i] = 2 * (1 + self.dx**2 * self.V[i, j] - 1j * 2 * self.dx**2 / self.dt) - (1 / a[i-1])
        
        return a
    
    def calc_a_y(self, i):
        a = np.zeros(self.Ny, dtype=complex)
        a[0] = 2 * (1 + self.dy**2 * self.V[i, 0] - (1j * 2 * self.dy**2 / self.dt))
        for j in range(1, self.Ny-1):
            a[j] = 2 * (1 + self.dy**2 * self.V[i, j] - 1j * 2 * self.dy**2 / self.dt) - (1 / a[j-1])
        
        return a
    
    def calc_omega_x(self, psi, j):
        omega = np.zeros(self.Nx, dtype=complex)
        
        for i in range(1, self.Nx-1):
            omega[i] = -psi[i-1,j] + 2*(1j*2* self.dx**2/self.dt +1 +self.dx**2 *self.V[i,j]) * psi[i,j] - psi[i+1,j]
            
        omega[0] = 0
        omega[-1] = 0
        return omega
    
    def calc_omega_y(self, psi, i):
        omega = np.zeros(self.Ny, dtype=complex)
        
        for j in range(1, self.Ny-1):
            omega[j] = -psi[i,j-1] + 2*(1j*2* self.dy**2/self.dt +1 +self.dy**2 *self.V[i,j]) * psi[i,j] - psi[i,j+1]
       
        omega[0] = 0
        omega[-1] = 0
        return omega
    
    def calc_b(self, omega, a):          
        b = np.zeros(len(omega), dtype=complex)
        b[0] = omega[0]
        for k in range(1, len(omega)-1):
            b[k] = b[k-1] / a[k-1] + omega[k]
        return b
        
    
    def calc_psi(self,psi, a,b):
        N= len(a)
        psi_new= np.zeros(N, dtype=complex)
        for k in range (N-2,0,-1):
            psi_new[k] = (1/a[k]) *(psi_new[k+1] - b[k])
        return psi_new
    
   
        
    def step(self):
        if self.t >= self.T:
            return False
        
        psi_half= np.zeros_like(self.psi, dtype=complex)
        for j in range(1, self.Ny-1):
            row=self.psi[:,j]
            a= self.calc_a_x(j)
            omega= self.calc_omega_x(self.psi,j)
            b= self.calc_b(omega,a)
            psi_half[:,j] = self.calc_psi(row,a,b)
        psi_half[:,0]= self.psi[:,0]
        psi_half[:,-1]=self.psi[:,-1]
            
        psi_new= np.zeros_like(self.psi, dtype=complex)
        for i in range(1, self.Nx-1):
            column=psi_half[i,:]
            a = self.calc_a_y(i)
            omega= self.calc_omega_y(psi_half, i)
            b= self.calc_b(omega,a)
            psi_new[i,:]= self.calc_psi(column,a,b)
        psi_new[0,:]  = psi_half[0,:]
        psi_new[-1,:] = psi_half[-1,:]
        
        self.psi=psi_new
        self.t += self.dt
        return True
        
        
    def total_calc(self, kx,ky):
        self.psi=self.initial_psi(kx,ky)
        self.probability_list.append(np.abs(self.psi)**2)
        self.psi_list.append(self.psi)
        self.time_list.append(self.t)
        
        while self.t < self.T: 
            if not self.step():
                break
            self.probability_list.append(np.abs(self.psi)**2)
            self.psi_list.append(self.psi)
            self.time_list.append(self.t)
            
    
    def plotting(self, time_idx):
            plt.figure(figsize=(8,6))
            prob = self.probability_list[time_idx]
            plt.imshow(prob.T, extent=[-self.Lx/2,self.Lx/2,-self.Ly/2,self.Ly/2], origin='lower', cmap='hot', aspect='auto')
            plt.colorbar(label='Probability density')
            
            X, Y = np.meshgrid(self.x, self.y, indexing='ij')
                
            V_scaled= self.V.T/ np.max(self.V)
            
            overlay = np.zeros((*V_scaled.shape, 4))
            overlay[..., 2] = 1.0 
            overlay[..., 3] = V_scaled   #[R,G,B,opacity]
            
            plt.imshow(overlay, extent=[-self.Lx/2, self.Lx/2, -self.Ly/2, self.Ly/2], 
                       origin='lower', aspect='auto')
            
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f't = {self.time_list[time_idx]:.2f}')
            plt.show()


if __name__ == "__main__":
    solver= CrankNicolson(Nx=150, Ny=150, Lx=100, Ly=100,dt=0.1, T=50.0)
    solver.barrier(height=2.0, center_x=0, center_y=0)
    solver.total_calc(kx=0.5, ky=0.5)
    for idx in [0, 50, 100, 500]:
        solver.plotting(idx)
    
    
    
    

        