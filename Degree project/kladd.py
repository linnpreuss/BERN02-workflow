#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 14:17:37 2026

@author: linnpreussjelvez
"""


if __name__ == "__main__":
    solver = CrankNicolson(L=500.0, dt=0.1)
    solver.barrier()
    k0_list=[0,2,5,10]
    V_list=[0.1,0.5,1.0,2.0]
    k0_array=np.array(k0_list)
    V_array=np.array(V_list)
    ratio=k0_array/k0_array
    for k0 in k0_list:
        solver.total_calc(k0)
        max_p = max(np.max(prob) for prob in solver.probability_list)
        print(f'k0 = {k0}, max probability = {max_p:.4f}')
        
        
    solver.plotting()

#%%

import numpy as np
import matplotlib.pyplot as plt

class CrankNicolson2D_ADI:
    def __init__(self, Nx=200, Ny=200, Lx=30.0, Ly=30.0, dt=0.001, T=2000.0):
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dt = dt
        self.T = T
        
        self.x = np.linspace(0, Lx, Nx)
        self.y = np.linspace(0, Ly, Ny)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        self.probability_list = []
        self.time_list = []
        self.psi_list = []
        
        self.t = 0.0
        self.V = np.zeros((Nx, Ny))
        
        self.rx = 1j * self.dt / (2 * self.dx**2)
        self.ry = 1j * self.dt / (2 * self.dy**2)
    
    def initial_psi(self, x0, y0, sigma_x, sigma_y, kx, ky):
        psi = np.exp(-((self.X - x0)**2)/(2*sigma_x**2) - 
                     ((self.Y - y0)**2)/(2*sigma_y**2))
        psi *= np.exp(1j * (kx * self.X + ky * self.Y))
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx * self.dy)
        return psi / norm
    
    def barrier_2d(self, x_center, y_center, width_x, width_y, height):
        self.V = np.zeros((self.Nx, self.Ny))
        mask = ((np.abs(self.X - x_center) < width_x/2) & 
                (np.abs(self.Y - y_center) < width_y/2))
        self.V[mask] = height
    
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
        return omega
    
    def calc_omega_y(self, psi, i):
        omega = np.zeros(self.Ny, dtype=complex)
        
        for j in range(1, self.Ny-1):
            omega[j] = -psi[i,j-1] + 2*(1j*2* self.dy**2/self.dt +1 +self.dx**2 *self.V[i,j]) * psi[i,j] - psi[i,j+1]
       
        return omega
    
    def calc_b(self, omega, a):
        b = np.zeros(len(omega), dtype=complex)
        b[0] = omega[0]
        for k in range(1, len(omega)-1):
            b[k] = b[k-1] / a[k-1] + omega[k]
        return b
    
    def solve_tridiagonal(self, a, b):
        n = len(a)
        psi_new = np.zeros(n, dtype=complex)
        
        for k in range(n-2, 0, -1):
            psi_new[k] = (1 / a[k]) * (psi_new[k+1] - b[k])
        
        return psi_new
    
    def step(self):
        if self.t >= self.T:
            return False
        psi_half = np.zeros((self.Nx, self.Ny), dtype=complex)
        
        # x direction
        for j in range(1, self.Ny-1):
            a = self.calc_a_x(j)
            omega = self.calc_omega_x(self.psi, j)
            b = self.calc_b(omega, a)
            
            psi_half[:, j] = self.solve_tridiagonal(a, b)
        
        psi_half[:, 0] = self.psi[:, 0]
        psi_half[:, -1] = self.psi[:, -1]
        
        # y direction
        psi_new = np.zeros((self.Nx, self.Ny), dtype=complex)
        
        for i in range(1, self.Nx-1):
            a = self.calc_a_y(i)
            omega = self.calc_omega_y(psi_half, i)
            b = self.calc_b(omega, a)
            psi_new[i, :] = self.solve_tridiagonal(a, b)
        
        psi_new[0, :] = psi_half[0, :]
        psi_new[-1, :] = psi_half[-1, :]
        
        self.psi = psi_new
        self.t += self.dt
        return True
    
    def total_calc(self, x0, y0, sigma_x, sigma_y, kx, ky):
        self.psi = self.initial_psi(x0, y0, sigma_x, sigma_y, kx, ky)
        self.probability_list.append(np.abs(self.psi)**2)
        self.psi_list.append(self.psi.copy())
        self.time_list.append(self.t)
        
        while self.t < self.T:
            self.probability_list.append(np.abs(self.psi)**2)
            self.psi_list.append(self.psi.copy())
            self.time_list.append(self.t)
    
    def plot_snapshot(self, time_idx):
        plt.figure(figsize=(10, 8))
        
        prob = self.probability_list[time_idx]
        
        plt.imshow(prob.T, extent=[0, self.Lx, 0, self.Ly], 
                   origin='lower', cmap='hot', aspect='auto')
        plt.colorbar(label='Probability density')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f't = {self.time_list[time_idx]:.2f}')
        
        if np.max(self.V) > 0:
            barrier_mask = self.V > np.max(self.V)/2
            if np.any(barrier_mask):
                # Plot barrier outline
                from scipy import ndimage
                edges = ndimage.sobel(barrier_mask.astype(float))
                plt.contour(self.x, self.y, edges.T, levels=[0.5], colors='cyan', alpha=0.5)
        
        plt.show()


if __name__ == "__main__":
    solver = CrankNicolson2D_ADI(Nx=150, Ny=150, Lx=40.0, Ly=40.0, dt=0.05, T=20.0)
    
    solver.barrier_2d(x_center=20.0, y_center=20.0, 
                      width_x=5.0, width_y=5.0, height=2.0)
    solver.total_calc(x0=10.0, y0=10.0, 
                      sigma_x=3.0, sigma_y=3.0,
                      kx=2.0, ky=2.0)
    
    for idx in [0, len(solver.time_list)//4, len(solver.time_list)//2, -1]:
        solver.plot_snapshot(idx)