#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 00:20:52 2026

@author: linnpreussjelvez
"""

import numpy as np
import matplotlib.pyplot as plt

class Parameters:
    # Initialize with all parameters given.
    def __init__(self, dx=0.1, L=500.0, dt = 0.1, T=150.0, V0=0.7, a=250.0, b=260.0, x0=200, sigma=20, q=2.0):
        self.dx = dx
        self.L = L
        self.dt = dt
        self.T = T
        self.V0 = V0
        self.a, self.b = min(a, b), max(a, b)
        self.x0 = x0
        self.sigma = sigma
        self.q = q

        # Derived parameters
        self.Nx = int(L / dx) + 1
        self.Nt = (T / dt)

        # Check spatial resolution to avoid aliasing
        wavelength = 2 * np.pi / abs(q) if q != 0 else np.inf
        min_points_per_wave = wavelength / dx

        if min_points_per_wave < 10:
            raise ValueError(
                f"Wavelength (≈{wavelength:.2f}) resolved by only {min_points_per_wave:.1f} points. "
                f"Reduce dx to <= {wavelength/10:.3f} for stable propagation."
            )
    # Get a spacial grid
    def spaceGrid(self):
        return np.linspace(0, self.L, self.Nx)

    # Get the initial wave function defined on all points on 
    # the spatial grid.
    def psiZero(self):
        x = self.spaceGrid()
        return np.exp(1j*self.q*x)*np.exp(-(x - self.x0)**2./(2.*self.sigma**2.))

    # Get the potential barrier defined on all points on 
    # the spatial grid
    def potential(self):
        V = np.zeros(self.Nx)
        for i in range(self.Nx):
            if i*self.dx > self.a and i*self.dx < self.b:
                V[i] = self.V0
        return V

# SchrodingerSolver is the base class for solvers. It implements methods
# for solving the system of equations at each time step, and return
# relevant observables from methods.
# The base class assumes that the system is brought on the form
# psi^n+1 = M * psi^n to solve by matrix multiplication. The task
# of derived classes is to construct M.

class SchrodingerSolver:
    def __init__(self, parameters):
        if parameters is None:
            parameters = Parameters()
        self.p = parameters
        self.M = None  # To be constructed by derived class
        self.psi = self.p.psiZero()
        self.normalize()
        self.t = 0
        self.x = self.p.spaceGrid()

    # Solve the system of equations for a time step by matrix multiplication.
    # The method returns False once max simulation time is reached and True 
    # otherwise.
    def step(self):
        if self.M is None:
            raise NotImplementedError("Matrix M must be set in the derived class")
        if self.t >= self.p.T:
            return False
        self.t+=self.p.dt
        self.psi = np.dot(self.M, self.psi)
        return True

    # Return the squared wave function at current time step.
    def getPsi2(self):
        return np.abs(self.psi)**2

    # Return (a copy of) the wave function at current time step.
    def getPsi(self):
        return self.psi.copy()

    # Normalize the wave function
    def normalize(self):
        norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.p.dx)
        if norm != 0:
            self.psi /= norm

    # Compute transmitted and reflected probabilities
    def computeTransRef(self):
        psi2 = self.getPsi2()
        transmitted = np.sum(psi2[self.x > self.p.b]) * self.p.dx
        reflected   = np.sum(psi2[self.x < self.p.a]) * self.p.dx
        return transmitted, reflected


class CrankNicolsonSolver(SchrodingerSolver):
    def __init__(self, parameters=None, useSparse=False):
        super().__init__(parameters)
        self.useSparse = useSparse

        # Construct potential
        V = self.p.potential()

        # Define coefficients
        diagonalA = 1 + 1j * self.p.dt / self.p.dx**2 + 1j * self.p.dt * V
        diagonalB = 1 - 1j * self.p.dt / self.p.dx**2 - 1j * self.p.dt * V
        offDiagonalA = -0.5j * self.p.dt / self.p.dx**2
        offDiagonalB = 0.5j * self.p.dt / self.p.dx**2

        # Construct A and B
        from scipy.sparse import diags, identity
        from scipy.sparse.linalg import factorized
        import numpy as np

        A = diags([offDiagonalA, diagonalA, offDiagonalA], offsets=[-1, 0, 1], shape=(self.p.Nx, self.p.Nx), format='csr')
        B = diags([offDiagonalB, diagonalB, offDiagonalB], offsets=[-1, 0, 1], shape=(self.p.Nx, self.p.Nx), format='csr')
        self.solveA = factorized(A)  # defines a callable solver
        self.B = B
        A = diags([offDiagonalA, diagonalA, offDiagonalA], offsets=[-1, 0, 1], shape=(self.p.Nx, self.p.Nx)).toarray()
        B = diags([offDiagonalB, diagonalB, offDiagonalB], offsets=[-1, 0, 1], shape=(self.p.Nx, self.p.Nx)).toarray()
        Ainv = np.linalg.inv(A)
        self.M = np.dot(Ainv, B)

    def step(self):
        if self.t >= self.p.T:
            return False

        if self.useSparse:
            rhs = self.B @ self.psi
            self.psi = self.solveA(rhs)
        else:
            self.psi = self.M @ self.psi

        self.t += self.p.dt
        return True


import matplotlib.patches as patches

p = Parameters(T=133, q=3, x0=200, dt=0.1, a=500, b=510, L=800)

steps=[100,500,1000,1500]

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Create ONE solver
solver = CrankNicolsonSolver(p)

for idx, step_count in enumerate(steps):
    # Reset the solver to initial state
    solver.psi = p.psiZero()
    solver.normalize()
    solver.t = 0
    
    # Run the simulation
    for i in range(step_count):
        solver.step()
    
    # Get current axis
    ax = axes[idx]
    
    # Plot
    ax.plot(solver.x, solver.getPsi2(), lw=2, color='blue')
    
    # Add potential barrier
    barrier = patches.Rectangle((solver.p.a, 0), solver.p.b - solver.p.a, solver.p.V0/10,
                                linewidth=1, edgecolor='r', facecolor='none', alpha=0.5)
    ax.add_patch(barrier)
    
    ax.set_xlim(0, solver.p.L)
    ax.set_ylim(0, 0.1)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$|\Psi(x)|^2$')
    ax.set_title(f'{step_count} steps (t = {solver.t:.2f})')
    ax.grid(True, alpha=0.3)
    
    # Add info
    trans, refl = solver.computeTransRef()
    ax.text(0.02, 0.95, f'T: {trans:.4f}\nR: {refl:.4f}', 
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()