# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 20:36:08 2016

@author: Necot

Molecular dynamics code Ex2
"""
import numpy as np
import matplotlib.pyplot as plt

# Define physical constants and conversion factors

me = 1 # mass electron
kb = 1 # Boltzman constant
Angstrom2Bohr = 1.8897
kJmol2Hartree = 0.00038
uma2me = 1822.888

    
def reflect_r(r, L):
    ''' Reflecting condition coordinate r
    '''
    
    r = L - (r - L)
    
    return r

def reflect_v(v):
    ''' Reflecting condition velocity coordinate v
    '''
    
    return -v

def distance_2points(x, y):
    ''' Calculate distance between two points: x and y
    
    x, y = tuples with cartesian coords. points x,y
    '''
    sum = 0
    if len(x) != len(y):
        raise IndexError('The two points must have the same dimension.')
    for i in range(len(x)):
        sum += (x[i] - y[i])**2
        
    return np.sqrt(sum)
        
def lennard_jones(sigma, epsilon, r):
    ''' Lennard-Jones potential for particles at distance r
    
    r = distance between the 2 particles
    sigma, epsilon = L-J potential parameter
    '''

    pref = 4 * sigma
    repulsion_term = (epsilon/r)**12
    attration_term = -(epsilon/r)**6

    return pref * (repulsion_term + attration_term)
 

def lennard_jones_force(sigma, epsilon, x, rd):
    ''' Define L-J force along x
    
    x = scalar (instant position along a cartesian coord.)
    rd = distances between 2 particles
    
    '''
    pref = 4*sigma
    
    rep_force = 12*x * epsilon**12/((rd**12)**7)
    attrat_force = -6*x * epsilon**6/((rd**12)**4)
    
    return pref * (rep_force + attrat_force)
    
def grav_force(m):
    ''' Gravitational force
    
    m = particle mass
    '''
    g =  9.80665
    
    return -m*g

def regular_grid(r0, space, L0):
    ''' Create a regular grid of len(r0) points with distance 'space' in 
    dim r0 dimensions, starting from L0
    
    r0 = ndarray with cartesian coordinates points
    '''
    dimension = len(r0[0])
    num = int(np.power(N, 1/dimension))

    
    r0[0] = L0
    
    for i in range(1, len(r0)):
        r0[i][0] = r0[0][0] + (i%num) * space
        r0[i][1] = r0[0][0] + (i//num) * space
            
    return r0
    
    
def plot_position(rt, x1, x2, y1,y2, t):
    ''' Plot positions at time t
    
    rt = vector containing the coordinates of the N particles at t
    (x1, x2) = start and end box along x
    (y1,y2) = start and end box along y
    '''
    
    plt.figure(figsize=(16, 8))
    ax = plt.gca()
    xr = rt[:,0]
    yr = rt[:,1]
    ax.scatter(xr, yr, label = 'Time {:}'.format(t))
    ax.legend(loc='upper left')
    plt.xlim(x1, x2)
    plt.ylim(y1, y2)

def calculate_T(N, dim, M, v):
    ''' Calculate temperature thorugh the system kinetic energy

    kbT = 1/(dim*(N-1))Sum_i Mi * vi^2
    '''
    
    pref = 1/(dim*(N-1))
    sum = 0
    for i  in range(len(v)):
        sum += M * (v[i][0]**2 + v[i][1]**2)
    
    return pref*sum


def force_i(r, i, sigma, epsilon, M, t):
    ''' Forces on particle i at time t
    
    f_i,x = Sum_j!=i L-J force
    f_i,y = Sum_j!=i L-J force + gravity
    
    r = list particles coords. at every time
    
    M = mass particle
    t = time to evaluate the force (index of a time list)
    x = x-coord position vector
    y = y-coord position vector
    '''
    
    fix = 0
    fiy = 0
    
    for j in range(len(r[:, 0, 0])):
        if i != j:
            rd = distance_2points(r[j,:,t], r[i,:,t])
            x = r[i,0,t] - r[j,0,t]
            y = r[i,1,t] - r[j,1,t]
            fix += lennard_jones_force(sigma, epsilon, x, rd)
            fiy += lennard_jones_force(sigma, epsilon, y, rd)
    
    fiy = fiy + grav_force(M)

    return fix, fiy    
    
def system_time_evol(r, v, time, sigma, epsilon, M):
    ''' Mechanic time evolution of the system.
    Calls velocity verlet algorithm
    '''
    fi = np.zeros(r.shape)
    dt = time[1]-time[0]
    for i in range(len(r[:,0,0])):
        fi[i, 0, 0], fi[i, 1, 0] = force_i(r, i, sigma, epsilon, M, 0)
    for t in range(1, len(time)):
        for i in range(len(r[:,0,0])):
           
            r[i, 0, t] = (r[i,0,t-1] + v[i,0,t-1]*dt +
                            0.5 * fi[i, 0, t-1] * dt**2)
            r[i, 1, t] = (r[i,1,t-1] + v[i,1,t-1]*dt +
                            0.5 * fi[i, 1, t-1] * dt**2)
            fi[i, 0, t], fi[i, 1, t] = force_i(r, i, sigma, epsilon, M, t)
            v[i, 0, t] = (v[i,0,t-1] + 
                            0.5*(fi[i, 0, t-1] + fi[i, 0, t])*dt)
            v[i, 1, t] = (v[i,1,t-1] + 
                            0.5*(fi[i, 1, t-1] + fi[i, 1, t])*dt)               
            
    return r, v, fi  
    
    
    
if __name__ == '__main__':
    
    # Simulation box definition
    
    L = 100*Angstrom2Bohr # Bohr
    dim_box = 2
    
    N = 4

    time = np.linspace(1, 1, 10)
    dt = time[1]-time[0]
    
    r = np.zeros((N, dim_box, len(time))) # Particle, dimension, time
    v = np.zeros((N, dim_box, len(time)))
    # L-J potential for Xenon
    
    '''
    sigma = 4.10*Angstrom2Bohr # Bohg
    epsilon = 1.77*kJmol2Hartree # Hartree
    M = 131*uma2me # electron masses
    '''
    # test values
    sigma, epsilon, M = 1, 1, 1
    part_dist = epsilon*2**(1/6)
    
    # Create intial condition
    part_dist = 30 # Angstrom 
    r0 = np.zeros((N, dim_box))  
    r0 = regular_grid(r0, sigma, part_dist)#*Angstrom2Bohr)
    v0 = np.random.rand(N, dim_box)#*10e-6
    
    
    for i in range(len(r[:,0,0])):
        r[i,0,0] = r0[i,0]
        r[i,1,0] = r0[i,1]
        v[i,0,0] = v0[i,0]
        v[i,1,0] = v0[i,1]
        
    plot_position(r[:,:,0], 0, L, 0, L, 0)
    T = np.zeros(len(time))
    T[0] = calculate_T(N, dim_box, M, v[:, :, 0])
    fix, fiy = force_i(r, 0, sigma, epsilon, M, 0)
    r, v, forces = system_time_evol(r,v, time, sigma, epsilon, M)
    
    plot_position(r[:,:,4], 0, L, 0, L, 0)
        
        
   