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
    
    rep_force = 12*x * epsilon**12/((rd**2)**7)
    attrat_force = -6*x * epsilon**6/((rd**2)**4)
    
    return pref * (rep_force + attrat_force)
    
def grav_force(m):
    ''' Gravitational force
    
    m = particle mass
    '''
    g =  9.80665
    
    return -m*g*0

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
    
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    xr = rt[:,0]
    yr = rt[:,1]
    ax.scatter(xr, yr, label = 'Time {:}'.format(t))
    ax.legend(loc='upper left')
    plt.xlim(x1, x2)
    plt.ylim(y1, 4*y2)

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
    cutoff_radius = 10*epsilon
    fix = 0
    fiy = 0
    
    print('At time: {}:'.format(t))
    print('Calc force on Particle', i)
    print('Position particle {}: \n ({}, {})'
          .format(i, r[i, 0, t], r[i, 1, t]))
    for j in range(len(r[:, 0, 0])):
        
        if i != j:
            print('by Particle', j)
            print('Position particle {}: \n ({}, {})'
                  .format(j, r[j, 0, t], r[j, 1, t]))
            rd = distance_2points(r[j,:,t], r[i,:,t])
            print(j, i, 'Distance: ', rd)
            print('Cutoff: ', cutoff_radius)
            if rd < cutoff_radius:
                x = r[i,0,t] - r[j,0,t]
                y = r[i,1,t] - r[j,1,t]
                fix += lennard_jones_force(sigma, epsilon, x, rd)
                fiy += lennard_jones_force(sigma, epsilon, y, rd)
            else:
                fix += 0
                fiy += 0
            
            
    fiy = fiy + grav_force(M)
    print('Force on x: ', fix)
    print('Force on y: ', fiy)
    print('---------------------------')
    return fix, fiy    
    
def system_time_evol(r, v, time, sigma, epsilon, M, L):
    ''' Mechanic time evolution of the system.
    Calls velocity verlet algorithm
    
    r = positions for every particle at every time
    v = velocities for every particle at every time
    
    sigma, epsilon = LJ parameters
    M = particles mass
    L = box lenght
    '''
    fi = np.zeros(r.shape)
    dt = time[1]-time[0]
    # Initialize Forces at time 0
    print('---Force initilization---')
    for i in range(len(r[:,0,0])):
        print('For Particle {}'.format(i))
        fi[i, 0, 0], fi[i, 1, 0] = force_i(r, i, sigma, epsilon, M, 0)
    print('---END Force initilization---')    
    # Time evolution
    for t in range(1, len(time)):
        # Calculate positions
        for i in range(len(r[:,0,0])):
            refl_ix, refl_iy = False, False
            r[i, 0, t] = (r[i,0,t-1] + v[i,0,t-1]*dt +
                            0.5 * fi[i, 0, t-1] * dt**2)
            r[i, 1, t] = (r[i,1,t-1] + v[i,1,t-1]*dt +
                            0.5 * fi[i, 1, t-1] * dt**2)
            if (r[i, 0, t] < 0): 
                refl_ix = True
                r[i, 0, t] = reflect_r(r[i,0,t], 0) 
            elif (r[i,0,t] > L):
                refl_ix = True
                r[i, 0, t] = reflect_r(r[i,0,t], L) 
            if (r[i, 1, t] < 0):
                refl_iy = True
                r[i, 1, t] = reflect_r(r[i,1,t], 0) 
                
        # Update forces
        for i in range(len(r[:,0,0])):
            fi[i, 0, t], fi[i, 1, t] = force_i(r, i, sigma, epsilon, M, t)
        
        # Calculate velocities
        for i  in range(len(r[:,0,0])):
            v[i, 0, t] = (v[i,0,t-1] + 
                            0.5*(fi[i, 0, t-1] + fi[i, 0, t])*dt)
            v[i, 1, t] = (v[i,1,t-1] + 
                            0.5*(fi[i, 1, t-1] + fi[i, 1, t])*dt)               
            if refl_ix:
                v[i, 0, t] = - v[i, 0, t]
            if refl_iy:
                v[i, 1, t] = - v[i, 1, t]
            
    return r, v, fi  
    
    
    
if __name__ == '__main__':
    
    # Simulation box definition
    
    L = 30
    dim_box = 2
    
    N = 2

    time = np.linspace(1, 10, 10)
    dt = time[1]-time[0]
    
    r = np.zeros((N, dim_box, len(time))) # Particle, dimension, time
    v = np.zeros((N, dim_box, len(time)))
    # L-J potential for Xenon
    
    '''
    sigma = 4.10*Angstrom2Bohr # Bohg
    epsilon = 1.77*kJmol2Hartree # Hartree
    M = 131*uma2me # electron masses
    part_dist = sigma
    '''
    # test values
    sigma, epsilon, M = 1, 1, 1
    part_dist = 2*epsilon*2**(1/6)
    
    # Create intial condition
    #part_dist = 30 # Angstrom 
    r0 = np.zeros((N, dim_box))  
    r0 = regular_grid(r0, part_dist, 10)
    v0 = 0*np.random.rand(N, dim_box)
    
    
    for i in range(len(r[:,0,0])):
        r[i,0,0] = r0[i,0]
        r[i,1,0] = r0[i,1]
        v[i,0,0] = v0[i,0]
        v[i,1,0] = v0[i,1]
        
    plot_position(r[:,:,0], 0, L, 0, L, 0)
    T = np.zeros(len(time))
    T[0] = calculate_T(N, dim_box, M, v[:, :, 0])
    #fix, fiy = force_i(r, 0, sigma, epsilon, M, 0)
    r, v, forces = system_time_evol(r, v, time, sigma, epsilon, M, L)
    
    plot_position(r[:,:,-1], 0, L, 0, L, 0)
        
        
   