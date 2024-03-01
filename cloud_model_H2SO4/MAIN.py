#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 11:57:26 2022

@author: paytonbeeler
"""

import numpy as np
import indexing, run, tqdm, processing
import sys, time, os, pickle
from scipy.integrate import ode
# import matplotlib.pyplot as plt
from numba import types
from numba.typed import Dict

print('Starting main run ...')

# read inputs from run file
y0_filename = str(sys.argv[1])
V = float(sys.argv[2])
condens_coeff = float(sys.argv[3])
thermal_accom = float(sys.argv[4])
t_stop = float(sys.argv[5])
dt = float(sys.argv[6])
T = float(sys.argv[7])
output_directory = str(sys.argv[8])
output_filename = str(sys.argv[9])
O3_photolysis = float(sys.argv[10])

# read alphas from files
filename = 'alphas.txt'
raw_data = np.loadtxt(filename, dtype='str', delimiter = ':')
alphas = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
for i in range(0, len(raw_data)):
    alphas[str(raw_data[i, 0])] = float(raw_data[i, 1])

# read molecular masses from files
filename = 'molecular_masses.txt'
raw_data = np.loadtxt(filename, dtype='str', delimiter = ':')
molec_masses = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
for i in range(0, len(raw_data)): 
	molec_masses[str(raw_data[i, 0])] = float(raw_data[i, 1])

# read Henry's Law coefficients from files
filename = 'Henry_constants.txt'
raw_data = np.loadtxt(filename, dtype='str', delimiter = ' ')
H0 = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
for i in range(0, len(raw_data)):  
    H0[str(raw_data[i, 0])[:-1]] = float(raw_data[i, 1])*np.exp(float(raw_data[i, 2])*((1/T)-(1/298)))

# read hygroscopicity parameters from files
filename = 'kappas.txt'
raw_data = np.loadtxt(filename, dtype='str', delimiter = ':')
kappas = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
for i in range(0, len(raw_data)): 
	kappas[str(raw_data[i, 0])] = float(raw_data[i, 1])

# read densities from files
filename = 'densities.txt'
raw_data = np.loadtxt(filename, dtype='str', delimiter = ':')
densities = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
for i in range(0, len(raw_data)): 
	densities[str(raw_data[i, 0])] = float(raw_data[i, 1])


# read y0 from file
filename = y0_filename
print('Reading', y0_filename)
raw_data = np.loadtxt(filename, dtype='str', delimiter = ':')
y0 = np.zeros(0)
y0_org_array = []
bins = 0
for i in range(0, len(raw_data)):
    y0 = np.append(y0, float(raw_data[i, 1]))
    y0_org_array.append(raw_data[i, 0])
    if raw_data[i, 0][:7] == 'H2O (aq': 
        bins += 1
y0_org_array = np.array(y0_org_array)
print(bins, 'particles')

# G is a dict with the group number of each aqueous species, 
# I is a dict with the indices of the gas species/parcel properties,
# dx is the number of parcel/gas properties tracked in the model
Ig, Gaq, dx = indexing.getGroupNumbers(y0_org_array, bins)

# create output file if it doesn't exist
if os.path.exists(output_directory) == False:
    os.mkdir(output_directory)
       
start_time = time.time()
print(' ')
print('Compiling...')
dydt = run.ODEs(1.0, y0, bins, V, condens_coeff, thermal_accom, alphas, molec_masses, H0, kappas, densities, O3_photolysis, Ig, Gaq, dx)

# for i in range(0, dx):
#     print(indexing.getName(i, y0_org_array), y0[i], dydt[i])
# for k in Gaq.keys():
#     print(indexing.getName(dx+Gaq[k]*bins, y0_org_array), y0[dx+Gaq[k]*bins], dydt[dx+Gaq[k]*bins])

elapsed_time = time.time() - start_time
print('Total compiling time:', np.round(elapsed_time, 2), 'seconds')
print(' ')



# solve system of ODEs
t0 = 0.0
start_time = time.time()

ode15s = ode(run.ODEs).set_integrator('lsoda', method='bdf',
                                      rtol=1E-7, atol=1E-14, nsteps=5000)

ode15s.set_initial_value(y0, t0).set_f_params(bins, V, condens_coeff, 
                                              thermal_accom, alphas, 
                                              molec_masses, H0, kappas, 
                                              densities, O3_photolysis, 
                                              Ig, Gaq, dx)
soln = y0
t = np.array([t0])

print('Integrating...')
pbar = tqdm.tqdm(total = len(np.linspace(0., t_stop, int(t_stop/dt+1))))
while ode15s.successful() and ode15s.t < t_stop:
    soln = np.vstack([soln, ode15s.integrate(ode15s.t+dt)])
    t = np.append(t, ode15s.t)
    if abs(np.sum(soln[-1, :])) >= 0:
        success = True
    else:
        # for i in range(0, len(soln[-1, :])):
        #     print(indexing.getName(i, y0_org_array), soln[-2, i])
        success = False
    if success == False:
        sys.exit()
    pbar.update(1)
pbar.close()


elapsed_time = time.time() - start_time
print(' ')
print(' ')
print('Total solving time:', np.round(elapsed_time, 2), 'seconds')
print(' ')

processing.write_files(soln, output_directory + '/' + output_filename, V, t, y0_org_array, bins, densities, kappas, molec_masses)
data = pickle.load(open(output_directory + '/' + output_filename, 'rb'))
print('Return code:', ode15s.get_return_code())
print(' ')
