import numpy as np
import matplotlib.pyplot as plt
import pickle
from glob import glob
from pathlib import Path
import os

density_dict = {'SO4': 1800,
                'BC': 1700,
                'OC': 1000,
                'H2O': 1000,
                'POM': 1200}
                    
                    
def load_rollout_data(path):
    ''' Load pickle rollout files output by GNS.
    Args:
    path: path to the pickle files (default gns/output/), where each file corresponds to a rollout.
    
    Returns:
    dictionary: keys are string names of the rollout files.
    '''
    rollouts = {}
    for file in Path(path).glob("*.pkl"):
        rollouts[file.name] = pickle.load(open(file, "rb"))
    return rollouts

def volume(chem, mass):
    return (mass[chem] / density_dict[chem])

def gd_from_vol(vol):
    return np.cbrt(6 * vol / np.pi) 

def mass_concentration(mass_of_particles, aero_number, chem='all'):
    some_mass = next(iter(mass_of_particles.values()))
    vol_sum = np.zeros_like(some_mass)
    mass_sum = np.zeros_like(some_mass)
    for c in mass_of_particles:
        mass_sum += mass_of_particles[c] # mass of particles
        vol_sum += volume(c, mass_of_particles) # vol of particles
    vol_air = np.dot(vol_sum[:,:], np.transpose(aero_number[0,:]))
    
    if chem != 'all':
        return np.sum(mass_of_particles[chem], axis=1) / vol_air
    else:
        return np.sum(mass_sum, axis=1) / vol_air
    
def mean_std_diameter(mass_of_particles):
    some_mass = next(iter(mass_of_particles.values()))
    vol_sum = np.zeros_like(some_mass)
    for c in mass_of_particles:
        vol_sum += volume(c, mass_of_particles) # vol of particles
    diam = gd_from_vol(vol_sum)
    
    mean_d = np.exp(np.mean(np.log(diam), axis=0)) 
    std_d = np.exp(np.sqrt(np.mean(np.log(diam / mean_d)**2, axis=0)))
    return (mean_d, std_d)

def nmae(truth, pred):
    return np.mean(np.sum(np.abs((truth - pred)/truth), axis=0) / truth.shape[0]) 
    
