import numpy as np
import matplotlib.pyplot as plt
import pickle
from glob import glob
from pathlib import Path
import os

density_dict = {'SO4': 1770,
                'BC': 1700,
                'OC': 1000,
                'H2O': 1000}
                    
                    
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


def nmae(truth, pred):
    return np.mean(np.sum(np.abs(truth - pred)/truth, axis=0) / truth.shape[0])

def gd_from_mass(chem, mass):
    return np.cbrt(6 * mass / (np.pi * density_dict[chem]))

def volume(diameter):
    return (np.pi / 6) * diameter ** 3

def mass_concentration(particles, chem_list, chem='all'):
    vol_sum = np.zeros_like(particles[:,:,0])
    mass_sum = np.zeros_like(particles[:,:,0])
    for i, c in enumerate(chem_list):
        mass_sum += particles[:,:,i]
        vol_sum += volume(gd_from_mass(c, particles[:,:,i]))
    vol_sum = np.sum(vol_sum, axis=0)
    
    if chem != 'all':
        return np.sum(particles[:,:,chem_list.index(chem)], axis=0) / vol_sum
    else:
        return np.sum(mass_sum, axis=0) / vol_sum
    
def mean_std_diameter(particles, chem_list):
    diam_sum = np.zeros_like(particles[:,:,0])
    for i, c in enumerate(chem_list):
        diam_sum += gd_from_mass(c, particles[:,:,i])
    
    mean_d = np.exp(np.mean(np.log(diam_sum), axis=0)) 
    std_d = np.exp(np.sqrt(np.mean(np.log(diam_sum / mean_d)**2, axis=0)))
    return (mean_d, std_d)
    
