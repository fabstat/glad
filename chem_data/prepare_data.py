from glob import glob
import re
import numpy as np
from pathlib import Path
from random import shuffle
import json
import os

def load_raw_data(path):
    ''' Load txt files output by partMC.
    Args:
    path: path to the text files, where each file has the masses of particles over time
    corresponding to one chemical.
    
    Returns:
    dictionary: keys are string names of chemicals and values are np.arrays of shape
    (number of time steps, number of particles)
    '''
    feats = {}
    for file in Path(path).glob("*.txt"):
        l = re.split("_", file.name)[:-1]
        if len(l) > 1:
            feats["_".join(l)] = np.loadtxt(file)
        else:
            feats[l[0]] = np.loadtxt(file)
    return feats

def normalize(X):
    if len(X.shape) > 1:
        x_min = X.min(axis=1).min(axis=0)
        x_max = X.max(axis=1).max(axis=0)
    else:
        x_min = X.min(axis=0)
        x_max = X.max(axis=0)
    return (X - x_min) / (x_max - x_min), x_min, x_max


def data_splits(ts_chems, ptypes, unumbers, mat_props, traincut=0.6, testcut=1.0):
    splits = {}
    idxs = list(range(ts_chems.shape[1]))
    shuffle(idxs)
    
    train_cutoff = int(ts_chems.shape[1]*traincut)
    test_cutoff = int(ts_chems.shape[1]*testcut)
    
    train_X = ts_chems[:,idxs[:train_cutoff],:] 
    test_X = ts_chems[:,idxs[train_cutoff:test_cutoff],:]
    val_X = ts_chems[:,idxs[test_cutoff:],:]
    
    train_ptype = ptypes[idxs[:train_cutoff]]
    test_ptype = ptypes[idxs[train_cutoff:test_cutoff]]
    val_ptype = ptypes[idxs[test_cutoff:]]
    
    train_unumber = unumbers[idxs[:train_cutoff]]
    test_unumber = unumbers[idxs[train_cutoff:test_cutoff]]
    val_unumber = unumbers[idxs[test_cutoff:]]
    
    if len(mat_props.shape) > 1:
        train_MP = mat_props[idxs[:train_cutoff],:]
        test_MP = mat_props[idxs[train_cutoff:test_cutoff],:]
        val_MP = mat_props[idxs[test_cutoff:],:]
    else:
        train_MP = mat_props[idxs[:train_cutoff]]
        test_MP = mat_props[idxs[train_cutoff:test_cutoff]]
        val_MP = mat_props[idxs[test_cutoff:]]
        
    splits["train_data"] = [train_X, train_ptype, train_unumber, train_MP]
    splits["test_data"] = [test_X, test_ptype, test_unumber, test_MP]
    
    
    if testcut < 1.0:
        splits["val_data"] = [val_X, val_ptype, val_unumber, val_MP]
        
    return splits, idxs, train_cutoff, test_cutoff  

def make_metadata_file(path, training_data):
    train_X = training_data[0]
    train_ptype = training_data[1]
    train_unumber = training_data[2]
    train_MP = training_data[3]
    
    train_vel = train_X[1:,:,:] - train_X[:-1,:,:]
    train_acc = train_vel[1:,:,:] - train_vel[:-1,:,:]
    
    vel_mean = list(np.max(np.mean(train_vel, axis=0), axis=0))
    vel_std = list(np.max(np.std(train_vel, axis=0), axis=0))
    acc_mean = list(np.max(np.mean(train_acc, axis=0), axis=0))
    acc_std = list(np.max(np.std(train_acc, axis=0), axis=0))

    # in case of zeros
    vel_std = list(np.where(np.abs(vel_std) <= 1e-22, 1e-22, vel_std))
    acc_std = list(np.where(np.abs(acc_std) <= 1e-22, 1e-22, acc_std))

    stack_min = np.min(np.min(train_X, axis=0), axis=0).round() 
    stack_max = np.max(np.max(train_X, axis=0), axis=0).round() 
    
    bounds = [list(t) for t in zip(stack_min, stack_max)]
    
    # Data to be written
    dictionary = {
        "bounds": bounds,
        "sequence_length": train_X.shape[0],
        "dim": train_X.shape[-1],
        "num_prop": train_MP.shape[-1],
        "dt": 1,
        "vel_mean": vel_mean,
        "vel_std": vel_std,
        "acc_mean": acc_mean,
        "acc_std": acc_std
    }
    # Serializing json
    json_object = json.dumps(dictionary, indent=4)
 
    # Writing to sample.json
    with open(os.path.join(path, "metadata.json"), "w") as outfile:
        outfile.write(json_object)
    
