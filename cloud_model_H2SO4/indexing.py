#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 13:46:43 2022

@author: paytonbeeler
"""

import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict

#@njit(error_model='numpy')
def getName(index, y0_org_array):
    
    if index >= 0 and index < len(y0_org_array):
        name = y0_org_array[index]
    else:
        print('WARNING: '+str(index)+' is outside array limits')
        return
    
    return name


#@njit(error_model='numpy')
def getIndex(search, y0_org_array):
        
    for i in range(0, len(y0_org_array)):
        if y0_org_array[i] == search:
            return i
        
    print('WARNING: no array element is labeled '+search)
    return


#@njit(error_model='numpy')
def getGroupNumbers(y0_org_array, bins):

    i = 0 
    group_number = 0
    offset = 0
    group_dict = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    gas_dict = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
    while(i < len(y0_org_array)):
        idx = str(y0_org_array[i])
        idx2 = str(y0_org_array[i-1])
        if ',' in idx and idx[:-2] != idx2[:-2]:
            if '(aq,' in idx:
                j = idx.rfind('(')
                name = idx[:j-1]
                group_dict[name] = group_number
                group_number += 1
            else:
                j = idx.rfind(',')
                name = idx[:j]
                group_dict[name] = group_number
                group_number += 1
            i += bins
        else:
            name = idx
            gas_dict[name] = i
            i += 1
            offset += 1
            
    return gas_dict, group_dict, offset
    
    
