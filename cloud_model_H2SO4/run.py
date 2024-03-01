# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:09:00 2022

@author: beel083
"""

import numpy as np
import parcel
import aq_chemistry, gas_chemistry

Na = 6.022e23 #molecules/mol
R = 8.314 # m^3*Pa/mol*K

def ODEs(t, y, bins, V, condens_coeff, thermal_accom, alphas, molec_masses, H_eff, kappas, densities, O3_photolysis, Ig, Gaq, dx):
        
    dydt = np.zeros(len(y))
    dydt = parcel.parcel_properties(t, y, dydt, V, bins, thermal_accom, condens_coeff, densities, kappas, molec_masses, Ig, Gaq, dx)
    dydt, gas2part_mass_transfer = aq_chemistry.particle_phase(y, dydt, bins, alphas, densities, molec_masses, H_eff, Ig, Gaq, dx)
    
    # gas2part_mass_transfer = np.zeros(len(y))
    dydt = gas_chemistry.gas_chem(y, dydt, bins, densities, Ig, Gaq, dx, molec_masses, O3_photolysis, gas2part_mass_transfer)
        
    return dydt


    
    
    
    
