#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 16:19:50 2023

@author: beel083
"""
import numpy as np
import parcel
from numba import njit
import warnings

warnings.filterwarnings("ignore")

R = 8.314 # m^3*Pa/mol*K
R_dry = 287.0 # Pa*m^3/kg*K
Mw = 18.015/1000.0 # molar mass of water, kg/mol
Ma = 28.9/1000.0 # molar mass of dry air, kg/mol
rho_w = 1000.0 # density of water, kg/m^3
kb = 1.380649e-23 #m^2 kg/s^2 K
Na = 6.022e23 #molecules/mol

@njit(error_model='numpy')
def gas_chem(y, dydt, Np, densities, Ig, Gaq, dx, molec_masses, O3_photolysis, gas2part):

    # gather the parcel/aerosol properties
    T = y[Ig['T']]
    P = y[Ig['P']]
    # dry_radii, wet_radii = parcel.calculate_radii(y, densities, Gaq, dx, Np)
    
    # constant for switching between molec/cc and mole fraction
    A = (R*T*100**3)/(P*Na)

    #H2SO4
    k = 4.8e-13                   # Lim: SO2 + OH --> H2SO4 + HO2
    reactions = k*y[Ig['SO2']]*y[Ig['OH']]
    particles = gas2part[Ig['H2SO4']]
    Xi = A*y[Ig['H2SO4']]
    dilution = (Na/(R*100**3))*((1/T**2)*(T*Xi*dydt[Ig['P']] - Xi*P*dydt[Ig['T']])) # molec/cm^3*s 
    dydt[Ig['H2SO4']] = reactions - particles + dilution
    
    #OH
    k1 = 4.8e-13                   # Lim 2005: SO2 + OH --> H2SO4 + HO2
    H2O = ((y[Ig['SS']]+1)*100)/100*np.power(10, 8.10765-1750.286/(T-273.15+235))/760*2.46e19*(298.15/T) # molec/cm^3
    k2 = 2.2e-10*H2O               # Lim 2005: O1D + H2O --> 2OH
    reactions = -k1*y[Ig['SO2']]*y[Ig['OH']] + 2*k2*y[Ig['O1D']]
    particles = gas2part[Ig['OH']]
    Xi = A*y[Ig['OH']]
    dilution = (Na/(R*100**3))*((1/T**2)*(T*Xi*dydt[Ig['P']] - Xi*P*dydt[Ig['T']])) # molec/cm^3*s 
    dydt[Ig['OH']] = reactions - particles + dilution
    
    #SO2
    k = 4.8e-13                   # Lim: SO2 + OH --> H2SO4 + HO2
    reactions = -k*y[Ig['SO2']]*y[Ig['OH']]
    particles = gas2part[Ig['SO2']]
    Xi = A*y[Ig['SO2']]
    dilution = (Na/(R*100**3))*((1/T**2)*(T*Xi*dydt[Ig['P']] - Xi*P*dydt[Ig['T']])) # molec/cm^3*s 
    dydt[Ig['SO2']] = reactions - particles + dilution
    
    #O3
    reactions = -O3_photolysis*y[Ig['O3']]
    particles = gas2part[Ig['O3']]
    Xi = A*y[Ig['O3']]
    dilution = (Na/(R*100**3))*((1/T**2)*(T*Xi*dydt[Ig['P']] - Xi*P*dydt[Ig['T']])) # molec/cm^3*s 
    dydt[Ig['O3']] = reactions - particles + dilution
    
    #O1D
    k2 = 2.2e-10*H2O               # Lim 2005: O1D + H2O --> 2OH
    reactions = O3_photolysis*y[Ig['O3']] - k2*y[Ig['O1D']]
    particles = 0
    Xi = A*y[Ig['O1D']]
    dilution = (Na/(R*100**3))*((1/T**2)*(T*Xi*dydt[Ig['P']] - Xi*P*dydt[Ig['T']])) # molec/cm^3*s 
    dydt[Ig['O1D']] = reactions - particles + dilution
    
    return dydt

