#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 22:13:41 2023

@author: beel083
"""

import numpy as np
import parcel
from numba import njit

R = 8.314 # m^3*Pa/mol*K
R_dry = 287.0 # Pa*m^3/kg*K
Mw = 18.015/1000.0 # molar mass of water, kg/mol
Ma = 28.9/1000.0 # molar mass of dry air, kg/mol
rho_w = 1000.0 # density of water, kg/m^3
kb = 1.380649e-23 #m^2 kg/s^2 K
Na = 6.022e23 #molecules/mol

@njit(error_model='numpy')
def particle_phase(y, dydt, Np, alphas, densities, molec_masses, H_eff, Ig, Gaq, dx):
    
    T = y[Ig['T']]
    dry_radii, wet_radii = parcel.calculate_radii(y, densities, Gaq, dx, Np)
    Ns = y[dx+Gaq['N']*Np:dx+Gaq['N']*Np+Np]
    pH = y[dx+Gaq['pH']*Np:dx+Gaq['pH']*Np+Np]
    Hplus = np.power(10, -pH) # M
    dHplus = np.zeros(len(Hplus))
    gas2part = np.zeros(dx)

    # find the concentration of each aqueous species in mol/m^3
    water_volumes = (1e-18*y[dx+Gaq['H2O']*Np:dx+Gaq['H2O']*Np+Np])/densities['H2O'] # m^3
    conc = np.zeros(Np*len(Gaq))
    for x in Gaq.keys():
        if x != 'N' and x != 'pH' and x != 'H2O':
            moles_x = (1e-18*y[dx+Gaq[x]*Np:dx+Gaq[x]*Np+Np])/molec_masses[x] # mol
            conc[Gaq[x]*Np:Gaq[x]*Np+Np] = moles_x/water_volumes # mol/m^3

    # total S(VI)
    name = 'H2SO4'
    Dg = (1/100**2)*1.9*np.power(molec_masses[name], (-2/3)) # est. gas-phase diffusion coefficient, m2/sec  (Schnoor 1996 via Lim)
    w = np.sqrt((8*R*T)/(np.pi*molec_masses[name])) # thermal velocity, m/s
    K1 = 1000
    K2 = 1.02E-2*np.exp(2720*((1/T)-(1/298))) #M
    kmt = np.power((np.power(wet_radii, 2)/(3*Dg))+((4.0*wet_radii)/(3.0*w*alphas[name])), -1.0) # mass uptake, 1/s
    H = H_eff[name]*(1000/101325)*(1+(K1/Hplus)+((K1*K2)/(Hplus*Hplus))) # mol/m^3*Pa
    dydt_gas = molec_masses[name]*water_volumes*kmt*(((y[Ig[name]]*100**3)/Na) - ((conc[Gaq['H2SO4']*Np:Gaq['H2SO4']*Np+Np]+conc[Gaq['BISULFATE']*Np:Gaq['BISULFATE']*Np+Np]+conc[Gaq['SULFATE']*Np:Gaq['SULFATE']*Np+Np])/(H*R*T))) # kg/s  
    gas2part[Ig[name]] = np.sum(dydt_gas*((Na*Ns)/(molec_masses[name]*100**3)))
    dS6_T = (dydt_gas)/(1000*water_volumes*molec_masses[name]) # mol/L*s
    
    # H2SO4
    name = 'H2SO4'
    dydt_eq = dS6_T/(1+(K1/Hplus)+((K1*K2)/(Hplus*Hplus))) # mol/L*s
    dH2SO4 = dydt_eq
    dydt[dx+Gaq[name]*Np:dx+Gaq[name]*Np+Np] = 1e18*1000*water_volumes*molec_masses[name]*dH2SO4 # fg/s
    
    # BISULFATE
    name = 'BISULFATE'
    dydt_eq = (dH2SO4*K1)/Hplus # mol/L*s
    dHplus += dydt_eq
    dHSO4 = dydt_eq
    dydt[dx+Gaq[name]*Np:dx+Gaq[name]*Np+Np] = 1e18*1000*water_volumes*molec_masses[name]*dHSO4 # fg/s
    
    # SULFATE
    name = 'SULFATE'
    dydt_eq = ((dHSO4*K2)/Hplus)
    dHplus += 2.0*dydt_eq
    dSO4 = dydt_eq # mol/L*s
    dydt[dx+Gaq[name]*Np:dx+Gaq[name]*Np+Np] = 1e18*1000*water_volumes*molec_masses[name]*dSO4 # fg/s
    
    # pH
    name = 'pH'
    dpH = (-0.434294*dHplus)/Hplus
    dydt[dx+Gaq[name]*Np:dx+Gaq[name]*Np+Np] = dpH # fg/s
    
    return dydt, gas2part






