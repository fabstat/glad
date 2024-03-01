# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 09:53:30 2022

@author: beel083
"""
import numpy as np
import indexing, pickle
#from pyrcel import thermo
#import matplotlib.pyplot as plt
from numba import njit
#import sys

Kw = 1E-14
gravity = 9.8 # m/s
R = 8.314 # m^3*Pa/mol*K
Lv_H2O = 2.25e6 # latent heat of vaporization of water, Pa*m^3/kg*K
R_dry = 287.0 # Pa*m^3/kg*K
Cp = 1004.0 # specific heat of dry air at constant pressure
Na = 6.022E23

Mw = 0.018 # molar mass of water, kg/mol
Ma = 28.9/1000.0 # molar mass of dry air, kg/mol
#rho_w = 1000.0 # density of water, kg/m^3


@njit(error_model='numpy')
def parcel_properties(t, y, dydt, V, Np, thermal_accom, condens_coeff, densities, kappas, molec_masses, Ig, Gaq, dx):
    
    # make sure everything is >= 0
    for name in Ig:
        if y[Ig[name]] < 0 and name != 'SS':
            y[Ig[name]] = 0
    for name in Gaq:
        for i in range(0, Np):
            if y[dx+Gaq[name]*Np+i] < 0:
                y[dx+Gaq[name]*Np+i] = 0
    
    # get current state of parcel
    T = y[Ig['T']]
    P = y[Ig['P']]
    SS = y[Ig['SS']]
    es = 611.2*np.exp((17.67*(T-273.15))/(T-273.15+243.5))
    wv = (SS + 1.0) * (0.622 * es / (P - es)) # kg water vapor per kg air
    
    # get current aerosol state
    Ns = y[dx+Gaq['N']*Np:dx+Gaq['N']*Np+Np] # this is the way to index everything using g and dx
    dry_radii, wet_radii = calculate_radii(y, densities, Gaq, dx, Np)
    kappas = calculate_kappas(y, densities, kappas, molec_masses, Gaq, dx, Np)
        
    # change in altitude and pressure
    dz_dt = V # m/s
    Tv = T*(1 + 0.61*wv) # K
    dP_dt = -((gravity*P)/(R_dry*Tv))*dz_dt # Pa/s
    dydt[Ig['z']] = dz_dt
    dydt[Ig['P']] = dP_dt
    
    # saturation vapor pressure, Pa
    Pv_sat = 611.2*np.exp((17.67*(T-273.15))/(T-273.15+243.5))
    
    ## Non-continuum diffusivity/thermal conductivity of air near
    ## near particle
    P_atm = P/101325 # atm
    Dv = np.power(10.0, -4)*(0.211/P_atm)*np.power(T/273.15, 1.94)
    ka = np.power(10.0, -3)*(4.39 + 0.071*T)
     
    Tv = T*(1+0.61*wv)
    rho_air = P/(R_dry*Tv)
    ka_r = ka/(1+(ka/(thermal_accom*wet_radii*rho_air*Cp))*np.sqrt((2*np.pi*Mw)/(R*T)))
    Dv_r = Dv/(1+(Dv/(condens_coeff*wet_radii))*np.sqrt((2*np.pi*Mw)/(R*T)))
    G_a = (densities['H2O']*R*T)/(Pv_sat*Dv_r*Mw)
    G_b = (Lv_H2O*densities['H2O']*((Lv_H2O*Mw/(R*T))-1.0))/(ka_r*T)
    G = 1.0 / (G_a + G_b)
    
    # change in wet radius of each particle
    sigma_water = 0.0761-1.55E-4*(T-273.15) # J/m^2
    a_w = np.power(1.0+kappas*(np.power(dry_radii,3)/(np.power(wet_radii, 3)-np.power(dry_radii, 3))), -1)
    Seq = a_w*np.exp((2.0*sigma_water*Mw)/(R*T*densities['H2O']*wet_radii)) - 1.0
    dr_dt = (G/wet_radii) * (SS-Seq) # m/s
        
    # water mass change in each particle
    dV_dt = 4.0*np.pi*(wet_radii**2)*dr_dt # m^3/s
    dmH2O = dV_dt*densities['H2O'] # kg/s
    dydt[dx+Gaq['H2O']*Np:dx+Gaq['H2O']*Np+Np] = dmH2O*1e18 #fg/s 
    
    # change in liquid water and water vapor mixing ratios,
    # temperature, pressure, and supersaturation
    dwc_dt = ((4.0*np.pi*densities['H2O'])/rho_air)*Ns*np.power(wet_radii, 2)*dr_dt
    dwc_dt = np.sum(dwc_dt)
    dwv_dt = -dwc_dt
    dT_dt = ((-gravity/Cp)*dz_dt)-((Lv_H2O/Cp)*dwv_dt)
    dydt[Ig['T']] = dT_dt
    alpha = ((gravity*Mw*Lv_H2O)/(Cp*R*np.power(T, 2)))-((gravity*Mw)/(R*T))
    gamma = ((P*Ma)/(Pv_sat*Mw))+((Mw*np.power(Lv_H2O, 2))/(Cp*R*np.power(T, 2)))
    dSS_dt = alpha*(dz_dt)-gamma*(dwc_dt)
    dydt[Ig['SS']] = dSS_dt
    
    return dydt


@njit(error_model='numpy')
def calculate_radii(y, densities, g, dx, Np):
    
    V_dry = np.zeros(Np)
    V_wet = np.zeros(Np)
    for species in densities:
        if species != 'H2O':
            V_dry += (y[dx+g[species]*Np:dx+g[species]*Np+Np]*1e-18)/densities[species] # m^3
    V_wet = V_dry + (y[dx+g['H2O']*Np:dx+g['H2O']*Np+Np]*1e-18)/densities['H2O'] # m^3
    r_dry = np.power((3.0*V_dry)/(4.0*np.pi), 1.0/3.0) # m
    r_wet = np.power((3.0*V_wet)/(4.0*np.pi), 1.0/3.0) # m
    return r_dry, r_wet


@njit(error_model='numpy')
def calculate_kappas(y, densities, kappas, molec_masses, g, dx, Np):
    
    V_dry = np.zeros(Np)
    Vi_ki = np.zeros(Np)
    for species in densities:
        if species != 'H2O':
            V_dry += y[dx+g[species]*Np:dx+g[species]*Np+Np]/densities[species] # m^3
            Vi_ki += (y[dx+g[species]*Np:dx+g[species]*Np+Np]/densities[species])*kappas[species] # m^3
    return (1/V_dry)*Vi_ki


