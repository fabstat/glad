#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:36:32 2023

@author: paytonbeeler
"""

import numpy as np
import matplotlib.pyplot as plt
# import tqdm, pickle, indexing
# import new_aq as aq_chemistry
# from scipy.optimize import curve_fit
import sys, os, shutil, pickle, warnings
from numba import types
# from scipy.special import erf
from numba.typed import Dict
# from scipy.integrate import ode
from scipy.optimize import fsolve

R = 8.314 # m^3*Pa/mol*K
Na = 6.022E23

warnings.filterwarnings('ignore')

def build_size_distribution(mode_fractions, dpgs, sigmas, Ntot, Dp_min=100.0, Dp_max=1000.0, plots=False, bins=101):
    if plots == True:
        fig, (ax) = plt.subplots(1, 1, figsize=(1*6.4, 1*4.8), constrained_layout=True)    
    Dps = np.logspace(np.log10(Dp_min), np.log10(Dp_max), bins)
    Ns = np.zeros(bins)
    for i in range(0, len(mode_fractions)):
        N_temp = lognormal_distribution(Dps, Ntot*mode_fractions[i], dpgs[i], sigmas[i])
        Ns += N_temp
        if plots == True:
            ax.plot(Dps, N_temp, label='mode '+str(i+1))
    if plots == True:
        ax.plot(Dps, Ns, '-k', label='total')
        ax.set_ylim(0, )
        ax.set_xlim(Dp_min, Dp_max)
        ax.legend()
        ax.set_xscale('log')
        ax.set_xlabel(r'diameter, $d_p$ (nm)')
        ax.set_ylabel(r'number concentration, $dN/dlog(d_p)$ ($cm^{-3}$)')
        fig.savefig('initialization_figures/size_distribution.png')
    return Dps, Ns # Dps are in nm and Ns are in cm^3
    
    
def lognormal_distribution(x, Ntot, Dpg, sigma):
    prefactor = Ntot/(np.sqrt(2.0*np.pi)*np.log(sigma)*x)
    numerator = -1.0*np.power(np.log(x)-np.log(Dpg), 2)
    denominator = 2.0*np.log(sigma)*np.log(sigma)
    N = prefactor*np.exp(numerator/denominator)
    return N


def sample_diameters(Dp_dist, N_dist, Np, plots = True):
    
    min_Dps = Dp_dist[:-1]
    max_Dps = Dp_dist[1:]
    min_exponent = np.log10(np.min(min_Dps))
    max_exponent = np.log10(np.max(max_Dps))
    
    Dps = []
    Ns = []
    
    temp_Dps = np.zeros(Np)
    temp_Ns = np.zeros(Np)
    particles_in_bin = np.zeros(len(min_Dps))

    while np.all(particles_in_bin) == False:
        particle_bins = np.zeros(Np)
        particles_in_bin = np.zeros(len(min_Dps))
        for i in range(0, Np):
            rand = np.random.rand(1)
            exponent = min_exponent + ((max_exponent-min_exponent)*rand[0])
            sampled_Dp = np.power(10, exponent)
            temp_Dps[i] = sampled_Dp*1e-9 # m
            for j in range(0, len(particles_in_bin)):
                if sampled_Dp >= min_Dps[j] and sampled_Dp < max_Dps[j]:
                    particles_in_bin[j] += 1
                    particle_bins[i] = j
        
        for i in range(0, len(particles_in_bin)):
            numbers = np.random.rand(int(particles_in_bin[i]))
            numbers /= np.sum(numbers)
            counter = 0
            for j in range(0, len(temp_Dps)):
                if int(particle_bins[j]) == i:
                    temp_Ns[j] = numbers[counter]*N_dist[i]*100**3
                    counter += 1
    
    for i in range(0, len(temp_Ns)):
        Dps.append(temp_Dps[i])
        Ns.append(temp_Ns[i])
      
    if plots==True:
        fig, (ax) = plt.subplots(1, 1, figsize=(1*6.4, 1*4.8), constrained_layout=True)    
        bottom = np.zeros(len(min_Dps)-1)
        widths = max_Dps[:-1] - min_Dps[:-1]
        hist = plt.hist(np.array((Dps))*1e9, bins = min_Dps, weights = np.array((Ns))/100**3, color = 'green', alpha = 0.0)
        ax.bar(min_Dps[:-1], hist[0], width = widths, bottom = bottom, facecolor = 'grey', label = 'sampled particles', edgecolor = 'k')
        ax.plot(Dp_dist, N_dist, '-r', label='size distribution')
        ax.set_xscale('log')
        ax.set_xlim(np.min(Dp_dist), np.max(Dp_dist))
        ax.set_ylim(0, )
        ax.legend()
        ax.set_xlabel(r'diameter, $d_p$ (nm)')
        ax.set_ylabel(r'number concentration, $dN/dlog(d_p)$ ($cm^{-3}$)')
        fig.savefig('initialization_figures/sampled_particle_sizes.png')
        
    return Dps, Ns # Dps are in m and Ns are in m^3


def distribute_masses(y0, kappa_i, density_i, Dps, Ns, Np, avg_POM_fraction, POM_fraction_std, min_sulfate_fraction, dx, Gaq):
    
    kappas=[]
    for i in range(0, len(Dps)):
        sulfate_mass_fraction = 0
        while(sulfate_mass_fraction < min_sulfate_fraction):
            POM_mass_fraction = np.random.normal(loc = avg_POM_fraction, scale = POM_fraction_std, size = 1)
            sulfate_mass_fraction = 1-POM_mass_fraction[0]
            Vs_Vpom = (sulfate_mass_fraction/POM_mass_fraction[0])*(density_i['POM']/density_i['SULFATE'])
            Vpom = 1/(Vs_Vpom+1) # volume fraction of POM
            Vs = 1-Vpom # volume fraction of sulfate
            Vtot = (4/3)*np.pi*(Dps[i]/2)**3 # m^3
            mass_sulfate = Vs*Vtot*density_i['SULFATE'] # kg
            mass_POM = Vpom*Vtot*density_i['POM'] # kg
            y0[dx+Gaq['SULFATE']*Np+i] = 1e18*mass_sulfate # fg
            y0[dx+Gaq['POM']*Np+i] = 1e18*mass_POM # fg
            kappas.append(np.average([kappa_i['SULFATE'], kappa_i['POM']], weights=[Vs, Vpom]))
    
    return y0, kappas


def equilibrate_h2o(dry_diameters, kappas, Ns, SS, P, T):
    
    R = 8.314 # m^3*Pa/mol*K
    Mw = 18.015/1000.0 # molar mass of water, kg/mol
    rho_w = 1000.0 # density of water, kg/m^3

    sigma_water = 0.0761-1.55E-4*(T-273.15) # J/m^2
    mass_water = lambda radius, dry_radius, Ni: (4.0*np.pi/3.0)*rho_w*(radius**3-dry_radius**3)
    
    diameters = []
    mH2O = []
    water_masses = []
    
    for D_dry, kappa, N in zip(dry_diameters, list(kappas), Ns):
        r_dry = D_dry/2.0
        a_w = lambda r: np.power(1.0+kappa*(np.power(r_dry,3)/(np.power(r, 3)-np.power(r_dry, 3))), -1)
        Seq = lambda r: a_w(r)*np.exp((2.0*sigma_water*Mw)/(R*T*rho_w*r)) - 1.0
        f = lambda r: Seq(r) - SS
        
        r = fsolve(f, r_dry)
                        
        if r[0] < r_dry:
            r[0] = r_dry
        
        mass_w = mass_water(r, r_dry, N)
        diameters.append(2.0*r[0])
        water_masses.append(mass_w[0])
        mH2O.append(mass_w[0])
    
    diameters = np.array((diameters))
    dry_diameters = np.array((dry_diameters))

    return np.array(water_masses)



def get_H_conc(y0, Np, P_H2SO4, densities, molec_masses, H0, dx, Ig, Gaq):
    
    P_H2SO4 /= 101325 # convert from Pa to atm
    water_volumes = 1000*((y0[dx+Gaq['H2O']*Np:dx+Gaq['H2O']*Np+Np]*1e-18)/densities['H2O']) # L
    sulfate_conc = ((y0[dx+Gaq['SULFATE']*Np:dx+Gaq['SULFATE']*Np+Np]*1e-18)/molec_masses['SULFATE'])/water_volumes # mol/L    
    K1 = 1000 # mol/L
    K2 = 1.02E-2*np.exp(2720*((1/T)-(1/298))) # mol/L 
    pHs = np.zeros(Np)
    bisulf_masses = np.zeros(Np)
    H2SO4_masses = np.zeros(Np)
    
    for i in range(0, Np):
        
        cube_term = (-1.0*sulfate_conc[i])/(K1*K2*P_H2SO4)
        square_term = H0['H2SO4']
        linear_term = H0['H2SO4']
        constant = H0['H2SO4']*K1*K2
        
        f = lambda x: cube_term*x**3+square_term*x**2+linear_term*x+constant
        H = fsolve(f, 1e1)
        pHs[i] = -1.0*np.log10(H[0])
        
        bisulf_conc = sulfate_conc[i]*(H[0]/K2) # mol/L
        H2SO4_conc = bisulf_conc*(H[0]/K1) # mol/L
        bisulf_masses[i] = 1e18*bisulf_conc*1000*water_volumes[i]*molec_masses['BISULFATE'] # fg
        H2SO4_masses[i] = 1e18*H2SO4_conc*1000*water_volumes[i]*molec_masses['H2SO4'] # fg
    
    y0[dx+Gaq['pH']*Np:dx+Gaq['pH']*Np+Np] = pHs
    y0[dx+Gaq['BISULFATE']*Np:dx+Gaq['BISULFATE']*Np+Np] = bisulf_masses
    y0[dx+Gaq['H2SO4']*Np:dx+Gaq['H2SO4']*Np+Np] = H2SO4_masses
    
    return y0



# ============================================================================================================================================

print('Initializing run ...')

# read in run information
T = float(sys.argv[1])
y0_filename = str(sys.argv[2])
Ntot = float(sys.argv[3])
mode_mean_diameters = str(sys.argv[4])
mode_standard_deviations = str(sys.argv[5])
lognormal_mode_fractions = str(sys.argv[6])
Np = int(sys.argv[7])
avg_POM_fraction=float(sys.argv[9])
POM_fraction_std=float(sys.argv[10])
min_sulfate_fraction=float(sys.argv[11])
SS=float(sys.argv[12])
P=float(sys.argv[13])
SO2_gas_conc=float(sys.argv[14])
OH_gas_conc=float(sys.argv[15])
H2SO4_gas_conc=float(sys.argv[16])
O3_gas_conc=float(sys.argv[17])


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
  
# read alphas from files
filename = 'alphas.txt'
raw_data = np.loadtxt(filename, dtype='str', delimiter = ':')
alphas = Dict.empty(key_type=types.unicode_type, value_type=types.float64)
for i in range(0, len(raw_data)):
    alphas[str(raw_data[i, 0])] = float(raw_data[i, 1])

# read in size distribution information
dpgs=np.zeros(0)
start=1
for i in range(0, len(mode_mean_diameters)):
    if mode_mean_diameters[i] == ',' or mode_mean_diameters[i] == ']':
        dpgs=np.append(dpgs, float(mode_mean_diameters[start:i]))
        start=i+1
sigmas=np.zeros(0)
start=1
for i in range(0, len(mode_standard_deviations)):
    if mode_standard_deviations[i] == ',' or mode_standard_deviations[i] == ']':
        sigmas=np.append(sigmas, float(mode_standard_deviations[start:i]))
        start=i+1
mode_fractions=np.zeros(0)
start=1
for i in range(0, len(lognormal_mode_fractions)):
    if lognormal_mode_fractions[i] == ',' or lognormal_mode_fractions[i] == ']':
        mode_fractions=np.append(mode_fractions, float(lognormal_mode_fractions[start:i]))
        start=i+1
if len(dpgs) != len(sigmas) or len(dpgs) != len(mode_fractions) or len(sigmas) != len(mode_fractions):
    print('ERROR: Number of mode fractions, geometric mean diameters, and standard deviations do not match!')
    sys.exit()
if np.sum(mode_fractions) != 1:
    print('ERROR: Sum of lognormal mode fractions does not equal 1!')
    sys.exit()
    
# convert make_plots to boolean variable
plots = int(sys.argv[8])
if plots==1:
    plots=True
else:
    plots=False
if plots == True:
    if os.path.isdir('initialization_figures') == False:
        os.mkdir('initialization_figures')
else:
    if os.path.isdir('initialization_figures') == True:
        shutil.rmtree('initialization_figures')

# check the mass fractions
if avg_POM_fraction > 1 or avg_POM_fraction <= 0:
    print('ERROR: average POM mass fraction is > 1 or <= 0!')
    sys.exit()
if POM_fraction_std <= 0:
    print('ERROR: standard deviation of POM mass fraction is <= 0!')
    sys.exit()
if min_sulfate_fraction > 1 or min_sulfate_fraction <= 0:
    print('ERROR: minimum sulfate mass fraction is > 1 or <= 0!')
    sys.exit()
    
# read gas input and aqueous species
Ig = pickle.load(open('model_species_gas.pkl', 'rb'))
Gaq = pickle.load(open('model_species_aq.pkl', 'rb'))
dx = len(Ig.keys())
y0 = np.zeros(dx+len(Gaq)*Np+Np)

# build size distribution
Dps, Ns = build_size_distribution(mode_fractions, dpgs, sigmas, Ntot, plots=plots)

# randomly sample particle sizes and weights based on distribution
Dps, Ns = sample_diameters(Dps, Ns, Np, plots=plots)
y0[dx+Gaq['N']*Np:dx+Gaq['N']*Np+Np] = Ns

# distribute mass of POM and sulfate in particles
y0, kappas = distribute_masses(y0, kappas, densities, Dps, Ns, Np, avg_POM_fraction, POM_fraction_std, min_sulfate_fraction, dx, Gaq)

# equilibrate water
water_masses = equilibrate_h2o(Dps, kappas, Ns, SS, P, T)
y0[dx+Gaq['H2O']*Np:dx+Gaq['H2O']*Np+Np] = water_masses*1e18 # fg

# set up the gas phase
P_SO2 = P*SO2_gas_conc*1e-9 # Pa
P_OH = P*OH_gas_conc*1e-9 # Pa
P_H2SO4 = P*H2SO4_gas_conc*1e-9 # Pa
P_O3 = P*O3_gas_conc*1e-9 # Pa
y0[Ig['SO2']] = (P_SO2*Na)/(R*T*100**3) # molec/cm^3
y0[Ig['OH']] = (P_OH*Na)/(R*T*100**3) # molec/cm^3
y0[Ig['H2SO4']] = (P_H2SO4*Na)/(R*T*100**3) # molec/cm^3
y0[Ig['O3']] = (P_O3*Na)/(R*T*100**3) # molec/cm^3

# set up ambient variables
y0[Ig['z']] = 0.0
y0[Ig['P']] = P
y0[Ig['T']] = T
y0[Ig['SS']] = SS

# find hydrogen ion concentration in each particle and balance species
y0 = get_H_conc(y0, Np, P_H2SO4, densities, molec_masses, H0, dx, Ig, Gaq)

print('Writing '+y0_filename+' ...')

f = open(y0_filename, 'w')
for x in Ig:
    f.write(x)
    f.write(': ')
    f.write(str(y0[Ig[x]]))
    f.write('\n')
for x in Gaq:
    if x != 'N' and x != 'pH':
        for i in range(0, Np):
            name = x + ' (aq,' + str(i+1) + ')'
            f.write(name)
            f.write(': ')
            f.write(str(y0[dx+Gaq[x]*Np+i]))
            f.write('\n')
    else:
        for i in range(0, Np):
            name = x + ', '  + str(i+1)
            f.write(name)
            f.write(': ')
            f.write(str(y0[dx+Gaq[x]*Np+i]))
            f.write('\n')
f.close()

print(' ')


