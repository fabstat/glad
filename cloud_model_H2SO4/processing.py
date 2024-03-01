#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 14:15:10 2022

@author: paytonbeeler
"""

import pickle#, os, indexing, sys
import matplotlib.pyplot as plt
#from pyrcel import thermo
import numpy as np
import parcel, indexing
import imageio, os, tqdm, sys


'''
def format_solution(soln, bins, co_condens):
    
    for i in range(0, len(soln)):
        for j in range(6, 79):
            if soln[i, j] < 0:
                soln[i, j] = 0
        for j in range(79+6*bins, len(soln[0])):
            if soln[i, j] < 0:
                soln[i, j] = 0  

    return soln
'''
def write_files(soln, file, V, t, y0_org_array, Np, densities, kappas, molec_masses):
    
    # G is a dict with the group number of each aqueous species, 
    # I is a dict with the indices of the gas species/parcel properties,
    # dx is the number of parcel/gas properties tracked in the model
    Ig, Gaq, dx = indexing.getGroupNumbers(y0_org_array, Np)
    
    output = {}
    output['velocity'] = V
    output['t'] = t
    
    i = 0
    while(i < len(y0_org_array)):
        if ',' in y0_org_array[i] and y0_org_array[i][:-2] != y0_org_array[i-1][:-2]:
            if '(aq,' in y0_org_array[i]:
                j = y0_org_array[i].rfind('(')
                name = y0_org_array[i][:j-1] + ' (aq)'
                output[name] = soln[:, i:i+Np]
            else:
                j = y0_org_array[i].rfind(',')
                name = y0_org_array[i][:j]
                output[name] = soln[:, i:i+Np]
            i += Np
        else:
            name = y0_org_array[i]
            output[name] = soln[:, i]
            i += 1
    
    # calculate kappas, wet and dry radii at t = 0 and add to dict
    dry_radii, wet_radii = parcel.calculate_radii(soln[0], densities, Gaq, dx, Np)
    k = parcel.calculate_kappas(soln[0], densities, kappas, molec_masses, Gaq, dx, Np)
    output['Ddry'] = 2.0*dry_radii
    output['Dp'] = 2.0*wet_radii
    output['kappa'] = k
        
    for i in range(1, len(soln)):
         dry_radii, wet_radii = parcel.calculate_radii(soln[i], densities, Gaq, dx, Np)
         k = parcel.calculate_kappas(soln[i], densities, kappas, molec_masses, Gaq, dx, Np)
         output['Ddry'] = np.vstack((output['Ddry'], 2.0*dry_radii))
         output['Dp'] = np.vstack((output['Dp'], 2.0*wet_radii))
         output['kappa'] = np.vstack((output['kappa'], k))
    
    pickle.dump(output, open(file,'wb'))
    
    return
'''
def particle_piechart_movie(particle, data, duration, model_setup, output_directory):
    
    fps = 30
    frames = fps*duration
    di = int(np.ceil(len(data['t'])/frames))
    
    images = []
    print('WRITING IMAGE FILES FOR PARTICLE '+str(particle)+':')
    pbar = tqdm.tqdm(total = int(len(data['t'])/di)+1)
    for i in range(0, len(data['t']), di):
        makeframe(i, particle, data, output_directory, model_setup, pbar)
        images.append(imageio.imread(output_directory+'/temp.png'))
        pbar.update(1)
    pbar.close()
    print(' ')
    fps = int(len(images)/duration)
    kargs = {'fps':fps, 'format':'FFMPEG', 'macro_block_size': None, 
              'ffmpeg_params': ['-s','928x352']}
    imageio.mimsave(output_directory+'/composition_tracking_'+str(particle)+'.mp4', images, **kargs)
    os.remove(output_directory+'/temp.png')
    


def makeframe(i, particle, data, output_directory, model_setup, pbar):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2.0*6.4, 1.0*4.8), constrained_layout=True)
    
    names = []
    for x in model_setup.keys():
        names.append(x)
    colors = ['green', 'lime', 'blue', 'red', 'C6', 'purple', 'gold', 'grey', 'turquoise', 'organge', 'darkorange']
    values = np.zeros(len(names))
    
    for j in range(0, len(names)):
        for n in model_setup[names[j]]:
            values[j] += data[n+' (aq)'][i, particle]
            
    
    for j in range(0, len(names)):
        if values[j] < 0:
            if abs(values[j]) > 1e-15:
                pbar.close()
                print(names[j], values[j])
                print(' ')
                print(' ')
                raise ValueError('Particle has negative mass of '+str(names[j])+'!')
            else:
                values[j] = 0
            

    ax2.pie(values, colors = colors, wedgeprops = {'linewidth': 1.0, 'edgecolor': 'k'})
    ax2.legend(names, loc = 'center', ncol = 1, bbox_to_anchor = (1.3, 0.5), frameon=False, fontsize=14)
    
    ax1.plot(data['Dp']*1e6, data['z'], '-k')
    ax1.set_xscale('log')
    ax1.set_xlabel('wet diameter (micron)', fontsize = 14)
    ax1.set_ylabel('altitude (m)', fontsize = 14)
    ax1.set_ylim(np.min(data['z']), np.max(data['z']))
    
    ax1.plot(1e6*data['Dp'][i, particle], data['z'][i], 'ro')
    
    fig.savefig(output_directory+'/temp.png')
    plt.close()
    
    return 


def mass_changes(data, model_setup, particle):
    
    fig, (ax) = plt.subplots(1, 1, figsize=(1.3*6.4, 1.0*4.8), constrained_layout=True)
    
    names = []
    for x in model_setup.keys():
        names.append(x)
    colors = ['green', 'lime', 'blue', 'red', 'C6', 'purple', 'gold', 'grey', 'turquoise', 'organge', 'darkorange']
    changes = np.zeros((len(names), len(data['t'])))
    
    for j in range(0, len(names)):
        for n in model_setup[names[j]]:
            changes[j] += data[n+' (aq)'][:, particle]
    
    bottom = np.zeros(len(data['t']))     
    for i in range(0, len(changes)):
        ax.fill_between(data['t'], bottom, bottom+changes[i], color=colors[i], label=names[i])
        bottom = bottom+changes[i]
        
    ax.set_xlabel('time (s)', fontsize = 14)
    ax.set_ylabel('mass (fg)', fontsize = 14)
    ax.legend(loc='center', bbox_to_anchor=(1.25, 0.5), fontsize = 14, frameon=False)
    ax.set_ylim(0, )
    ax.set_xlim(0, )
    
    print(' ')
    print('CHANGE IN MASSES (fg):')
    for i in range(0, len(changes)):
        if abs(changes[i][-1]-changes[i][0]) < 1e-20:
            dm = 0.0
        else:
            dm = changes[i][-1]-changes[i][0]
        print(names[i]+':', dm)
        
    return



def size_changes(data, model_setup, densities, particle):
    
    fig, (ax) = plt.subplots(1, 1, figsize=(1.3*6.4, 1.0*4.8), constrained_layout=True)
    
    names = []
    for x in model_setup.keys():
        names.append(x)
    colors = ['green', 'lime', 'blue', 'red', 'C6', 'purple', 'gold', 'grey', 'turquoise', 'organge', 'darkorange']
    #changes = np.zeros((len(names), len(data['t'])))
    total_volume = (4.0/3.0)*np.pi*np.power(0.5*data['Ddry'][:,particle], 3) # m^3
    V_percentage = np.zeros((len(names), len(data['t'])))
    
    for j in range(0, len(names)):
        V = 0.0
        for n in model_setup[names[j]]:
            V += (1e-18*data[n+' (aq)'][:, particle])/densities[n] # m^3
        V_percentage[j] = V/total_volume
          
    bottom = np.zeros(len(data['t']))     
    for i in range(0, len(V_percentage)):
        ax.fill_between(data['t'], bottom, bottom+(V_percentage[i]*1e9*data['Ddry'][:, particle]), color=colors[i], label=names[i])
        bottom = bottom+(V_percentage[i]*1e9*data['Ddry'][:, particle])
        
    ax.set_xlabel('time (s)', fontsize = 14)
    ax.set_ylabel('Dry Diameter (nm)', fontsize = 14)
    ax.legend(loc='center', bbox_to_anchor=(1.25, 0.5), fontsize = 14, frameon=False)
    ax.set_ylim(0, )
    ax.set_xlim(0, )
    
    print(' ')
    print('CHANGE IN DIAMETER (nm):')
    for i in range(0, len(V_percentage)):
        Ddry = 1e9*((V_percentage[i][-1]*data['Ddry'][-1, particle]) - (V_percentage[i][0]*data['Ddry'][0, particle]))
        print(names[i]+':', Ddry)
    print('TOTAL:', 1e9*(data['Ddry'][-1, particle] - data['Ddry'][0, particle]))
    
    return





def classify_single_particle(mass_thresholds, masses):
    
    differences = {}
    differences['BC'] = 100*abs((masses['BC']-mass_thresholds['BC'][1])/mass_thresholds['BC'][1])
    differences['dust'] = 100*abs((masses['dust']-mass_thresholds['dust'][1])/mass_thresholds['dust'][1])
    differences['sulfate'] = 100*abs((masses['sulfate']-mass_thresholds['sulfate/nitrate']['sulfate'][1])/mass_thresholds['sulfate/nitrate']['sulfate'][1])
    differences['nitrate'] = 100*abs((masses['nitrate']-mass_thresholds['nitrate']['nitrate'][1])/mass_thresholds['nitrate']['nitrate'][1])
    differences['organics'] = 100*abs((masses['organics']-mass_thresholds['organics'][1])/mass_thresholds['organics'][1])
    
    differences = dict(sorted(differences.items(), key=lambda item: item[1]))

    #find out which group the particle belongs to
    if list(differences.keys())[0] == 'sulfate':
        particle_class = 'sulfate'
        index = 'sulfate/nitrate'
    elif list(differences.keys())[0] == 'nitrate':
        particle_class = 'nitrate'
        index = 'nitrate'
    elif list(differences.keys())[0] == 'organics':
        particle_class = 'organics'
        index = 'organics'
    elif list(differences.keys())[0] == 'BC':
        particle_class = 'BC'
        index = 'BC'
    elif list(differences.keys())[0] == 'dust':
        particle_class = 'dust'
        index = 'dust'
    
    return particle_class, index
    


def classify_particles_thresholds(filename, tolerance, mass_thresholds_file):
    
    file = open(filename, 'rb')
    data = pickle.load(file)
    
    bins = len(data['BC'][0])
    
    # read mass threhsolds from file
    filename = mass_thresholds_file
    raw_data = np.loadtxt(filename, dtype='str', delimiter = ' ')

    mass_thresholds = {'BC': [float(raw_data[3][1]), float(raw_data[3][2])],
                       'dust': [float(raw_data[4][1]), float(raw_data[4][2])],
                       'organics': [float(raw_data[2][1]), float(raw_data[2][2])],
                       'nitrate': {'nitrate': [float(raw_data[1][1]), float(raw_data[1][2])], 'organics': [0.1, 0.0]},
                       'sulfate/nitrate': {'sulfate': [float(raw_data[0][1]), float(raw_data[0][2])], 'nitrate': [0.05, 0.0], 'organics': [0.1, 0.0]}}
    
    activated = []
    unactivated = []
    
    for i in range(len(data['Dp'][0])):
        radius = 0.5*data['Dp'][-1, i]
        T = data['T'][-1]
        dry_radius = 0.5*data['Ddry'][-1, i]
        kappa = data['kappa'][-1, i]
        r_crit, SS_crit = thermo.kohler_crit(T, dry_radius, kappa)
        if radius >= r_crit:
            activated.append(i)
        else:
            unactivated.append(i)

    dry_species = ['ARO1', 'ARO2', 'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 
                   'LIM2', 'OC', 'IEPOX', 'IEPOXOS', 'tetrol', 'dust', 
                   'BC', 'NO3-', 'SO4--', 'Cl', 'MSA', 'CO3--', 
                   'Na', 'Ca', 'OIN', 'NH4+']
    
    model_setup = {'organics': ['ARO1', 'ARO2', 'ALK1', 'OLE1', 'API1', 'API2', 'LIM1', 'LIM2', 'OC', 'IEPOX', 'IEPOXOS', 'tetrol'],
                   'dust': ['dust'],
                   'BC': ['BC'],
                   'nitrate': ['NO3-'],
                   'sulfate': ['SO4--'],
                   'others': ['Cl', 'MSA', 'CO3--', 'Na', 'Ca', 'OIN', 'NH4+']}
    
    Ni_activated = {'organics': 0, 'dust': 0, 'BC': 0, 'nitrate': 0, 
               'sulfate/nitrate': 0, 'other': 0}
    
    Nf_activated = {'organics': 0, 'dust': 0, 'BC': 0, 'nitrate': 0, 
               'sulfate/nitrate': 0, 'other': 0}
    
    Ni_unactivated = {'organics': 0, 'dust': 0, 'BC': 0, 'nitrate': 0, 
               'sulfate/nitrate': 0, 'other': 0}
    
    Nf_unactivated = {'organics': 0, 'dust': 0, 'BC': 0, 'nitrate': 0, 
               'sulfate/nitrate': 0, 'other': 0}        
    
    print(' ')
    for particle in activated:
        
        # ======================================
        # ======================================
        # ======================================
        # initial composition of activated particles
        # ======================================
        # ======================================
        # ======================================
        
        #find total mass of particle at t=0
        total_mass = 0
        for species in dry_species:
            total_mass += data[species][0, particle]
        
        #find mass of each species at t=0
        masses = {}

        dust_mass = 0
        for species in model_setup['dust']:
            dust_mass += data[species][0, particle]
        masses['dust'] = dust_mass/total_mass
        
        BC_mass = 0
        for species in model_setup['BC']:
            BC_mass += data[species][0, particle]
        masses['BC'] = BC_mass/total_mass
        
        nitrate_mass = 0
        for species in model_setup['nitrate']:
            nitrate_mass += data[species][0, particle]
        masses['nitrate'] = nitrate_mass/total_mass
        
        sulfate_mass = 0
        for species in model_setup['sulfate']:
            sulfate_mass += data[species][0, particle]
        masses['sulfate'] = sulfate_mass/total_mass
        
        org_mass = 0
        for species in model_setup['organics']:
            org_mass += data[species][0, particle]
        org_mass = org_mass/total_mass
        masses['organics'] = org_mass
        
        other_mass = 0
        for species in model_setup['others']:
            other_mass += data[species][0, particle]
        other_mass = other_mass/total_mass
        masses['others'] = other_mass
        
        if masses['BC'] > mass_thresholds['BC'][1]-tolerance:
            original_class = 'BC'
            Ni_activated['BC'] += data['N'][0, particle]
        elif masses['dust'] > mass_thresholds['dust'][1]-tolerance:
            original_class = 'dust'
            Ni_activated['dust'] += data['N'][0, particle]
        elif masses['sulfate'] > mass_thresholds['sulfate/nitrate']['sulfate'][1]-tolerance:
            original_class = 'sulfate'
            Ni_activated['sulfate/nitrate'] += data['N'][0, particle]
        elif masses['nitrate'] > mass_thresholds['nitrate']['nitrate'][1]-tolerance:
            original_class = 'nitrate'
            Ni_activated['nitrate'] += data['N'][0, particle]
        elif masses['organics'] > mass_thresholds['organics'][1]-tolerance:
            original_class = 'organics'
            Ni_activated['organics'] += data['N'][0, particle]
        else:
            original_class, index = classify_single_particle(mass_thresholds, masses)
            Ni_activated[index] += data['N'][0, particle]
        
        # ======================================
        # ======================================
        # ======================================
        # final composition of activated particles
        # ======================================
        # ======================================
        # ======================================
        
        #find total mass of particle at t=t_end
        total_mass = 0
        for species in dry_species:
            total_mass += data[species][-1, particle]
        
        #find mass of each species at t=0
        masses = {}

        dust_mass = 0
        for species in model_setup['dust']:
            dust_mass += data[species][-1, particle]
        masses['dust'] = dust_mass/total_mass
        
        BC_mass = 0
        for species in model_setup['BC']:
            BC_mass += data[species][-1, particle]
        masses['BC'] = BC_mass/total_mass
        
        nitrate_mass = 0
        for species in model_setup['nitrate']:
            nitrate_mass += data[species][-1, particle]
        masses['nitrate'] = nitrate_mass/total_mass
        
        sulfate_mass = 0
        for species in model_setup['sulfate']:
            sulfate_mass += data[species][-1, particle]
        masses['sulfate'] = sulfate_mass/total_mass
        
        org_mass = 0
        for species in model_setup['organics']:
            org_mass += data[species][-1, particle]
        org_mass = org_mass/total_mass
        masses['organics'] = org_mass
        
        other_mass = 0
        for species in model_setup['others']:
            other_mass += data[species][-1, particle]
        other_mass = other_mass/total_mass
        masses['others'] = other_mass
        
        if masses['BC'] > mass_thresholds['BC'][1]-tolerance:
            final_class = 'BC'
            Nf_activated['BC'] += data['N'][-1, particle]
        elif masses['dust'] > mass_thresholds['dust'][1]-tolerance:
            final_class = 'dust'
            Nf_activated['dust'] += data['N'][-1, particle]
        elif masses['sulfate'] > mass_thresholds['sulfate/nitrate']['sulfate'][1]-tolerance:
            final_class = 'sulfate'
            Nf_activated['sulfate/nitrate'] += data['N'][-1, particle]
        elif masses['nitrate'] > mass_thresholds['nitrate']['nitrate'][1]-tolerance:
            final_class = 'nitrate'
            Nf_activated['nitrate'] += data['N'][-1, particle]
        elif masses['organics'] > mass_thresholds['organics'][1]-tolerance:
            final_class = 'organics'
            Nf_activated['organics'] += data['N'][-1, particle]
        else:
            final_class, index = classify_single_particle(mass_thresholds, masses)
            Nf_activated[index] += data['N'][-1, particle]
    
    for particle in unactivated:
        
        # ======================================
        # ======================================
        # ======================================
        # initial composition of unactivated particles
        # ======================================
        # ======================================
        # ======================================
        
        #find total mass of particle at t=0
        total_mass = 0
        for species in dry_species:
            total_mass += data[species][0, particle]
        
        #find mass of each species at t=0
        masses = {}

        dust_mass = 0
        for species in model_setup['dust']:
            dust_mass += data[species][0, particle]
        masses['dust'] = dust_mass/total_mass
        
        BC_mass = 0
        for species in model_setup['BC']:
            BC_mass += data[species][0, particle]
        masses['BC'] = BC_mass/total_mass
        
        nitrate_mass = 0
        for species in model_setup['nitrate']:
            nitrate_mass += data[species][0, particle]
        masses['nitrate'] = nitrate_mass/total_mass
        
        sulfate_mass = 0
        for species in model_setup['sulfate']:
            sulfate_mass += data[species][0, particle]
        masses['sulfate'] = sulfate_mass/total_mass
        
        org_mass = 0
        for species in model_setup['organics']:
            org_mass += data[species][0, particle]
        org_mass = org_mass/total_mass
        masses['organics'] = org_mass
        
        other_mass = 0
        for species in model_setup['others']:
            other_mass += data[species][0, particle]
        other_mass = other_mass/total_mass
        masses['others'] = other_mass
        
        if masses['BC'] > mass_thresholds['BC'][1]-tolerance:
            Ni_unactivated['BC'] += data['N'][0, particle]
        elif masses['dust'] > mass_thresholds['dust'][1]-tolerance:
            Ni_unactivated['dust'] += data['N'][0, particle]
        elif masses['sulfate'] > mass_thresholds['sulfate/nitrate']['sulfate'][1]-tolerance:
            Ni_unactivated['sulfate/nitrate'] += data['N'][0, particle]
        elif masses['nitrate'] > mass_thresholds['nitrate']['nitrate'][1]-tolerance:
            Ni_unactivated['nitrate'] += data['N'][0, particle]
        elif masses['organics'] > mass_thresholds['organics'][1]-tolerance:
            Ni_unactivated['organics'] += data['N'][0, particle]
        else:
            particle_class, index = classify_single_particle(mass_thresholds, masses)
            Ni_unactivated[index] += data['N'][0, particle]
            
        # ======================================
        # ======================================
        # ======================================
        # final composition of unactivated particles
        # ======================================
        # ======================================
        # ======================================
        
        #find total mass of particle at t=t_end
        total_mass = 0
        for species in dry_species:
            total_mass += data[species][-1, particle]
        
        #find mass of each species at t=0
        masses = {}
        
        dust_mass = 0
        for species in model_setup['dust']:
            dust_mass += data[species][-1, particle]
        masses['dust'] = dust_mass/total_mass
        
        BC_mass = 0
        for species in model_setup['BC']:
            BC_mass += data[species][-1, particle]
        masses['BC'] = BC_mass/total_mass
        
        nitrate_mass = 0
        for species in model_setup['nitrate']:
            nitrate_mass += data[species][-1, particle]
        masses['nitrate'] = nitrate_mass/total_mass
        
        sulfate_mass = 0
        for species in model_setup['sulfate']:
            sulfate_mass += data[species][-1, particle]
        masses['sulfate'] = sulfate_mass/total_mass
        
        org_mass = 0
        for species in model_setup['organics']:
            org_mass += data[species][-1, particle]
        org_mass = org_mass/total_mass
        masses['organics'] = org_mass
        
        other_mass = 0
        for species in model_setup['others']:
            other_mass += data[species][-1, particle]
        other_mass = other_mass/total_mass
        masses['others'] = other_mass
        
        if masses['BC'] > mass_thresholds['BC'][1]-tolerance:
            Nf_unactivated['BC'] += data['N'][-1, particle]
        elif masses['dust'] > mass_thresholds['dust'][1]-tolerance:
            Nf_unactivated['dust'] += data['N'][-1, particle]
        elif masses['sulfate'] > mass_thresholds['sulfate/nitrate']['sulfate'][1]-tolerance:
            Nf_unactivated['sulfate/nitrate'] += data['N'][-1, particle]
        elif masses['nitrate'] > mass_thresholds['nitrate']['nitrate'][1]-tolerance:
            Nf_unactivated['nitrate'] += data['N'][-1, particle]
        elif masses['organics'] > mass_thresholds['organics'][1]-tolerance:
            Nf_unactivated['organics'] += data['N'][-1, particle]
        else:
            particle_class, index = classify_single_particle(mass_thresholds, masses)
            Nf_unactivated[index] += data['N'][-1, particle]
        
    
    uncategorized = Ni_activated['other'] + Ni_unactivated['other'] + Nf_activated['other'] + Nf_unactivated['other']
    
    if uncategorized > 0:
        print(' ')
        print('WARNING: Some particles have not been classified, consider increasing tolerance!')
        print(' ')
    
    original_particle_classes = []
    final_particle_classes = []
    
    for i in range(0, bins):
        #find total mass of particle at t=0
        total_mass = 0
        for species in dry_species:
            total_mass += data[species][0, i]
        
        #find mass of each species at t=0
        masses = {}

        dust_mass = 0
        for species in model_setup['dust']:
            dust_mass += data[species][0, i]
        masses['dust'] = dust_mass/total_mass
        
        BC_mass = 0
        for species in model_setup['BC']:
            BC_mass += data[species][0, i]
        masses['BC'] = BC_mass/total_mass
        
        nitrate_mass = 0
        for species in model_setup['nitrate']:
            nitrate_mass += data[species][0, i]
        masses['nitrate'] = nitrate_mass/total_mass
        
        sulfate_mass = 0
        for species in model_setup['sulfate']:
            sulfate_mass += data[species][0, i]
        masses['sulfate'] = sulfate_mass/total_mass
        
        org_mass = 0
        for species in model_setup['organics']:
            org_mass += data[species][0, i]
        org_mass = org_mass/total_mass
        masses['organics'] = org_mass
        
        other_mass = 0
        for species in model_setup['others']:
            other_mass += data[species][0, i]
        other_mass = other_mass/total_mass
        masses['others'] = other_mass
        
        if masses['BC'] > mass_thresholds['BC'][1]-tolerance:
            original_particle_classes.append('BC')
        elif masses['dust'] > mass_thresholds['dust'][1]-tolerance:
            original_particle_classes.append('dust')
        elif masses['sulfate'] > mass_thresholds['sulfate/nitrate']['sulfate'][1]-tolerance:
            original_particle_classes.append('sulfate/nitrate')
        elif masses['nitrate'] > mass_thresholds['nitrate']['nitrate'][1]-tolerance:
            original_particle_classes.append('nitrate')
        elif masses['organics'] > mass_thresholds['organics'][1]-tolerance:
            original_particle_classes.append('organics')
        else:
            particle_class, index = classify_single_particle(mass_thresholds, masses)
            original_particle_classes.append(particle_class)

        #find total mass of particle at t=t_end
        total_mass = 0
        for species in dry_species:
            total_mass += data[species][-1, i]
        
        #find mass of each species at t=0
        masses = {}

        dust_mass = 0
        for species in model_setup['dust']:
            dust_mass += data[species][-1, i]
        masses['dust'] = dust_mass/total_mass
        
        BC_mass = 0
        for species in model_setup['BC']:
            BC_mass += data[species][-1, i]
        masses['BC'] = BC_mass/total_mass
        
        nitrate_mass = 0
        for species in model_setup['nitrate']:
            nitrate_mass += data[species][-1, i]
        masses['nitrate'] = nitrate_mass/total_mass
        
        sulfate_mass = 0
        for species in model_setup['sulfate']:
            sulfate_mass += data[species][-1, i]
        masses['sulfate'] = sulfate_mass/total_mass
        
        org_mass = 0
        for species in model_setup['organics']:
            org_mass += data[species][-1, i]
        org_mass = org_mass/total_mass
        masses['organics'] = org_mass
        
        other_mass = 0
        for species in model_setup['others']:
            other_mass += data[species][-1, i]
        other_mass = other_mass/total_mass
        masses['others'] = other_mass
        
        if masses['BC'] > mass_thresholds['BC'][1]-tolerance:
            final_particle_classes.append('BC')
        elif masses['dust'] > mass_thresholds['dust'][1]-tolerance:
            final_particle_classes.append('dust')
        elif masses['sulfate'] > mass_thresholds['sulfate/nitrate']['sulfate'][1]-tolerance:
            final_particle_classes.append('sulfate/nitrate')
        elif masses['nitrate'] > mass_thresholds['nitrate']['nitrate'][1]-tolerance:
            final_particle_classes.append('nitrate')
        elif masses['organics'] > mass_thresholds['organics'][1]-tolerance:
            final_particle_classes.append('organics')
        else:
            particle_class, index = classify_single_particle(mass_thresholds, masses)
            final_particle_classes.append(particle_class)
    
    return Ni_activated, Nf_activated, Ni_unactivated, Nf_unactivated, original_particle_classes, final_particle_classes
 
def combine_parcels(file1, file2, new_file):
    
    file = open(file1, 'rb')
    group1 = pickle.load(file)
    file = open(file2, 'rb')
    group2 = pickle.load(file)

    new_data = {}
    new_data['T'] = np.zeros(len(group1['T']))
    new_data['P'] = np.zeros(len(group1['P']))
    new_data['SS'] = np.zeros(len(group1['SS']))
    new_data['z'] = np.zeros(len(group1['z']))
    new_data['t'] = group1['t']
    new_data['wc'] = np.zeros(len(group1['wc']))
    new_data['wv'] = np.zeros(len(group1['wv']))

    for i in range(0, len(group1['T'])):
        new_data['T'][i] = np.mean([group1['T'][i], group2['T'][i]])
        new_data['P'][i] = np.mean([group1['P'][i], group2['P'][i]])
        new_data['SS'][i] = np.mean([group1['SS'][i], group2['SS'][i]])
        new_data['z'][i] = np.mean([group1['z'][i], group2['z'][i]])
        new_data['wc'][i] = np.mean([group1['wc'][i], group2['wc'][i]])
        new_data['wv'][i] = np.mean([group1['wv'][i], group2['wv'][i]])

    filename = os.getcwd() + '/gas_input.txt'
    data = np.loadtxt(filename, dtype='str', delimiter = ": ")

    for i in range(0, len(data)):
        name = data[i][0] + ' (gas)'
        new_data[name] = np.zeros(len(group1[name]))
        for j in range(0, len(group1[name])):
            new_data[name][j] = np.mean([group1[name][j], group2[name][j]])
       
    names = ['Ddry', 'Dp', 'kappa', 'N', 'density', 'pH']

    for name in names:
        new_data[name] = np.hstack((group1[name], group2[name]))

    filename = os.getcwd() + '/particle_input.txt'
    data = np.loadtxt(filename, dtype='str', delimiter = ": ")

    for i in range(0, len(data)):
        name = data[i][0]
        new_data[name] = np.hstack((group1[name], group2[name]))
        
    pickle.dump(new_data, open(new_file,'wb'))
    return
'''    
    
    
