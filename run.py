from glob import glob
from pathlib import Path
import os
import re

# chem_data.chemgns flags
#'action', None, ['prepare', 'analyze', 'predict'], help='Prepare raw data for training or analyze rollout results.'
#'raw_data_path', 'chem_data/raw_data/', help='The raw dataset directory.'
#'preped_data_path', 'gns/data/', help='The path for saving the prepared data for training.'
#'rollout_data_path', 'gns/output/', help='The rollout dataset directory.'
#'proc_data_path', 'chem_data/proc_data/', help='The path for saving the prepared data for training.'
#'material_properties', ['BC', 'OC', 'aero_number'], help='List of material properties.'
#'particle_chem', ['H2O', 'SO4'], help='List of particle phase chemicals.'
#'gases', ['H2SO4'], help='List of gas phase chemicals.'


#### Set these as appropriate:
# PartMC-MOSAIC Data:
# raw_data_path = "./chem_data/processed_output_some/"
# topredict_path = "./chem_data/to_predict/"
# rollout_dicts = "./chem_data/proc_data/"

# npz_path = "./gns/data/"
# model_path = "./gns/model/"
# rollouts_path = "./gns/output/"

# material_properties = ['aero_number', 'BC', 'OC']
# mat_prop_str = "','".join(material_properties)
# particle_chem = ['H2O', 'SO4']
# part_chem_str = "','".join(particle_chem)
# gases = ['H2SO4']
# gases_str = "','".join(gases)
# train_steps = 300
# scenarios = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# total_reps = 0


#### For Payton's Data:
raw_data_path = "./chem_data/processed_output_so2/"
topredict_path = "./chem_data/to_predict/"
rollout_dicts = "./chem_data/proc_data_payton/"

npz_path = "./gns/data/"
model_path = "./gns/model/"
rollouts_path = "./gns/output/"

material_properties = ['aero_number', 'POM']
mat_prop_str = "','".join(material_properties)
particle_chem = ['H2O', 'H2SO4a', 'HSO4', 'SO4']
part_chem_str = "','".join(particle_chem)
gases = ['SO2', 'OH', 'H2SO4']
gases_str = "','".join(gases)
train_steps = 1000
scenarios = [10]
total_reps = 0
####

total_steps = train_steps
old_ts = total_steps
example_number = 10
rollout_number = 0

for dir in os.listdir(raw_data_path):
    if rollout_number < len(scenarios):
        if dir.startswith("."):
            print(f"Skipping {dir}")
            continue

        path = raw_data_path + dir
        example_folder = npz_path + "example" + str(example_number) + "/"
        rollout_folder = rollouts_path + "rollout" + str(scenarios[rollout_number])
        dict_folder = rollout_dicts + "ex" + str(scenarios[rollout_number])
        
        os.system(f"mkdir -p {example_folder}")
        os.system(f"mkdir -p {rollout_folder}")
        os.system(f"mkdir -p {dict_folder}")
        
        os.system(f"python -m chem_data.chemgns --action='prepare' --raw_data_path={path}  --preped_data_path={example_folder} " +
                  f"--universe={scenarios[rollout_number]} --material_properties={mat_prop_str} --gases={gases_str} " + 
                  f"--particle_chem={part_chem_str} --proc_data_path={dict_folder} --share_path={rollout_dicts}")
    
        if total_steps == train_steps:
            os.system(f"python -m gns.train --data_path={example_folder} --model_path={model_path} --output_path={rollout_folder} -ntraining_steps={total_steps}")
        else:
            os.system(f"python -m gns.train --data_path={example_folder} --model_path={model_path} --output_path={rollout_folder} " +
                      f"--model_file='model-{old_ts}.pt' --train_state_file='train_state-{old_ts}.pt' -ntraining_steps={total_steps}")

        os.system(f"python -m gns.train --mode='rollout' --data_path={example_folder} --model_path={model_path} --output_path={rollout_folder} " +
                  f"--model_file='model-{total_steps}.pt' --train_state_file='train_state-{total_steps}.pt'")
        os.system(f"python -m chem_data.chemgns --action='analyze' --rollout_data_path={rollout_folder} " +  
                  f"--material_properties={mat_prop_str} --gases={gases_str} " + 
                  f"--particle_chem={part_chem_str} --proc_data_path={dict_folder} --share_path={rollout_dicts}")

        old_ts = total_steps
        total_steps += train_steps
        example_number += 1
        rollout_number += 1

reps = total_reps
while reps > 0:
    example_folder = npz_path + "example" + str(example_number-1) + "/"
    rollout_folder = rollouts_path + "rollout" + str(scenarios[rollout_number-1]) + "rep" + str(reps)
    dict_folder = rollout_dicts + "rep" + str(reps)
    os.system(f"mkdir -p {rollout_folder}")
    os.system(f"mkdir -p {dict_folder}")
    
    os.system(f"python -m gns.train --data_path={example_folder} --model_path={model_path} --output_path={rollout_folder} " +
                      f"--model_file='model-{old_ts}.pt' --train_state_file='train_state-{old_ts}.pt' -ntraining_steps={total_steps}")
    os.system(f"python -m gns.train --mode='rollout' --data_path={example_folder} --model_path={model_path} --output_path={rollout_folder} " +
                  f"--model_file='model-{total_steps}.pt' --train_state_file='train_state-{total_steps}.pt'")
    os.system(f"python -m chem_data.chemgns --action='analyze' --rollout_data_path={rollout_folder} " +  
                  f"--material_properties={mat_prop_str} --gases={gases_str} " + 
                  f"--particle_chem={part_chem_str} --proc_data_path={dict_folder} --share_path={rollout_dicts}")
    old_ts = total_steps
    total_steps += train_steps
    reps -= 1