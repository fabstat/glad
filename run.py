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

raw_data_path = "./chem_data/processed_output_so2/"
topredict_path = "./chem_data/to_predict"
rollout_dicts = "./chem_data/proc_data"

npz_path = "./gns/data/"
model_path = "./gns/model/"
rollouts_path = "./gns/output/"

gases = ['SO2', 'H2SO4']
train_steps = 300

total_steps = train_steps
old_ts = total_steps
rollout_number = 0
for dir in os.listdir(raw_data_path):
    if dir == ".DS_Store":
        continue
    
    path = raw_data_path + dir
    os.system(f"python -m chem_data.chemgns --action='prepare' --raw_data_path={path}  --preped_data_path={npz_path} --gases={gases[0]} {gases[1]}")
    
    rollout_folder = rollouts_path + "rollout" + str(rollout_number)
    os.system(f"mkdir -p {rollout_folder}")
    rollout_number += 1
    
    if total_steps == train_steps:
        os.system(f"python -m gns.train --data_path={npz_path} --model_path={model_path} --output_path={rollout_folder} -ntraining_steps={total_steps}")
    else:
        os.system(f"python -m gns.train --data_path={npz_path} --model_path={model_path} --output_path={rollout_folder} " +
                  f"--model_file='model-{old_ts}.pt' --train_state_file='train_state-{old_ts}.pt' -ntraining_steps={total_steps}")
        
    os.system(f"python -m gns.train --mode='rollout' --data_path={npz_path} --model_path={model_path} --output_path={rollout_folder} " +
              f"--model_file='model-{total_steps}.pt' --train_state_file='train_state-{total_steps}.pt'")
    os.system(f"python -m chem_data.chemgns --action='analyze' --rollout_data_path={rollout_folder}  --proc_data_path={rollout_dicts} --gases={gases[0]} {gases[1]}")
    
    old_ts = total_steps
    total_steps += train_steps