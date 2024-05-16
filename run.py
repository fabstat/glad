from glob import glob
from pathlib import Path
import os
import re


#### Set these as appropriate:
# PartMC-MOSAIC Data:
raw_data_path = "./chem_data/processed_output_some/"
topredict_path = "./chem_data/to_predict/"
rollout_dicts = "./chem_data/proc_data/"

npz_path = "./gns/data/"
model_path = "./gns/model/"
rollouts_path = "./gns/output/"

material_properties = ['aero_number', 'BC', 'OC']
mat_prop_str = "','".join(material_properties)
particle_chem = ['H2O', 'SO4']
part_chem_str = "','".join(particle_chem)
gases = ['H2SO4']
gases_str = "','".join(gases)
train_steps = 300
available_scenarios = ['0001_simple_cond', '0002_simple_cond', '0003_simple_cond', '0005_simple_cond', 
             '0006_simple_cond', '0007_simple_cond','0008_simple_cond','0009_simple_cond', '0012_simple_cond']
scenarios = [1]
example_to_test = 1 # set to be a number from the scenarios list
total_reps = 20




total_steps = train_steps
old_ts = total_steps
example_number = 0
rollout_number = 0

for scenario in scenarios:
    if rollout_number < len(scenarios):
        # if dir.startswith("."):
        #     print(f"Skipping {dir}")
        #     continue

        path = raw_data_path + available_scenarios[scenario]
        print(path)
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
            
  
        if rollout_number == len(scenarios) - 1:
            rollex = example_to_test
            example_folder = npz_path + "example" + str(rollex) + "/"
            rollout_folder = rollouts_path + "rollout" + str(rollex)
            dict_folder = rollout_dicts + "ex" + str(rollex)
            total_steps = train_steps * (1 + rollex)
            os.system(f"python -m gns.train --mode='rollout' --data_path={example_folder} --model_path={model_path} --output_path={rollout_folder} " +
                  f"--model_file='model-{total_steps}.pt' --train_state_file='train_state-{total_steps}.pt'")
            os.system(f"python -m chem_data.chemgns --action='analyze' --rollout_data_path={rollout_folder} " +  
                      f"--material_properties={mat_prop_str} --gases={gases_str} " + 
                      f"--particle_chem={part_chem_str} --proc_data_path={dict_folder} --share_path={rollout_dicts}")
        else:
            os.system(f"python -m gns.train --mode='rollout' --data_path={example_folder} --model_path={model_path} --output_path={rollout_folder} " +
                  f"--model_file='model-{total_steps}.pt' --train_state_file='train_state-{total_steps}.pt'")
            
        

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