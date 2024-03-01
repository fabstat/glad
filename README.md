# Chem GNS
GNS of 2D Chemical Compositions

## Requirements

I exported my environment which will have way more stuff than one needs to run this code - like LP and TDA packages. If you just want the stuff you need for the GNS, focus on all the 'torch' requirements and 'pyg', make sure you have the version of these that match you CUDA toolkit, and make sure to have the CUDA toolkit that works with your GPU driver.

I am using a conda environment, but I installed most of the packages with pip.

The GNS I am using can be found here: https://github.com/geoelements/gns  - so check their 'requirements.txt' for a more efficient list.

## Make the data by running bash script in cloud_model_H2SO4 folder

## Process data by running cells of notebook in cloud_model_H2SO4 folder

## Make the necessary directories: data, model, output, figures

## Train

Read 'train.py' file to learn more about the arguments and options:

`python train.py --data_path="./data/" --model_path="./model/" --output_path="./output/" -ntraining_steps=300`

## Predict

`python train.py --mode="rollout" --data_path="./data/" --model_path="./model/" --output_path="./output/" --model_file="model-300.pt" --train_state_file="train_state-300.pt"`

## Make figures with code in notebook in cloud_model_H2SO4 folder

## Render *** needs to be adapted for higher dimensional datasets ***

`python render_rollout.py --output_mode="gif" --rollout_dir="./output/" --rollout_name="rollout_0"`

** Input data is processed in either 'ship_plume_data/LagrangianModel_2D.ipynp' or 'ship_plume_data/data_processing_adaptation.ipynb' (this one uses already processed LES data) **

The 'data' folder needs to have a 'metadata.json' and three npz files: 'train.npz', 'test.npz', 'valid.npz' - these are the input to the GNS as it is written. To update the input files, modify and *run* one of the aforementioned notebook files.
