from absl import flags
from absl import app
import pickle
import matplotlib.pyplot as plt
from chem_data.prepare_data import *
from chem_data.analyze_results import *


flags.DEFINE_enum('action', None, ['prepare', 'analyze', 'predict'],
    help='Prepare raw data for training or analyze rollout results.')

flags.DEFINE_string('raw_data_path', 'chem_data/raw_data/', help='The raw dataset directory.')
flags.DEFINE_string('preped_data_path', 'gns/data/', help='The path for saving the prepared data for training.')
flags.DEFINE_string('rollout_data_path', 'gns/output/', help='The rollout dataset directory.')
flags.DEFINE_string('proc_data_path', 'chem_data/proc_data/', help='The path for saving the prepared data for training.')
flags.DEFINE_string('share_path', 'chem_data/proc_data/', help='The path for saving shared pre/post data.')

flags.DEFINE_list('material_properties', ['BC', 'OC', 'aero_number'], help='List of material properties.')
flags.DEFINE_list('particle_chem', ['H2O', 'SO4'], help='List of particle phase chemicals.')
flags.DEFINE_list('gases', ['H2SO4'], help='List of gas phase chemicals.')

FLAGS = flags.FLAGS



def main(_):
    myflags = {}
    myflags["raw_data_path"] = FLAGS.raw_data_path
    myflags["preped_data_path"] = FLAGS.preped_data_path
    myflags["rollout_data_path"] = FLAGS.rollout_data_path
    myflags["proc_data_path"] = FLAGS.proc_data_path
    myflags["share_path"] = FLAGS.share_path
    myflags["material_properties"] = FLAGS.material_properties
    myflags["particle_chem"] = FLAGS.particle_chem
    myflags["gases"] = FLAGS.gases
        
    if FLAGS.action in ['prepare', 'predict']:
        feats_dict = load_raw_data(myflags["raw_data_path"])

        # material properties don't change over time
        # shape[0] must equal the number of particles
        mat_prop = []
        for prop in myflags["material_properties"]:
            mat_prop += [feats_dict[prop][0]]
        MP = np.vstack(mat_prop)
        MP = MP.transpose()

        # these make up the dimensions of the gns
        time_changing_features = []
        ptype = [] # gns algorithm supports different type of particles
        for i, chem in enumerate(myflags["particle_chem"] + myflags["gases"]):
            if i < len(myflags["particle_chem"]):
                ptype += [np.array([1]*feats_dict[chem].shape[1])]
                time_changing_features += [feats_dict[chem]]
            else:
                ptype += [np.array([2]*feats_dict[chem].shape[1])]
                time_changing_features += [np.log(feats_dict[chem])]
            # else:
            #     time_changing_features += [np.log(feats_dict[chem])]
        X = np.stack(time_changing_features, axis=-1)
        X = X[1:,:,:] # data for time step 0 is too different
        ptype = np.concatenate(ptype)
        
        ##### notes/changes: ##############################
        # put just gas in log scale
        # look into if all the particles stay positive
        # convert them back to their true values
        # process splitting - gas info is on top particles, not ideal
        # better to have a gas gns
        # the mass of h2so4 is balanced between the particles
        # d m s04 / dt non normalized info on what happened to the gas
        # total change of of the mass of so4 per volume of air occupied per time
        # feed the above information into the gas h2so4
        # however much the particles are going that is how the h2so4 is being depleated
        # side by side comparisons of time series
        # dig through the mosaic paper - they solve analytically
        # look up first order loss/reactions
        ####################################################

        # normalize values to be in the 0-1 interval
        norm_X, min_x, max_x = normalize(X)
        norm_MP, min_mp, max_mp = normalize(MP)
        
        unnorm = [min_x, max_x, min_mp, max_mp]
        filename = os.path.join(myflags["share_path"], f'unnorm.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(unnorm, f)
        

        if FLAGS.action == 'prepare':
            split_dict, idxs, train_cutoff, test_cutoff = data_splits(norm_X, ptype, norm_MP, traincut=0.6, testcut=1.0)
            train_pre = np.array(split_dict["train_data"], dtype="object")
            test_pre = np.array(split_dict["test_data"], dtype="object")
            np.savez(os.path.join(myflags["preped_data_path"], "train.npz"), x=train_pre)
            np.savez(os.path.join(myflags["preped_data_path"], "test.npz"), x=test_pre)

            if "val_data" in split_dict:
                val_pre = np.array(split_dict["val_data"], dtype="object")
                np.savez(os.path.join(myflags["preped_data_path"], "valid.npz"), x=val_pre)

            make_metadata_file(myflags["preped_data_path"], split_dict["train_data"])
        else:
            pred_pre = np.array([norm_X, ptype, norm_MP], dtype="object")
            np.savez(os.path.join(myflags["preped_data_path"], "predict.npz"), x=pred_pre)
    
    elif FLAGS.action == 'analyze':
        rollout_dict = load_rollout_data(myflags["rollout_data_path"])
        # each rollout keys: ['initial_positions', 'predicted_rollout', 'ground_truth_rollout',
        # 'particle_types', 'material_property', 'metadata', 'loss'])
        
        
        filename = os.path.join(myflags["share_path"], 'unnorm.pkl')
        with open(filename, 'rb') as f: 
            unnorm = pickle.load(f) 

        name_i = 0
        for rollout_name in rollout_dict:
            ro = rollout_dict[rollout_name]
            true_x = ro['ground_truth_rollout']*(unnorm[1] - unnorm[0]) + unnorm[0]
            pred_x = ro['predicted_rollout']*(unnorm[1] - unnorm[0]) + unnorm[0]
            mat_prop = ro['material_property']*(unnorm[3] - unnorm[2]) + unnorm[2]
           
            reshaped_mat_prop = np.stack([np.tile(mat_prop[:,i], (true_x[:,:,0].shape[0],1)) 
                                          for i in range(mat_prop.shape[1])], axis=-1)
            outdata_dict = {}
            outdata_dict['loss'] = ro['loss']
            outdata_dict['true_x'] = {}
            outdata_dict['pred_x'] = {}
            outdata_dict['mat_prop'] = {}
            x_names = myflags["particle_chem"] + myflags["gases"]
            mp_names = myflags["material_properties"]
            for i in range(true_x.shape[-1]):
                if i < len(myflags["particle_chem"]): 
                    outdata_dict['true_x'][x_names[i]] = true_x[:,:,i]
                    outdata_dict['pred_x'][x_names[i]] = pred_x[:,:,i]
                else:
                    outdata_dict['true_x'][x_names[i]] = np.exp(true_x[:,:,i])
                    outdata_dict['pred_x'][x_names[i]] = np.exp(pred_x[:,:,i])
            
            for j in range(reshaped_mat_prop.shape[-1]):
                outdata_dict['mat_prop'][mp_names[j]] = reshaped_mat_prop[:,:,j]
                
            filename = os.path.join(myflags["proc_data_path"], f'{rollout_name[:-4]}{name_i}_dict.pkl')
            with open(filename, 'wb') as f:
                pickle.dump(outdata_dict, f)
            name_i += 1
                
            

if __name__ == '__main__':
      app.run(main)