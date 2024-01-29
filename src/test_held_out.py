from __future__ import print_function
import numpy as np
import pandas as pd
import torch
import os
import utils
import TD3
import json
import time
from tensorboardX import SummaryWriter
from arguments import get_args_test
import matplotlib.pyplot as plt 
import seaborn as sns 
import gym
import sys
from tqdm import tqdm
import checkpoint as cp
from config import *
from utils import SubprocVecEnv, load_dict_from_file
from test_train_morphologies import test_morphologies, train_morphologies
from types import SimpleNamespace
# from stable_baselines3.common.vec_env import SubprocVecEnv # Commented because there is a problem with stable_baselines3 installation

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def test(args_test:dict, args:dict):
        
    # Set up directories ===========================================================
    exp_name = "EXP_%04d" % (args['expID'])
    
    exp_path = os.path.join(DATA_DIR, exp_name) # Experiement path
    rb_path = os.path.join(BUFFER_DIR, exp_name) # Replay buffer path
        
    # Retrieve MuJoCo XML files for testing ========================================
    envs_test_names = []
    args['graphs'] = dict()
    args['action_ids'] = dict()
    # existing envs
    if not args['custom_xml']:
        for morphology in args['morphologies']:
            envs_test_names += [
                name[:-4]
                for name in os.listdir(XML_DIR)
                if (".xml" in name) and (morphology in name) 
            ]
            
        envs_test_names = [env_name for env_name in envs_test_names if env_name in train_morphologies]+\
            [env_name for env_name in envs_test_names if env_name in test_morphologies]
            
        for name in envs_test_names:
            args['graphs'][name], args['action_ids'][name] = utils.getGraphStructure(
                os.path.join(XML_DIR, "{}.xml".format(name)),
                args['observation_graph_type'],
                return_action_ids=True
            )
    # custom envs
    else:
        if os.path.isfile(args['custom_xml']):
            assert ".xml" in os.path.basename(args['custom_xml']), "No XML file found."
            name = os.path.basename(args['custom_xml'])
            envs_test_names.append(name[:-4])  # truncate the .xml suffix
            args['graphs'][name[:-4]], args['action_ids'][name[:-4]] = utils.getGraphStructure(
                args['custom_xml'], args['observation_graph_type'],
                return_action_ids=True
            )
        elif os.path.isdir(args['custom_xml']):
            for name in os.listdir(args['custom_xml']):
                if ".xml" in name:
                    envs_test_names.append(name[:-4])
                    args['graphs'][name[:-4]], args['action_ids'][name[:-4]] = utils.getGraphStructure(
                        os.path.join(args['custom_xml'], name), args['observation_graph_type'],
                        return_action_ids=True
                    )

    envs_test_names.sort()
    print("#" * 50 + "\ntraining envs: {}\n".format(envs_test_names) + "#" * 50)
    
    # Set up testing env and policy ================================================
    args['limb_obs_size'], args['max_action'] = utils.registerEnvs(
        envs_test_names, args['max_episode_steps'], args['custom_xml'], use_restricted_obs=args['use_restricted_obs']
    )
    max_num_limbs = max([len(args['graphs'][env_name]) for env_name in envs_test_names])
    
    # create vectorized testing env
    obs_max_len = (
        max([len(args['graphs'][env_name]) for env_name in envs_test_names])
        * args['limb_obs_size']
    )
    
    # determine the maximum number of children in all the testing envs
    if args['max_children'] is None:
        args['max_children'] = utils.findMaxChildren(envs_test_names, args['graphs'])
    args['max_num_limbs'] = max_num_limbs
    
    # Because we test, we remove the training policies from this :
    envs_test_names = [env_name for env_name in envs_test_names if env_name in test_morphologies]
    num_envs_test = len(envs_test_names)
    print("#" * 50 + "\ntesting envs: {}\n".format(envs_test_names) + "#" * 50)
    
    envs_test = [
        utils.makeEnvWrapper(name, obs_max_len, args['seed']) for name in envs_test_names
    ]

    envs_test = SubprocVecEnv(envs_test)  # vectorized env
    
    # set random seeds
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    
    # setup agent policy
    policy = TD3.TD3(SimpleNamespace(**args))
    
    cp.load_model_only(exp_path=exp_path, policy = policy, model_name="model.pyth")
    policy.models2eval()
    
    # we do the test over multiple seeds
    
    test_rewards = {env_name:[] for env_name in envs_test_names}
    
    for num_test in tqdm(list(range(4))):
        
        collect_done = False
        
        obs_list = envs_test.reset()
        done_list = [False for i in range(num_envs_test)]
        episode_reward_list = [0 for i in range(num_envs_test)]
        episode_timesteps_list = [0 for i in range(num_envs_test)]
        
        # Play an episode for each morphology
        while not collect_done:
            
            # Select the action according to the policy
            action_list = []
            for i in range(num_envs_test):
                # dynamically change the graph structure of the modular policy
                policy.change_morphology(args['graphs'][envs_test_names[i]], 
                                            args['action_ids'][envs_test_names[i]])
                # remove 0 padding of obs before feeding into the policy (trick for vectorized env)
                obs = np.array(
                    obs_list[i][
                        : args['limb_obs_size'] * len(args['graphs'][envs_test_names[i]])
                    ]
                )
                policy_action = policy.select_action(obs)
        
                # add 0-padding to ensure that size is the same for all envs
                policy_action = np.append(
                    policy_action,
                    np.array([0 for i in range(max_num_limbs - policy_action.size)]),
                )
                action_list.append(policy_action)
                
                # perform action in the environment
            
            new_obs_list, reward_list, curr_done_list, _ = envs_test.step(action_list)
        
            # record if each env has ever been 'done'
            done_list = [done_list[i] or curr_done_list[i] for i in range(num_envs_test)]
            
            for i in range(num_envs_test):
                
                # Save the cumulative reward
                if not done_list[i]:
                    episode_reward_list[i] += reward_list[i]
                    
                # Max number of timesteps per episode, if more then stop it 
                if episode_timesteps_list[i] + 1 == args["max_episode_steps"]:
                    done_list[i] = True
                    
                episode_timesteps_list[i]+=1
                
            obs_list = new_obs_list
            collect_done = all(done_list)
            
        # Store the cumulative rewards of the environments in the test_rewards dictioanry
        for i in range(num_envs_test):
            test_rewards[envs_test_names[i]].append(episode_reward_list[i])
                
    df = pd.DataFrame.from_dict(test_rewards)
    df['expID'] = args['expID']
    
    print(df)
    print(df.drop(columns=['expID']).mean(axis=1))
    print(df.drop(columns=['expID']).mean(axis=1).mean())
    
    df.to_csv(path_or_buf=os.path.join(exp_path,"test_held_out.csv"), index=False)
     
    return 
    
if __name__ == "__main__":
    
    print('The device used is : ', device)
    
    # Get all the arguments from the experiment we want to test
    
    args_test = get_args_test()
    
    # Path to the TensorFlow event file    
    
    results_directory = f'./{args_test.results_directory}'
    experiment_folder_name = f'/EXP_{args_test.ExpID}'
    experiment_directory = results_directory+experiment_folder_name
    try:
        for filename in os.listdir(experiment_directory):
            if '.pyth' in filename: 
                event_file_name = f'/{filename}'
    except:
        print('It seems that the Id entered don\'t rely to a specific experiment')
    
    # Retrieve the arguments of the training procedure
    file_path = experiment_directory+'/args.txt'
    args = load_dict_from_file(file_path)
        
    test(args_test, args)