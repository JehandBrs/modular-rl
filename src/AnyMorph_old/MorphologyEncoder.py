import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import *
from NameToMorphologyMapping import name_to_morphology_sequence_mapping, build_name_to_morphology_sequence_mapping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

obs_scale = 1000.0
act_scale = 1000.0

class ActorMorphologyEncoder(nn.Module):
    """ The morphology encoder architecture """
    def __init__(self, token_embedding_size = 32, sinusoidal_embedding_size = 96, use_function=False, envs_train = None, envs_train_names = None):
        super().__init__()
        self.token_embedding_size = token_embedding_size
        self.sinusoidal_embedding_size = sinusoidal_embedding_size

        self.sinusoidal_embedding_frequencies = self.get_sinusoidal_embedding_freq().view(1, -1)
                
        self.morphology_tokens = MorphologyTokens(use_function=use_function, envs_train =envs_train, envs_train_names=envs_train_names)
        
        self.mapping_to_action_embedding_matrix = self.morphology_tokens.get_mapping_to_embedding_action_matrix()
        self.mapping_to_observation_embedding_matrix = self.morphology_tokens.get_mapping_to_embedding_observation_matrix()
      
        self.nb_sensors_per_limb = 19
        self.action_embedding = nn.Embedding(sum(self.morphology_tokens.num_limbs.values())-len(self.morphology_tokens.num_limbs.keys()) +1, self.token_embedding_size)
        self.observation_embedding = nn.Embedding(sum(self.morphology_tokens.num_limbs.values())*self.nb_sensors_per_limb +3, self.token_embedding_size)


    def forward(self, state_t, env_name):
            
        batch_size = state_t.size()[0]
        state_len = state_t.size()[1]                 
        
        cos = torch.cos(torch.matmul(obs_scale*state_t.view(batch_size, state_len, 1), self.sinusoidal_embedding_frequencies))
        sin = torch.sin(torch.matmul(obs_scale*state_t.view(batch_size, state_len, 1), self.sinusoidal_embedding_frequencies))
                
        X = torch.cat([
                torch.cat([self.observation_embedding(torch.tensor(self.mapping_to_observation_embedding_matrix[env_name]).to(device)).unsqueeze(0)]*batch_size, dim=0),
                cos, sin], dim=2)        

        morphology_action_tokens = torch.cat([self.action_embedding(torch.tensor(self.mapping_to_action_embedding_matrix[env_name]).to(device)).unsqueeze(0)]*batch_size, dim=0)
            
        return X, morphology_action_tokens

    def get_sinusoidal_embedding_freq(self, k1 = 1/10, kM = 1000):
        # alpha = np.power(kM/k1, 1/(self.sinusoidal_embedding_size/2-1))
        
        idx = torch.arange(0, self.sinusoidal_embedding_size, 2).to(device=device)
        div_term = torch.exp(idx * (-math.log(10000.0) / self.sinusoidal_embedding_size))
        
        return div_term
        
        # return torch.FloatTensor([k1*alpha**i for i in range(int(self.sinusoidal_embedding_size/2))]).to(device)


class CriticMorphologyEncoder(nn.Module):
    """ The morphology encoder architecture """
    def __init__(self, token_embedding_size = 32, sinusoidal_embedding_size = 96, use_function=False, envs_train=None, envs_train_names=None):
        super().__init__()
        self.token_embedding_size = token_embedding_size
        self.sinusoidal_embedding_size = sinusoidal_embedding_size

        self.sinusoidal_embedding_frequencies = self.get_sinusoidal_embedding_freq().view(1, -1)
                
        self.morphology_tokens = MorphologyTokens(use_function=use_function, envs_train = envs_train, envs_train_names=envs_train_names)
        
        self.mapping_to_action_embedding_matrix = self.morphology_tokens.get_mapping_to_embedding_action_matrix()
        self.mapping_to_observation_embedding_matrix = self.morphology_tokens.get_mapping_to_embedding_observation_matrix()
      
        self.nb_sensors_per_limb = 19
        self.action_embedding = nn.Embedding(sum(self.morphology_tokens.num_limbs.values())-len(self.morphology_tokens.num_limbs.keys()) +1, self.token_embedding_size)
        self.observation_embedding = nn.Embedding(sum(self.morphology_tokens.num_limbs.values())*self.nb_sensors_per_limb +3, self.token_embedding_size)


    def forward(self, state_t, action_t, env_name):
        

        batch_size = state_t.size()[0]
        state_len = state_t.size()[1]
        action_len = action_t.size()[1]
                
        cos_s = torch.cos(torch.matmul(obs_scale*state_t.view(batch_size, state_len, 1), self.sinusoidal_embedding_frequencies))
        sin_s = torch.sin(torch.matmul(obs_scale*state_t.view(batch_size, state_len, 1), self.sinusoidal_embedding_frequencies))
                
        X = torch.cat([
                torch.cat([self.observation_embedding(torch.tensor(self.mapping_to_observation_embedding_matrix[env_name]).to(device)).unsqueeze(0)]*batch_size, dim=0),
                cos_s, sin_s], dim=2)
        
        
        cos_a = torch.cos(torch.matmul(act_scale*action_t.view(batch_size, action_len, 1), self.sinusoidal_embedding_frequencies))
        sin_a = torch.sin(torch.matmul(act_scale*action_t.view(batch_size, action_len, 1), self.sinusoidal_embedding_frequencies))
        
        morphology_action_tokens = torch.cat([
                torch.cat([self.action_embedding(torch.tensor(self.mapping_to_action_embedding_matrix[env_name]).to(device)).unsqueeze(0)]*batch_size, dim=0),
                cos_a, sin_a], dim=2)
        
        return torch.cat([X, morphology_action_tokens], dim=1)

    def get_sinusoidal_embedding_freq(self, k1 = 1/10, kM = 1000):
        # alpha = np.power(kM/k1, 1/(self.sinusoidal_embedding_size/2-1))
        
        idx = torch.arange(0, self.sinusoidal_embedding_size, 2).to(device=device)
        div_term = torch.exp(idx * (-math.log(10000.0) / self.sinusoidal_embedding_size))
        
        return div_term
        
        # return torch.FloatTensor([k1*alpha**i for i in range(int(self.sinusoidal_embedding_size/2))]).to(device)
    

class MorphologyTokens():
    """ a Class to get the tokens matrixes for the morphology embeddings from name to indexes"""
    def __init__(self, name_to_morphology_sequence_mapping:dict = name_to_morphology_sequence_mapping, use_function=False, envs_train=None, envs_train_names=None):
        if use_function:
            self.mapping_to_embedding_matrix = build_name_to_morphology_sequence_mapping(envs_train, envs_train_names)
        else:
            self.mapping_to_embedding_matrix = name_to_morphology_sequence_mapping
            
        self.num_limbs = {
            'cheetah':7,
            'hopper':5,
            'humanoid':9,
            'walker':7
        }
        self.morphology_tokens_length = sum(self.num_limbs.values())
        self.mapping_to_action_embedding_matrix = {}
        self.mapping_to_observation_embedding_matrix = {}

        # Get the action mapping
        m = 0
        for env_base_name, n_limbs in self.num_limbs.items():
            for env_name, index_sensors in self.mapping_to_embedding_matrix.items():
                if env_base_name in env_name:
                    self.mapping_to_action_embedding_matrix[env_name] = [index_sensor + m for index_sensor in index_sensors]
            m += n_limbs-1
            
        # Get the observation mapping
        m = 0
        for env_base_name, n_limbs in self.num_limbs.items():
            for env_name, index_sensors in self.mapping_to_embedding_matrix.items():
                if env_base_name in env_name:
                    self.mapping_to_observation_embedding_matrix[env_name] = [m]+[index_sensor + 1 + m for index_sensor in index_sensors]
            m += n_limbs
            
        self.mapping_to_observation_embedding_matrix = {
            name_env : [19*limb_index + i for limb_index in limb_indexes for i in range(19)] for name_env, limb_indexes in self.mapping_to_observation_embedding_matrix.items()
        }
        
        # Add pendulum to it
        self.mapping_to_observation_embedding_matrix['pendulum'] = [532, 533, 534]
        self.mapping_to_action_embedding_matrix['pendulum'] = [24]

    def get_mapping_to_embedding_action_matrix(self, ):
        return self.mapping_to_action_embedding_matrix
    
    def get_mapping_to_embedding_observation_matrix(self, ):
        return self.mapping_to_observation_embedding_matrix
    