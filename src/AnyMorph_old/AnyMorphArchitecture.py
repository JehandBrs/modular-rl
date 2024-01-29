import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import *
from MorphologyEncoder import ActorMorphologyEncoder, CriticMorphologyEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AnyMorphActor(nn.Module):
    def __init__(self, token_embedding_size = 32, sinusoidal_embedding_size = 96, use_function=False, envs_train =None, envs_train_names=None):
        super().__init__()
        self.token_embedding_size = token_embedding_size
        self.sinusoidal_embedding_size = sinusoidal_embedding_size
        self.d_model = 128
        self.n_head = 2
        
        # All the actions and observations embeddings
        
        self.morphology_encoder = ActorMorphologyEncoder(use_function=use_function, envs_train =envs_train, envs_train_names=envs_train_names).to(device)
                
        self.linear_observation_embedding_transformation = nn.Linear(
            in_features = self.token_embedding_size+self.sinusoidal_embedding_size, 
            out_features = self.d_model, 
            device=device)
        
        self.linear_action_embedding_transformation = nn.Linear(
            in_features = self.token_embedding_size, 
            out_features = self.d_model,
            device=device)
        
        # All the Transformer architecture
        self.transformer_layer = nn.Transformer(
            d_model=self.d_model, 
            nhead=self.n_head, 
            num_encoder_layers=3, 
            batch_first=True,
            num_decoder_layers=3, 
            dim_feedforward=256, 
            dropout=0., 
            activation="relu", 
            device=device,)
        
        self.action_layer = nn.Linear(
            in_features = self.d_model,
            out_features = 1,
            device=device
        )
        
        self.tanh = nn.Tanh()
        
    def forward(self, state_t, env_name):
        X, morphology_action_tokens = self.morphology_encoder.forward(state_t, env_name)
        
        obs_embedded = self.linear_observation_embedding_transformation(X)
        act_embedded = self.linear_action_embedding_transformation(morphology_action_tokens)
                
        transformers_output = self.transformer_layer(
            obs_embedded, 
            act_embedded
            )
                
        y_t = self.action_layer(transformers_output)
        
        return self.tanh(y_t).squeeze(2)
    
class AnyMorphCritic(nn.Module):
    def __init__(self, token_embedding_size = 32, sinusoidal_embedding_size = 96, use_function=False, envs_train =None, envs_train_names=None):
        super().__init__()
        self.token_embedding_size = token_embedding_size
        self.sinusoidal_embedding_size = sinusoidal_embedding_size
        self.d_model = 128
        self.n_head = 2
        
        # All the actions and observations embeddings
        
        self.morphology_encoder = CriticMorphologyEncoder(use_function=use_function, envs_train =envs_train, envs_train_names=envs_train_names)
        self.q_encoder = nn.Embedding(1, self.d_model)
                
        self.linear_observation_embedding_transformation = nn.Linear(
            in_features = self.token_embedding_size+self.sinusoidal_embedding_size, 
            out_features = self.d_model, 
            device=device)
        
        self.linear_action_embedding_transformation = nn.Linear(
            in_features = self.token_embedding_size, 
            out_features = self.d_model,
            device=device)
        
        # All the Transformer architecture
        self.transformer_layer = nn.Transformer(
            d_model=self.d_model, 
            nhead=self.n_head, 
            num_encoder_layers=3, 
            batch_first=True,
            num_decoder_layers=3, 
            dim_feedforward=256, 
            dropout=0., 
            activation="relu", 
            device=device,)
        
        self.Q_layer = nn.Linear(
            in_features = self.d_model,
            out_features = 1,
            device=device
        )
                
    def forward(self, state_t, action_t, env_name):
        
        X = self.morphology_encoder(state_t, action_t, env_name)
        
        critic_embedded = self.linear_observation_embedding_transformation(X)
        
        transformers_output = self.transformer_layer(
            critic_embedded, 
            self.q_encoder(torch.LongTensor([[0] for i in range(critic_embedded.size()[0])]).to(device))
            )
        
        Q = self.Q_layer(transformers_output)
        
        return Q.squeeze(2)
        