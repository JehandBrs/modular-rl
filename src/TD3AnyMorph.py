# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from AnyMorphArchitecture import AnyMorphActor, AnyMorphCritic
from utils import MLPBase, MLPSimpleActor, MLPSimpleCritic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

action_space_dim = 8
observation_space_dim = 171

class Actor(nn.Module):
	def __init__(self, max_action, use_function=False, envs_train = None, envs_train_names=None, model = 'AnyMorph'):
		super(Actor, self).__init__()
		self.model = model

		if model == 'AnyMorph':
			self.actor = AnyMorphActor(use_function=use_function, envs_train =envs_train, envs_train_names=envs_train_names).to(device)
		else:
			self.actor = MLPSimpleActor(observation_space_dim, action_space_dim)
		self.max_action = max_action
		

	def forward(self, state, env_name):
		if self.model == 'AnyMorph':
			return self.actor(state, env_name)
		else:
			return self.actor(state)


class Critic(nn.Module):
	def __init__(self,use_function=False, envs_train =None, envs_train_names=None, model = 'AnyMorph'):
		super(Critic, self).__init__()
		self.model = model

		# Q1, Q2 architecture
		if model == 'AnyMorph':
			self.q1 = AnyMorphCritic(use_function=use_function, envs_train =envs_train, envs_train_names=envs_train_names).to(device)
			self.q2 = AnyMorphCritic(use_function=use_function, envs_train =envs_train, envs_train_names=envs_train_names).to(device)
		else:
			self.q1 = MLPSimpleCritic(observation_space_dim+action_space_dim, 1)
			self.q2 = MLPSimpleCritic(observation_space_dim+action_space_dim, 1)

	def forward(self, state, action, env_name):
		if self.model == 'AnyMorph':
			Q_1 = self.q1(state, action, env_name)
			Q_2 = self.q2(state, action, env_name)
			return Q_1, Q_2
		else:
			Q_1 = self.q1(torch.cat((state, action), 1))
			Q_2 = self.q2(torch.cat((state, action), 1))
			return Q_1, Q_2

	def Q1(self, state, action, env_name):
		if self.model == 'AnyMorph':
			return self.q1(state, action, env_name)
		else:
			return self.q1(torch.cat((state, action), 1))


class TD3(object):
	def __init__(
		self,
		discount=0.99,
		# tau=0.046,
  		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
  		grad_clipping_value = 0.1,
		max_action = 1.,
		lr = 1e-4,
  		use_function=False, 
    	envs_train =None, 
     	envs_train_names=None,
		model = 'AnyMorph'
	):

		self.actor = Actor(max_action, use_function=use_function, envs_train =envs_train, envs_train_names=envs_train_names, model = model).to(device)
		self.actor_target = Actor(max_action, use_function=use_function, envs_train =envs_train, envs_train_names=envs_train_names, model = model).to(device)
		self.actor_target.load_state_dict(self.actor.state_dict())
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

		self.critic = Critic(use_function=use_function, envs_train =envs_train, envs_train_names=envs_train_names, model = model).to(device)
		self.critic_target = Critic(use_function=use_function, envs_train =envs_train, envs_train_names=envs_train_names, model = model).to(device)
		self.critic_target.load_state_dict(self.critic.state_dict())
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
  
		pytorch_actor_params = sum(p.numel() for p in self.actor.parameters() if p.requires_grad)
		pytorch_critic_params = sum(p.numel() for p in self.critic.parameters() if p.requires_grad)

		print('Actor number of parameters : ', pytorch_actor_params)
		print('Critic number of parameters : ', pytorch_critic_params)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.grad_clipping_value = grad_clipping_value
		self.max_action = max_action
		self.total_it = 0

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()   
     
	def train_single(self, replay_buffer, iterations, env_name, batch_size=100):
		for it in range(iterations):
      		# sample replay buffer
			x, y, u, r, d = replay_buffer.sample(batch_size)
			state = torch.FloatTensor(x).to(device)
			next_state = torch.FloatTensor(y).to(device)
			action = torch.FloatTensor(u).to(device)
			reward = torch.FloatTensor(r).to(device)
			not_done = torch.FloatTensor(1 - d).to(device)
			# select action according to policy and add clipped noise
      
			with torch.no_grad():
				noise = torch.FloatTensor(u).data.normal_(0, self.policy_noise).to(device)
				noise = noise.clamp(-self.noise_clip, self.noise_clip)
				#print(noise.size())

				next_action = self.actor_target(next_state, env_name) + noise
				next_action = next_action.clamp(-self.max_action, self.max_action).to(device)
				#print(next_action.size())

				# Qtarget = reward + discount * min_i(Qi(next_state, pi(next_state)))
				target_Q1, target_Q2 = self.critic_target(next_state, next_action, env_name)
				target_Q = torch.min(target_Q1, target_Q2)
				target_Q = reward + (not_done * self.discount * target_Q)
    
			# get current Q estimates
			current_Q1, current_Q2 = self.critic(state, action, env_name)

			# compute critic loss
			critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
   
			# optimize the critic
			self.critic_optimizer.zero_grad()
			critic_loss.backward()
			if self.grad_clipping_value > 0:
				nn.utils.clip_grad_norm_(
					self.critic.parameters(), self.grad_clipping_value
				)
			self.critic_optimizer.step()

			# delayed policy updates
			if it % self.policy_freq == 0:
				# print('policy update')
				# compute actor loss
				actor_loss = -self.critic.Q1(state, self.actor(state, env_name), env_name).mean()

				# if it%10==0:
				# 	print('Actor loss : ', actor_loss.mean(), ' - Critic loss : ', critic_loss.mean())
    
				# optimize the actor
				self.actor_optimizer.zero_grad()
				actor_loss.backward()
				if self.grad_clipping_value > 0:
					nn.utils.clip_grad_norm_(
					self.actor.parameters(), self.grad_clipping_value
				)
				self.actor_optimizer.step()

				# update the frozen target models
				for param, target_param in zip(self.critic.parameters(),
												self.critic_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.actor.parameters(),
												self.actor_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self, fname):
		torch.save(self.actor.state_dict(), "%s_actor.pth" % fname)
		torch.save(self.critic.state_dict(), "%s_critic.pth" % fname)

	def load(self, fname):
		self.actor.load_state_dict(torch.load("%s_actor.pth" % fname))
		self.critic.load_state_dict(torch.load("%s_critic.pth" % fname))
