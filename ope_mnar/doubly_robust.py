import numpy as np
import scipy
import time
from termcolor import colored
import torch
from torch import nn
import collections
from scipy.stats import norm
# RL environment
from gym.spaces import Tuple
from gym.vector.utils.spaces import batch_space
# visualization
import seaborn as sns
import matplotlib.pyplot as plt

from base import SimulationBase
from density import StateActionVisitationRatio, StateActionVisitationRatioSpline, StateActionVisitationRatioExpoLinear
from utils import SimpleReplayBuffer, DiscretePolicy, MLPModule
from batch_rl.dqn import QNetwork

class DRL(SimulationBase):
    """
    Kallus, Nathan, and Masatoshi Uehara. "Efficiently breaking the curse of horizon in off-policy 
    evaluation with double reinforcement learning." Operations Research (2022).
    """
    
    def __init__(self, env, n, horizon, eval_env=None, discount=0.9, device=None, seed=0):

        super().__init__(
            env=env, 
            n=n, 
            horizon=horizon, 
            discount=discount, 
            eval_env=eval_env
        )

        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.gamma = discount

    def estimate_omega(self, omega_estimator, target_policy, initial_state_sampler, estimate_omega_kwargs):
        if not hasattr(self, 'replay_buffer'):
            self.replay_buffer = SimpleReplayBuffer(trajs=self.masked_buffer, seed=self.seed)
        if not hasattr(self, 'target_policy'): 
            self.target_policy = target_policy
        self.initial_state_sampler = initial_state_sampler

        self.omega_estimator = omega_estimator
        self.omega_estimator.masked_buffer = self.masked_buffer
        self.omega_estimator.estimate_omega(
            target_policy=target_policy, 
            initial_state_sampler=initial_state_sampler, 
            **estimate_omega_kwargs
        )

    def estimate_Q(self, Q_estimator, target_policy, estimate_Q_kwargs):
        if not hasattr(self, 'replay_buffer'):
            self.replay_buffer = SimpleReplayBuffer(trajs=self.masked_buffer, seed=self.seed)
        if not hasattr(self, 'target_policy'): 
            self.target_policy = target_policy

        self.Q_estimator = Q_estimator
        self.Q_estimator.masked_buffer = self.masked_buffer
        self.Q_estimator._initial_obs = self._initial_obs
        self.Q_estimator.estimate_Q(
            target_policy=target_policy, 
            **estimate_Q_kwargs
        )

    def omega(self, states, actions):
        return self.omega_estimator.omega(states, actions)

    def Q(self, states, actions):
        return self.Q_estimator.Q(states, actions)

    def get_value(self):
        states = self.replay_buffer.states
        actions = self.replay_buffer.actions
        rewards = self.replay_buffer.rewards
        next_states = self.replay_buffer.next_states
        initial_states = self._initial_obs

        est_omega = self.omega(states, actions)
        est_omega = est_omega / np.mean(est_omega) # normalize

        est_Q = self.Q_estimator.Q(states, actions).squeeze()
        est_next_V = self.Q_estimator.V(next_states)
        bellman_error = rewards + self.gamma * est_next_V - est_Q
        Q_debias = est_omega * bellman_error / (1 - self.gamma)
        integrated_Q = np.mean(self.Q_estimator.V(initial_states))

        return np.mean(Q_debias) + integrated_Q

    def get_value_interval(self, alpha=0.05):
        states = self.replay_buffer.states
        actions = self.replay_buffer.actions
        rewards = self.replay_buffer.rewards
        next_states = self.replay_buffer.next_states
        initial_states = self._initial_obs

        est_omega = self.omega(states, actions)
        est_omega = est_omega / np.mean(est_omega) # normalize

        est_Q = self.Q_estimator.Q(states, actions).squeeze()
        est_next_V = self.Q_estimator.V(next_states)
        bellman_error = rewards + self.gamma * est_next_V - est_Q
        Q_debias = est_omega * bellman_error / (1 - self.gamma)
        integrated_Q = np.mean(self.Q_estimator.V(initial_states))
        V = np.mean(Q_debias) + integrated_Q
        V_int_sigma_sq = np.mean(Q_debias ** 2)
        std = (V_int_sigma_sq ** 0.5) / (len(Q_debias) ** 0.5)
        lower_bound = V - norm.ppf(1 - alpha / 2) * std
        upper_bound = V + norm.ppf(1 - alpha / 2) * std

        inference_summary = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'value': V,
            'std': std
        }
        return inference_summary