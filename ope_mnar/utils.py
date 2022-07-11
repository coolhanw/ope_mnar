import os
import numpy as np
import json, pickle, joblib
import math
import gc
from functools import reduce
from scipy.stats import norm
from scipy.optimize import minimize
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import gym
from gym.spaces import Box, Discrete
from gym.vector import VectorEnv
from gym.utils import seeding
import matplotlib.pyplot as plt

__all__ = [
    'sigmoid', 'constant_fn', 'normcdf', 'iden', 'MinMaxScaler', 'MLPModule',
    'SimEnv', 'VectorSimEnv', 'VectorSepsisEnv'
]

def sigmoid(x, beta=1):
    return 1 / (1 + np.exp(-beta * x))

def constant_fn(val):
    """Create a function that returns a constant.
    
    Args:
        val (float): the constant value of the returned function
    
    Returns:
        callable
    """

    def func(*args, **kwargs):
        return val

    return func

class normcdf():
    """Transform the state using normal CDF."""

    def __init__(self):
        self.scaled_low = 0.001
        self.scaled_high = 0.999

    def fit(self, S):
        return

    def transform(self, S):
        return norm.cdf(S)

    def inverse_transform(self, S):
        return norm.ppf(
            np.clip(S, a_min=self.scaled_low, a_max=self.scaled_high))

class iden():
    """Identity transformation."""

    def __init__(self):
        self.scaled_low = -np.inf
        self.scaled_high = np.inf

    def fit(self, S):
        return

    def transform(self, S):
        return S

    def inverse_transform(self, S):
        return S

class MinMaxScaler():
    """Transform the state onto [0,1] using min and max value."""

    def __init__(self, min_val=None, max_val=None):
        self.data_min_ = min_val
        self.data_max_ = max_val

    def fit(self, S):
        if self.data_min_ is None or np.min(self.data_min_) == -np.inf:
            self.data_min_ = np.nanmin(S, axis=0)
        if self.data_max_ is None or np.max(self.data_max_) == np.inf:
            self.data_max_ = np.nanmax(S, axis=0)

    def transform(self, S):
        return (S - self.data_min_) / (self.data_max_ - self.data_min_)

    def inverse_transform(self, S):
        return S * (self.data_max_ - self.data_min_) + self.data_min_

class ExpoTiltingClassifierMNAR():
    """Implement the semiparametric IPW method for nonignorable missingness.
    
    Reference: Shao, J., & Wang, L. (2016). Semiparametric inverse propensity weighting for nonignorable missing data.
    """

    def _create_expg_func(self, u, y, delta):
        """
        Args:
            u (np.ndarray): dimension (k,u_dim)
            y (np.ndarray): dimension (k,y_dim)
            delta (np.ndarray): dimension (k,1)
            bandwidth (np.ndarray): dimension (k,u_dim,u_dim)
            
        Returns:
            expg_hat (callable)
        """
        self.kernel = None  # initialize the kernel

        if len(u.shape) == 1:
            u = u.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(delta.shape) == 1:
            delta = delta.reshape(-1, 1)
        u_sample = u  # (k,u_dim)
        k = len(u)
        u_dim = u.shape[1]
        y_dim = y.shape[1]
        y = np.nan_to_num(y, nan=0)

        def expg_hat(u, gamma, bandwidth):
            """
            Args:
                u (np.ndarray): dimension (n,u_dim)
                gamma (np.ndarray): dimension (y_dim,)
                bandwidth (np.ndarray): dimension (n,u_dim,u_dim)
                
            Returns:
                est (np.ndarray): dimension (n,1)
            """
            if len(u.shape) == 1:
                u = u.reshape(-1, u_dim)
            size = len(u)
            # reuse kernel to avoid duplicated calculation in optimization process
            if self.kernel is None:
                # make sure bandwidth is a valid covariance matrix
                if bandwidth.shape == (size, ):
                    bandwidth = np.tile(np.expand_dims(np.eye(u_dim), axis=0),
                                        reps=(size, 1, 1)) * bandwidth.reshape(
                                            size, 1, 1)  # (k,u_dim,u_dim)
                elif bandwidth.shape == (u_dim, u_dim):
                    bandwidth = np.tile(np.expand_dims(np.eye(u_dim), axis=0),
                                        reps=(size, 1, 1))
                assert bandwidth.shape == (size, u_dim, u_dim)
                expo = np.zeros(shape=(size, k))
                for i in range(k):
                    u_dist = u - u_sample[i]  # (n,u_dim)
                    expo[:, i] = -0.5 * reduce(np.matmul, [
                        np.expand_dims(u_dist, axis=1),
                        np.linalg.inv(bandwidth),
                        np.expand_dims(u_dist, axis=2)
                    ]).reshape(size, )
                _ = gc.collect()
                kernel = np.exp(expo) / (
                    (2 * math.pi)**u_dim * np.linalg.det(bandwidth).reshape(
                        size, 1))**(1 / 2)  # (n,k)
                self.kernel = kernel
            gamma_y = np.clip(
                np.dot(y, gamma).reshape(-1, 1), -709.78,
                709.78)  # (k,1), use the bounds to avoid overflow
            est = np.sum(np.dot(self.kernel, 1 - delta), axis=1) / np.sum(
                np.dot(self.kernel, delta * np.exp(gamma_y)), axis=1)  # (n,1)
            return est

        return expg_hat

    def _create_estEq_func(self,
                           L,
                           z,
                           u,
                           y,
                           delta,
                           bandwidth,
                           expg_func,
                           aggregate='mean'):
        """
        Args:
            L (int): number of bins to discretize the instrument variable
            z (np.ndarray): dimension (k,1)
            u (np.ndarray): dimension (k,u_dim)
            y (np.ndarray): dimension (k,y_dim)
            delta (np.ndarray): dimension (k,1)
            bandwidth (np.ndarray): dimension (k,u_dim,u_dim)
            expg_func (callable): exp(g)
            aggregate (str): if 'mean', then aggregate by taking average

        Returns:
            estEq (callable)
        """
        if len(u.shape) == 1:
            u = u.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        y = np.nan_to_num(y, nan=0)
        delta = delta.squeeze()
        z = z.squeeze()

        def estEq(gamma):
            """
            Args:
                gamma (np.ndarray): dimension (y_dim,)
            """
            gamma_y = np.clip(np.dot(y, gamma), -709.78,
                              709.78)  # dimension (k,), clip to avoid overflow
            if self.kernel is None:
                expg_est = expg_func(u=u, gamma=gamma, bandwidth=bandwidth)
            else:
                expg_est = expg_func(u=u, gamma=gamma, bandwidth=None)
            pi = 1 / (1 + expg_est * np.exp(gamma_y))  # (k,)
            pi[delta == 0] = 1.
            assert delta.shape == pi.shape
            v = delta / pi - 1
            z_onehot = np.eye(L)[z - 1]
            z_onehot = z_onehot[:, :
                                -1]  # remove one dimension due to redundancy
            comp_mat = z_onehot * v.reshape(-1, 1)
            if aggregate == 'mean':
                return np.nanmean(comp_mat, axis=0)
            elif aggregate is None:
                return comp_mat

        return estEq

    def estimate_gamma(self,
                       L,
                       z,
                       u,
                       y,
                       delta,
                       bandwidth=None,
                       seed=None,
                       gamma_init=None,
                       bounds=None,
                       verbose=True):
        """Estimate gamma.
        
        Args:
            L (int): number of bins to discretize the instrument variable
            z (np.ndarray): dimension (k,1)
            u (np.ndarray): dimension (k,u_dim)
            y (np.ndarray): dimension (k,y_dim)
            delta (np.ndarray): dimension (k,1)
            bandwidth (np.ndarray): dimension (k,u_dim,u_dim)
            seed (int): random seed to general initial values
            gamma_init (int or np.ndarray): initial value of gamma, only used in simulation
            bounds (tuple): bounds for value search
            verbose (bool): If True, print intermediate results
            
        Returns:
            gamma_hat (int or np.ndarray)
        """
        if len(u.shape) == 1:
            u = u.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(delta.shape) == 1:
            delta = delta.reshape(-1, 1)
        if len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1):
            y_dim = 1
        else:
            y_dim = y.shape[1]
        u_dim = u.shape[1]
        assert L >= 1 + y_dim, "unidentifiable"
        if L == 1 + y_dim:
            self.expg_func = self._create_expg_func(u=u, y=y, delta=delta)
            self.estEq = self._create_estEq_func(L=L,
                                                 z=z,
                                                 u=u,
                                                 y=y,
                                                 delta=delta,
                                                 bandwidth=bandwidth,
                                                 aggregate='mean',
                                                 expg_func=self.expg_func)

            def estEq_sq(gamma):
                M_mean = self.estEq(gamma)
                return np.matmul(M_mean.T, M_mean)

            gamma_hat_list = []
            estEq_sq_list = []
            # try several initial values to avoid local optimum
            reps = 5
            if not gamma_init:
                gamma_init_list = np.random.normal(size=(reps, y_dim))
            else:
                gamma_init_list = np.tile(np.array(gamma_init).reshape((
                    1,
                    y_dim,
                )),
                                          reps=(reps, 1))
            for i in range(reps):
                gamma_init = gamma_init_list[i].reshape((y_dim, ))
                gamma_hat = minimize(fun=estEq_sq,
                                     x0=gamma_init,
                                     bounds=bounds,
                                     method='L-BFGS-B')
                gamma_hat_list.append(gamma_hat.x)
                estEq_sq_list.append(estEq_sq(gamma_hat.x))
                if verbose:
                    print(f'gamma_init: {gamma_init}')
                    print(f'estEq(gamma_init): {estEq_sq(gamma_init)}')
                    print(f'estEq({gamma_hat.x}): {estEq_sq(gamma_hat.x)}')
            gamma_hat = gamma_hat_list[np.argmin(estEq_sq_list)]
            expg_hat = self.expg_func(u=u, gamma=-0.5, bandwidth=bandwidth)
            gamma_y = np.clip(np.dot(y, (-0.5, )), -709.78, 709.78)  # (k,)
            pi_hat = 1 / (1 + expg_hat * np.exp(gamma_y))
            logit = np.log(expg_hat) + gamma_y
            expg_hat = self.expg_func(u=u,
                                      gamma=gamma_hat,
                                      bandwidth=bandwidth)
            gamma_y = np.clip(np.dot(y, gamma_hat), -709.78, 709.78)  # (k,)
            pi_hat = 1 / (1 + expg_hat * np.exp(gamma_y))
            logit = np.log(expg_hat) + gamma_y
            if verbose:
                print(
                    f'expg_hat (gamma={round(gamma_hat[0],3)})',
                    '0.0/0.25/0.5/0.75/1.0 quantile:{0:.2f}/{1:.2f}/{2:.2f}/{3:.2f}/{4:.2f}'
                    .format(np.nanmin(expg_hat),
                            np.nanquantile(expg_hat, 0.25),
                            np.nanquantile(expg_hat, 0.5),
                            np.nanquantile(expg_hat,
                                           0.75), np.nanmax(expg_hat)))
                print(
                    f'logit (gamma={round(gamma_hat[0],3)})',
                    '0.0/0.25/0.5/0.75/1.0 quantile:{0:.2f}/{1:.2f}/{2:.2f}/{3:.2f}/{4:.2f}'
                    .format(np.nanmin(logit), np.nanquantile(logit, 0.25),
                            np.nanquantile(logit, 0.5),
                            np.nanquantile(logit, 0.75), np.nanmax(logit)))
                print(
                    f'pi_hat (gamma={round(gamma_hat[0],3)})',
                    '0.0/0.25/0.5/0.75/1.0 quantile:{0:.2f}/{1:.2f}/{2:.2f}/{3:.2f}/{4:.2f}'
                    .format(np.nanmin(pi_hat), np.nanquantile(pi_hat, 0.25),
                            np.nanquantile(pi_hat, 0.5),
                            np.nanquantile(pi_hat, 0.75), np.nanmax(pi_hat)))
                print(f'estimating equation (gamma={round(gamma_hat[0],3)}):')
                print(self.estEq((gamma_hat[0], )))
            _ = gc.collect()
            return gamma_hat
        else:
            # generalized method of moments
            self.expg_func = self._create_expg_func(u=u, y=y, delta=delta)
            self.estEq_full = self._create_estEq_func(L=L,
                                                      z=z,
                                                      u=u,
                                                      y=y,
                                                      delta=delta,
                                                      bandwidth=bandwidth,
                                                      aggregate=None,
                                                      expg_func=self.expg_func)

            def step1_func(gamma):
                M_mean = np.mean(self.estEq_full(gamma), axis=0)
                _ = gc.collect()
                return np.matmul(M_mean.T, M_mean)

            gamma_hat_list = []
            estEq_sq_list = []
            # try several initial values to avoid local optimum
            reps = 5
            if not gamma_init:
                gamma_init_list = np.random.normal(size=(reps, y_dim))
            else:
                gamma_init_list = np.tile(np.array(gamma_init).reshape(
                    (1, y_dim)),
                                          reps=(reps, 1))
            for i in range(reps):
                gamma_init = gamma_init_list[i].reshape((y_dim, ))
                optresult1 = minimize(fun=step1_func,
                                      x0=gamma_init,
                                      method='L-BFGS-B',
                                      bounds=bounds)
                gamma_hat_step1 = optresult1.x
                gamma_hat_list.append(gamma_hat_step1)
                estEq_sq_list.append(step1_func(gamma_hat_step1))
                if verbose:
                    print(f'step1, gamma_init: {gamma_init}')
                    print(
                        f'step1, estEq(gamma_init): {step1_func(gamma_init)}')
                    print(
                        f'step1, estEq({gamma_hat_step1}): {step1_func(gamma_hat_step1)}'
                    )
            gamma_hat_step1 = gamma_hat_list[np.argmin(estEq_sq_list)]
            M = self.estEq_full(gamma_hat_step1)
            W_inv_hat = 1 / M.shape[0] * np.matmul(M.T, M)
            W_hat = np.linalg.inv(W_inv_hat)

            def step2_func(gamma):
                Q = reduce(np.matmul, [
                    np.mean(self.estEq_full(gamma), axis=0).reshape(1, -1),
                    W_hat,
                    np.mean(self.estEq_full(gamma), axis=0).reshape(-1, 1)
                ])
                _ = gc.collect()
                return Q.squeeze()

            gamma_init = gamma_hat_step1
            optresult2 = minimize(fun=step2_func,
                                  x0=gamma_init,
                                  bounds=bounds,
                                  method='L-BFGS-B')
            gamma_hat_step2 = optresult2.x
            if verbose:
                print(f'step2, gamma_init: {gamma_init}')
                print(f'step2, estEq(gamma_init): {step2_func(gamma_init)}')
                print(
                    f'step2, estEq({gamma_hat_step2}): {step2_func(gamma_hat_step2)}'
                )
            expg_hat = self.expg_func(u=u, gamma=-0.5, bandwidth=bandwidth)
            gamma_y = np.clip(np.dot(y, (-0.5, )), -709.78, 709.78)  # (k,)
            logit = np.log(expg_hat) + gamma_y
            pi_hat = 1 / (1 + expg_hat * np.exp(gamma_y))
            expg_hat = self.expg_func(u=u,
                                      gamma=gamma_hat_step2,
                                      bandwidth=bandwidth)
            gamma_y = np.clip(np.dot(y, gamma_hat_step2), -709.78,
                              709.78)  # (k,)
            logit = np.log(expg_hat) + gamma_y
            pi_hat = 1 / (1 + expg_hat * np.exp(gamma_y))
            if verbose:
                print(
                    f'expg_hat (gamma={round(gamma_hat_step2[0],3)})',
                    '0.0/0.25/0.5/0.75/1.0 quantile:{0:.2f}/{1:.2f}/{2:.2f}/{3:.2f}/{4:.2f}'
                    .format(np.nanmin(expg_hat),
                            np.nanquantile(expg_hat, 0.25),
                            np.nanquantile(expg_hat, 0.5),
                            np.nanquantile(expg_hat,
                                           0.75), np.nanmax(expg_hat)))
                print(
                    f'logit (gamma={round(gamma_hat_step2[0],3)})',
                    '0.0/0.25/0.5/0.75/1.0 quantile:{0:.2f}/{1:.2f}/{2:.2f}/{3:.2f}/{4:.2f}'
                    .format(np.nanmin(logit), np.nanquantile(logit, 0.25),
                            np.nanquantile(logit, 0.5),
                            np.nanquantile(logit, 0.75), np.nanmax(logit)))
                print(
                    f'pi_hat (gamma={round(gamma_hat_step2[0],3)})',
                    '0.0/0.25/0.5/0.75/1.0 quantile:{0:.2f}/{1:.2f}/{2:.2f}/{3:.2f}/{4:.2f}'
                    .format(np.nanmin(pi_hat), np.nanquantile(pi_hat, 0.25),
                            np.nanquantile(pi_hat, 0.5),
                            np.nanquantile(pi_hat, 0.75), np.nanmax(pi_hat)))
                print(
                    f'estimating equation (gamma={round(gamma_hat_step2[0],3)}):'
                )
                print(np.mean(self.estEq_full((gamma_hat_step2[0], )), axis=0))
            _ = gc.collect()
            return gamma_hat_step2

    def fit(self,
            L,
            z,
            u,
            y,
            delta,
            bandwidth=None,
            seed=None,
            gamma_init=None,
            bounds=None,
            verbose=True):
        """A wrapper function of estimate_gamma()
        
        Args:
            L (int): number of bins to discretize the instrument variable
            z (np.ndarray): dimension (k,1)
            u (np.ndarray): dimension (k,u_dim)
            y (np.ndarray): dimension (k,y_dim)
            delta (np.ndarray): dimension (k,1)
            bandwidth (np.ndarray): dimension (k,u_dim,u_dim)
            seed (int): random seed to general initial values
            gamma_init (int or np.ndarray): initial value of gamma, only used in simulation
            bounds (tuple): bounds for value search
            verbose (bool): If True, print intermediate results
            
        Returns:

        """
        self.L = L
        if len(u.shape) == 1:
            u = u.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(delta.shape) == 1:
            delta = delta.reshape(-1, 1)
        if len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1):
            y_dim = 1
        else:
            y_dim = y.shape[1]
        u_dim = u.shape[1]
        if bandwidth is None:  # default
            self.bandwidth_dict = {}
            bandwidth = np.zeros(shape=(len(z), u_dim, u_dim))
            bandwidth_factor = 7.5  # needs to be tuned for each scenario!
            for i in range(1, L + 1):
                if u_dim == 1:
                    bandwidth[z == i] = np.square(bandwidth_factor *
                                                  np.std(u[z == i], ddof=1) *
                                                  (np.sum(z == i)**(-1 / 3)))
                    self.bandwidth_dict[i] = np.square(
                        bandwidth_factor * np.std(u[z == i], ddof=1) *
                        (np.sum(z == i)**(-1 / 3)))
                else:
                    bandwidth[z == i] = np.square(bandwidth_factor) * np.cov(
                        u[z == i], rowvar=False) * np.square(
                            np.sum(z == i)**(-1 / 3))
                    self.bandwidth_dict[i] = bandwidth[
                        z == i] = np.square(bandwidth_factor) * np.cov(
                            u[z == i], rowvar=False) * np.square(
                                np.sum(z == i)**(-1 / 3))
        self.gamma_hat = self.estimate_gamma(L=L,
                                             z=z,
                                             u=u,
                                             y=y,
                                             delta=delta,
                                             bandwidth=bandwidth,
                                             seed=seed,
                                             gamma_init=gamma_init,
                                             bounds=bounds,
                                             verbose=verbose)

    def predict_proba(self, u, z, y):
        """Estimate pi (the probability of being observed)
        
        Args:
            u (np.ndarray): dimension (k,u_dim)
            z (np.ndarray): dimension (k,1)
            y (np.ndarray): dimension (k,y_dim)
            
        Returns:
            pi_est (np.ndarray)
        """
        if len(u.shape) == 1:
            u = u.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        u_dim = u.shape[1]
        assert hasattr(self, 'expg_func'), 'please run function fit() first'
        # reset self.kernel, it will be re-calculated in function self.expg_func()
        self.kernel = None
        bandwidth = np.zeros(shape=(len(z), u_dim, u_dim))
        for i in range(1, self.L + 1):
            assert hasattr(self, 'bandwidth_dict')
            bandwidth[z == i] = self.bandwidth_dict[i]
        expg_hat = self.expg_func(u=u,
                                  gamma=self.gamma_hat,
                                  bandwidth=bandwidth)  # (k,)
        gamma_y = np.clip(np.dot(y, self.gamma_hat), -709.78, 709.78)  # (k,)
        pi_est = 1 / (1 + expg_hat * np.exp(gamma_y))
        return pi_est

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({'L': self.L, 'gamma_hat': self.gamma_hat}, f)

    def load(self, filename):
        with open(filename, 'wb') as f:
            log_dict = pickle.load(f)
            self.L = log_dict.get('L', None)
            self.gamma_hat = log_dict.get('gamma_hat', None)

class SimEnv(gym.Env):

    def __init__(self,
                 state_trans_model,
                 reward_model,
                 dropout_model=None,
                 T=50,
                 dim=2,
                 num_actions=2,
                 low=-np.inf,
                 high=np.inf,
                 dtype=np.float32):
        """
        Args:
            state_trans_model (callable): return next state
            reward_model (callable): return reward
            dropout_model (callable): return dropout probability
            T (int): horizon
            dim (int): dimension of state variables
            num_actions (int): number of different actions
            low (float): lower bound of state variables
            high (float): upper bound of state variables
            dtype (data-type): data type of state variables
        """
        self.low = low
        self.high = high
        self.observation_space = Box(low=low,
                                     high=high,
                                     shape=(dim, ),
                                     dtype=dtype)
        self.action_space = Discrete(n=num_actions)
        self.last_obs = None
        self.dim = dim
        self.T = T
        self.count = 0
        self.vectorized = False
        self._np_random = np.random

        assert callable(state_trans_model)
        self.state_trans_model = state_trans_model
        assert callable(reward_model)
        self.reward_model = reward_model
        if dropout_model:
            assert callable(dropout_model)
        else:
            dropout_model = constant_fn(val=0) # no dropout
        self.dropout_model = dropout_model

        self.instrument_var_index = 1
        self.noninstrument_var_index = 0

        self.seed()

    def reset(self, S_init=None):
        """
        Args:
            S_init (np.ndarray): initial state

        Returns:
            self.last_obs (np.ndarray): initial state
        """
        self.count = 0
        self.states_history = []
        self.actions_history = []
        self.rewards_history = []
        self.survival_prob = 1
        self.next_survival_prob = 1

        if S_init is None:
            self.last_obs = self.observation_space.sample()
        else:
            self.last_obs = S_init
        self.last_obs = np.clip(a=self.last_obs,
                                a_min=self.low,
                                a_max=self.high)
        self.states_history.append(self.last_obs)
        
        return self.last_obs

    def seed(self, seed=None):
        """
        Args:
            seed (int): seed of the action_space and observation_space

        Returns:
            a list of seed
        """
        self._np_random, seed = seeding.np_random(seed)
        self.action_space._np_random, seed = seeding.np_random(seed)
        self.observation_space._np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        Args:
            action (int): the action taken

        Returns:
            S_next (sample from observation_space): next state
		    reward (np.ndarray): reward received
		    done (np.ndarray): indicator of whether the episode has ended
		    env_infos (dict): auxiliary diagnostic information
        """
        action = int(action)
        self.actions_history.append(action)
        self.count += 1

        S_next = self.state_trans_model(obs=self.last_obs,
                                        action=action,
                                        rng=self._np_random)
        S_next = np.clip(S_next, a_min=self.low, a_max=self.high)
        reward = self.reward_model(obs=self.last_obs,
                                    action=action,
                                    next_obs=S_next,
                                    rng=self._np_random)
        self.states_history.append(S_next)
        self.rewards_history.append(reward)

        # dropout probability
        dropout_prob = self.dropout_model(
            obs_history=np.array(self.states_history),
            action_history=np.array(self.actions_history),
            reward_history=np.array(self.rewards_history))
        self.survival_prob = self.next_survival_prob
        next_survival_prob = self.next_survival_prob * (1 - dropout_prob)
        self.next_survival_prob = next_survival_prob

        dropout_next = 1 * (np.random.uniform(low=0, high=1) >
                            1 - dropout_prob)
        if dropout_next:
            self.count = 0

        if self.count >= self.T:
            done = True
            self.count = 0
        else:
            done = False

        env_infos = {
            'next_survival_prob':
            next_survival_prob,  # probability of observing the next step
            'dropout': dropout_next,  # dropout indicator of the next step 
            'dropout_prob':
            dropout_prob  # dropout probability of the current step
        }
        
        self.last_obs = S_next.tolist()

        return S_next, reward, done, env_infos

class VectorSimEnv(VectorEnv):

    def __init__(self,
                 num_envs,
                 vec_state_trans_model,
                 vec_reward_model,
                 vec_dropout_model=None,
                 T=50,
                 dim=2,
                 num_actions=2,
                 low=-np.inf,
                 high=np.inf,
                 dtype=np.float32):
        """
        Args:
            num_envs (int): number of environments in parallel
            vec_state_trans_model (callable): return next states
            vec_reward_model (callable): return rewards
            vec_dropout_model (callable): return dropout probabilities
            T (int): horizon
            dim (int): dimension of state variables
            num_actions (int): number of different actions
            low (float): lower bound of state variables
            high (float): upper bound of state variables
            dtype (data-type): data type of state variables
        """
        action_space = Discrete(n=num_actions)
        observation_space = Box(low=low, high=high, shape=(dim, ), dtype=dtype)
        super().__init__(num_envs, observation_space, action_space)

        self.vectorized = True
        self.observations = None
        self.dim = dim
        self.T = T  # max length of trajectory
        self._np_random = np.random
        self.last_obs = None
        self.low = low
        self.high = high

        assert callable(vec_state_trans_model)
        self.state_trans_model = vec_state_trans_model
        assert callable(vec_reward_model)
        self.reward_model = vec_reward_model
        if vec_dropout_model:
            assert callable(vec_dropout_model)
        else:
            vec_dropout_model = constant_fn(val=0)
        self.dropout_model = vec_dropout_model

        self.instrument_var_index = 1
        self.noninstrument_var_index = 0

        self.seed()

    def reset_async(self, S_inits=None):
        self.count = 0
        self.actions_history = None  # (num_env,T)
        self.rewards_history = None  # (num_env,T)
        self.survival_prob = np.ones(shape=(self.num_envs, 1),
                                     dtype=np.float32)
        self.next_survival_prob = np.ones(shape=(self.num_envs, 1),
                                          dtype=np.float32)
        self.dropout_next = np.zeros(shape=(self.num_envs, 1), dtype=np.int8)
        self.state_mask = np.ones(shape=(self.num_envs, 1), dtype=np.int8)
        self.next_state_mask = np.ones(shape=(self.num_envs, 1), dtype=np.int8)

        if S_inits is not None:
            assert len(
                S_inits
            ) == self.num_envs, "The length of S_inits should be the same as num_envs"
        else:
            S_inits = self.observation_space.sample()

        S_inits = np.clip(a=S_inits, a_min=self.low, a_max=self.high)

        self.observations = S_inits  # (num_env,dim)
        self.states_history = np.expand_dims(a=S_inits,
                                             axis=1)  # (num_env,1,dim)
        self.states_history_mask = self.state_mask  # (num_env,1)

        self.last_obs = self.observations

    def reset_wait(self, timeout=None):
        """
        Args:
		    timeout (int or float, optional): the number of seconds before the call to reset_wait times out. If
			None, the call to reset_wait never times out.
		
        Returns:
            observations (sample from observation_space): a batch of observations from the vectorized environment.
		"""
        return self.observations

    def reset(self, S_inits=None):
        """Reset all sub-environments and return a batch of initial observations.
        
        Returns:
            observations (sample from observation_space): a batch of observations from the vectorized environment.
        """
        self.reset_async(S_inits)
        return self.reset_wait()

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    @property
    def np_random(self):
        """Lazily seed the rng since this is expensive and only needed if sampling from this space.
        """
        if self._np_random is None:
            self.seed()

        return self._np_random

    def step_async(self, actions):
        """
        Args: 
            actions (iterable of samples from action_space): list of actions taken in each environment
		"""
        self.count += 1
        actions_arr = np.array(actions).reshape(-1, 1)
        if self.actions_history is None:
            self.actions_history = actions_arr
        else:
            self.actions_history = np.concatenate(
                [self.actions_history, actions_arr], axis=1)
        S_next = self.state_trans_model(obs=self.observations,
                                        action=actions_arr,
                                        rng=self._np_random)  # (num_envs,dim)
        # clip state
        S_next = np.clip(a=S_next, a_min=self.low, a_max=self.high)
        if len(S_next.shape) == 1:
            S_next = S_next.reshape(1, -1)
        reward = self.reward_model(obs=self.observations,
                                   action=actions_arr,
                                   next_obs=S_next,
                                   rng=self._np_random)  # (num_envs,1)
        if self.count < self.T:
            self.states_history = np.concatenate(
                [self.states_history,
                 np.expand_dims(a=S_next, axis=1)],
                axis=1)
        if self.rewards_history is None:
            self.rewards_history = reward
        else:
            self.rewards_history = np.concatenate(
                [self.rewards_history, reward], axis=1)

        if self.rewards_history.shape[1] < 2:
            self.dropout_prob = self.dropout_model(
                obs_history=self.states_history,
                action_history=self.actions_history,
                reward_history=np.concatenate(
                    [np.zeros(shape=(self.num_envs, 1)), reward],
                    axis=1))  # (num_envs,1)
        else:
            self.dropout_prob = self.dropout_model(
                obs_history=self.states_history,
                action_history=self.actions_history,
                reward_history=self.rewards_history)  # (num_envs,1)

        self.survival_prob = self.next_survival_prob
        self.next_survival_prob *= 1 - self.dropout_prob
        self.dropout_next = (self.dropout_next == 1) * 1 + (
            self.dropout_next < 1) * (self.np_random.uniform(
                low=0, high=1, size=(len(self.dropout_prob), 1)) <
                                      self.dropout_prob)  # (num_envs,1)
        self.next_state_mask = np.minimum(
            self.state_mask,
            1 - self.dropout_next)  # (num_envs,1), element-wise minimum
        if self.count < self.T:
            self.states_history_mask = np.concatenate(
                [self.states_history_mask, self.next_state_mask],
                axis=1)  # (num_envs,1)
        self.state_mask = self.next_state_mask
        self.observations = S_next
        self.last_obs = self.observations

    def step_wait(self, timeout=None):
        """
		Args:
		    timeout (int or float, optional): number of seconds before the call to step_wait times out. If
			None, the call to step_wait never times out.
		
		Returns:
		    observations (sample from observation_space): a batch of observations from the vectorized environment.
		    rewards (np.ndarray): a vector of rewards from the vectorized environment.
		    dones (np.ndarray): a vector whose entries indicate whether the episode has ended.
		    infos (list of dict): a list of auxiliary diagnostic information.
		"""
        if self.count >= self.T:
            dones = np.array([[True]] * self.num_envs,
                             dtype=np.bool_)  # (num_envs,1)
            self.count = 0
        else:
            dones = np.array([[False]] * self.num_envs,
                             dtype=np.bool_)  # (num_envs,1)

        env_infos = {
            'next_survival_prob': self.next_survival_prob.copy(
            ),  # probability of observing the next step
            'dropout': self.dropout_next.astype(
                np.int8).copy(),  # dropout indicator of the next step 
            'dropout_prob': self.dropout_prob.copy(
            ),  # dropout probability of the current step
            'state_mask':
            self.state_mask.copy()  # 1 indicates observed, 0 otherwise
        }

        return self.observations.copy(), self.rewards_history[:, -1].reshape(
            -1, 1).copy(), dones.copy(
            ), env_infos  # create a copy to aviod mutating values

    def close_extras(self, **kwargs):
        r"""Clean up the extra resources e.g. beyond what's in this base class. """
        pass

class VectorSimSynthEnv(VectorEnv):
    """Vectorized environment with learned model, assume no dropout."""

    def __init__(
            self,
            num_envs,
            T=20,
            env_config_file=None,
            dtype=np.float32):
        """
        Args:
            num_envs (int): number of environments in parallel
            T (int): horizon
            dim (int): dimension of state variables
            env_config_file (str): path to the environment configuration json file
            dtype (data-type): data type of state variables
        """

        module_path = os.path.dirname(__file__)
        if not env_config_file:
            env_config_file = os.path.join(module_path, 'lm_config.json')
        with open(env_config_file) as json_file:
            config = json.load(json_file)
        self.dim = config['S_dim']
        self.num_actions = config['num_actions']
        self.low = np.array(config['low']).astype(dtype)
        self.high = np.array(config['high']).astype(dtype)

        action_space = Discrete(n=self.num_actions)
        observation_space = Box(low=self.low,
                                high=self.high,
                                shape=(self.dim, ),
                                dtype=dtype)
        super().__init__(num_envs, observation_space, action_space)

        state_model_path = config['state_model']
        try:
            with open(os.path.join(module_path, state_model_path),
                      'rb') as file:
                self.state_model = pickle.load(file)
        except:
            self.state_model = joblib.load(
                os.path.join(module_path, state_model_path))
        self.state_scaler = self.state_model.get('scaler', iden())
        self._static_index = self.state_model.get('static_state_index', [])
        self._dynamic_index = self.state_model.get('dynamic_state_index',
                                                   list(range(self.dim)))
        reward_model_path = config['reward_model']
        try:
            with open(os.path.join(module_path, reward_model_path),
                      'rb') as file:
                self.reward_model = pickle.load(file)
        except:
            self.reward_model = joblib.load(
                os.path.join(module_path, reward_model_path))
        self.reward_scaler = self.reward_model.get('scaler', iden())
        for a in range(self.num_actions):
            assert a in self.state_model.keys()
            assert a in self.reward_model.keys()

        self.observations = None
        self.T = T  # max length of trajectory
        self.vectorized = True

        self.seed()

    def reset_async(self, S_inits=None):
        self.count = 0
        self.actions_history = None  # (num_env,T)
        self.rewards_history = None  # (num_env,T)

        if S_inits is not None:
            assert len(
                S_inits
            ) == self.num_envs, "The length of S_inits should be the same as num_envs"
        else:
            S_inits = self.observation_space.sample()
        self.observations = S_inits  # (num_env,dim)
        self.states_history = np.expand_dims(a=S_inits,
                                             axis=1)  # (num_env,1,dim)

    def reset_wait(self, timeout=None):
        """
		Args:
		    timeout (int or float, optional): number of seconds before the call to reset_wait times out. If
			    None, the call to reset_wait never times out.
		
        Returns:
		    observations (sample from observation_space): a batch of observations from the vectorized environment.
		"""
        return self.observations

    def reset(self, S_inits=None):
        """Reset all sub-environments and return a batch of initial observations.
        
        Args:
            S_inits (sample from observation_space): initial states. If None, randomly sample from observation_space.
        
        Returns:
            observations (sample from observation_space): a batch of observations from the vectorized environment.
        """
        self.reset_async(S_inits)
        return self.reset_wait()

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    @property
    def np_random(self):
        """Lazily seed the rng since this is expensive and only needed if sampling from this space.
        """
        if self._np_random is None:
            self.seed()

        return self._np_random

    def step_async(self, actions):
        """
		Args:
		    actions (iterable of samples from action_space): list of actions.
		"""
        self.count += 1
        actions_arr = np.array(actions).reshape(-1)
        if self.actions_history is None:
            self.actions_history = actions_arr.reshape(-1, 1)
        else:
            self.actions_history = np.concatenate(
                [self.actions_history,
                 actions_arr.reshape(-1, 1)], axis=1)
        S_next = np.zeros(shape=self.observations.shape)
        S_next[:, self._static_index] = self.observations[:,
                                                          self._static_index]
        rewards = np.zeros(shape=self.num_envs)
        for a in range(self.num_actions):
            if any(actions_arr == a) == False:
                continue
            S_a = self.state_scaler.transform(
                self.observations[actions_arr == a])
            dynamic_state_pred = self.state_scaler.inverse_transform(
                self.state_model[a].predict(X=S_a))  # on the original scale

            S_next[actions_arr == a,
                   len(self._static_index):] = dynamic_state_pred

            S_next = np.clip(S_next, a_min=self.low, a_max=self.high)

            R_a = self.reward_scaler.transform(
                np.hstack([self.observations, S_next])[actions_arr == a])
            rewards[actions_arr == a] = self.reward_model[a].predict(
                X=R_a).squeeze()  # (num_envs,)

        if len(S_next.shape) == 1:
            S_next = S_next.reshape(1, -1)
        rewards = rewards.reshape(-1, 1)  # (num_envs,1)

        self.states_history = np.concatenate(
            [self.states_history,
             np.expand_dims(a=S_next, axis=1)], axis=1)
        if self.rewards_history is None:
            self.rewards_history = rewards
        else:
            self.rewards_history = np.concatenate(
                [self.rewards_history, rewards], axis=1)
        self.observations = S_next

    def step_wait(self, timeout=None):
        """
		Args:
		    timeout (int or float, optional): number of seconds before the call to step_wait times out. 
                If None, the call to step_wait never times out.
		
		Returns:
            observations (sample from observation_space): a batch of observations from the vectorized environment.
            rewards (np.ndarray): a vector of rewards from the vectorized environment.
            dones (np.ndarray): a vector whose entries indicate whether the episode has ended.
            infos (list of dict): a list of auxiliary diagnostic information.
		"""
        if self.count >= self.T:
            dones = np.array([[True]] * self.num_envs,
                             dtype=np.bool_)  # (num_envs,1)
            self.count = 0
        else:
            dones = np.array([[False]] * self.num_envs,
                             dtype=np.bool_)  # (num_envs,1)
        env_infos = {}
        return self.observations.copy(), self.rewards_history[:, -1].reshape(
            -1, 1).copy(), dones.copy(
            ), env_infos  # create a copy to aviod mutating values

    def close_extras(self, **kwargs):
        r"""Clean up the extra resources e.g. beyond what's in this base class. """
        pass

class VectorSepsisEnv(VectorEnv):
    def __init__(self,
                 num_envs,
                 T=20,
                 static_state_list=[],
                 dynamic_state_list=[],
                 action_levels=None,
                 vec_state_trans_model=None,
                 vec_reward_model=None,
                 vec_dropout_model=None,
                 low=-np.inf,
                 high=np.inf,
                 dtype=np.float32):
        
        action_space = Discrete(n=action_levels)
        state_list = static_state_list + dynamic_state_list
        state_dim = len(state_list)
        observation_space = Box(low=low, high=high, shape=(state_dim, ), dtype=dtype)
        super().__init__(num_envs, observation_space, action_space)

        self.vectorized = True
        self.observations = None
        self.dim = state_dim
        self.T = T  # max length of trajectory
        self._np_random = np.random
        self.last_obs = None
        self.low = low
        self.high = high
        if vec_state_trans_model is not None:
            assert callable(vec_state_trans_model)
            self.state_trans_model = vec_state_trans_model
        else:
            self.state_trans_model = None
        if vec_reward_model:
            assert callable(vec_reward_model)
            self.reward_model = vec_reward_model
        else:
            self.reward_model = None
        if vec_dropout_model is not None:
            assert callable(vec_dropout_model)
        else:
            vec_dropout_model = constant_fn(val=0)
        self.dropout_model = vec_dropout_model
        self.seed()

    def reset_async(self, S_inits=None):
        pass

    def reset_wait(self, timeout=None):
        """
		Parameters
		----------
		timeout : int or float, optional
			Number of seconds before the call to `reset_wait` times out. If
			`None`, the call to `reset_wait` never times out.
		Returns
		-------
		observations : sample from `observation_space`
			A batch of observations from the vectorized environment.
		"""
        return self.observations

    def reset(self, S_inits=None):
        r"""Reset all sub-environments and return a batch of initial observations.
        
        Returns
        -------
        observations : sample from `observation_space`
            A batch of observations from the vectorized environment.
        """
        self.reset_async(S_inits)
        return self.reset_wait()

    def seed(self, seed=None):
        self._np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return [seed]

    @property
    def np_random(self):
        """Lazily seed the rng since this is expensive and only needed if
        sampling from this space.
        """
        if self._np_random is None:
            self.seed()
        return self._np_random

    def step_async(self, actions):
        """
		Parameters
		----------
		actions : iterable of samples from `action_space` 
			List of actions.
		"""
        pass

    def step_wait(self, timeout=None):
        """
		Parameters
		----------
		timeout : int or float, optional
			Number of seconds before the call to `step_wait` times out. If
			`None`, the call to `step_wait` never times out.
		
		Returns
		-------
		observations : sample from `observation_space`
			A batch of observations from the vectorized environment.
		rewards : `np.ndarray` instance (dtype `np.float_`)
			A vector of rewards from the vectorized environment.
		dones : `np.ndarray` instance (dtype `np.bool_`)
			A vector whose entries indicate whether the episode has ended.
		infos : list of dict
			A list of auxiliary diagnostic information.
		"""
        pass

    def close_extras(self, **kwargs):
        r"""Clean up the extra resources e.g. beyond what's in this base class. """
        pass


class MLPModule(pl.LightningModule):
    """Multilayer perceptron module."""

    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_sizes=[64, 64],
            hidden_nonlinearity=torch.relu,
            hidden_w_init=nn.init.xavier_normal_,
            hidden_b_init=nn.init.zeros_,
            output_nonlinearities=None,
            output_bias=True,
            output_w_inits=nn.init.xavier_normal_,
            output_b_inits=nn.init.zeros_,
            batch_normalization=False,
            lr=1e-3,
            loss=F.mse_loss):
        super().__init__()

        self._hidden_sizes = hidden_sizes
        if self._hidden_sizes is not None:
            self._layers = nn.ModuleList()
            prev_size = input_dim
            for size in self._hidden_sizes:
                hidden_layers = nn.Sequential()
                linear_layer = nn.Linear(prev_size, size)
                hidden_w_init(linear_layer.weight)
                hidden_b_init(linear_layer.bias)
                hidden_layers.add_module('linear', linear_layer)
                if batch_normalization:
                    hidden_layers.add_module('batch_normalization',
                                             nn.BatchNorm1d(size))
                if hidden_nonlinearity:
                    hidden_layers.add_module('non_linearity',
                                             hidden_nonlinearity)
                self._layers.append(hidden_layers)
                prev_size = size
        else:
            prev_size = input_dim

        linear_layer = nn.Linear(prev_size, output_dim, bias=output_bias)
        output_w_inits(linear_layer.weight)
        if output_bias:
            output_b_inits(linear_layer.bias)
        self._layers.add_module('linear', linear_layer)
        if output_nonlinearities:
            self._layers.add_module('non_linearity', output_nonlinearities)

        self.lr = lr
        self.loss = loss

    def forward(self, x):
        if self._hidden_sizes is not None:
            for layer in self._layers:
                x = layer(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(input=y_hat, target=y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(input=y_hat, target=y)
        self.log('valid_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(input=y_hat, target=y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict(self, X):
        self.eval()
        dtype_numpy = isinstance(X, np.ndarray)
        if dtype_numpy:
            X = torch.from_numpy(X)
        y_pred = self.forward(X)
        self.train()
        if dtype_numpy:
            return y_pred.detach().numpy()
        else:
            return y_pred

# the following functions will be useful for policy optimization

def get_next_block_idx(current_block_idx, K_n, K_T):
    n, t = current_block_idx
    if n < K_n: return [n + 1, t]
    else:
        if t < K_T:
            return [1, t + 1]
        else:
            return None

def get_idx_pos(current_block_idx, n, T, n_min, T_min):
    """given the current block index return the corresponding position"""
    K_n = n // n_min
    K_T = T // T_min
    k_n, k_T = current_block_idx
    if k_n < K_n:
        if k_T < K_T:
            return (k_n - 1) * n_min, (k_n) * n_min, T_min
        else:
            return (k_n - 1) * n_min, (k_n) * n_min, T_min + T - K_T * T_min
    else:
        if k_T < K_T:
            return (k_n - 1) * n_min, n, T_min
        else:
            return (k_n - 1) * n_min, n, T_min + T - K_T * T_min
