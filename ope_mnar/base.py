import numpy as np
import copy
import pandas as pd
import os
import gc
import joblib
import time
from collections import Counter
import pathlib
# RL environment
from gym.spaces import Tuple
from gym.vector.utils.spaces import batch_space
# ML model
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# visualization
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    iden,
    MinMaxScaler,
    MLPModule,
    ExpoTiltingClassifierMNAR
)


class SimulationBase(object):

    def __init__(self, env=None, n=500, horizon=None, discount=0.8, eval_env=None):
        """
        Args:
            env (gym.Env): dynamic environment
            n (int): the number of subjects (trajectories)
            horizon (int): the maximum length of trajectories
            discount (float): discount factor
            eval_env (gym.Env): dynamic environment to evaluate the policy, if not specified, use env
        """
        assert env is not None or eval_env is not None, "please provide env or eval_env"
        self.env = env
        self.vectorized_env = env.vectorized if env is not None else True
        if eval_env is None and env is not None:
            self.eval_env = copy.deepcopy(env)
        else:
            self.eval_env = eval_env
        self.n = n
        self.max_T = self.env.T if horizon is None else horizon # maximum horizon
        self.gamma = discount
        self.obs_policy = lambda S: self.env.action_space.sample() if self.env is not None else None  # uniform sample

        if self.env is not None:
            if self.vectorized_env:
                self.env.num_envs = n
                self.env.observation_space = batch_space(
                    self.env.single_observation_space, n=self.env.num_envs)
                self.env.action_space = Tuple(
                    (self.env.single_action_space, ) * self.env.num_envs)
                self.num_actions = self.env.single_action_space.n # the number of candidate discrete actions
                self.state_dim = self.env.single_observation_space.shape[0]
                # store the last observation which is particularly designed for append block to make 
                # sure that the append block's first state can match the last state in current buffer
                self.last_obs = self.env.observation_space.sample()
            else:
                self.num_actions = self.env.action_space.n # the number of candidate discrete actions
                self.state_dim = self.env.observation_space.shape[0]
                # store the last observation which is particularly designed for append block to make 
                # sure that the append block's first state can match the last state in current buffer
                self.last_obs = np.vstack([
                    self.env.observation_space.sample() for _ in range(self.n)
                ])

        self.low = np.array([np.inf] * self.state_dim) # initial values, will be updated when generating trajectories
        self.high = np.array([-np.inf] * self.state_dim) # initial values, will be updated when generating trajectories
        self.masked_buffer = {} # masked data buffer, only observed data are included
        self.full_buffer = {}
        self.misc_buffer = {}  # hold any other information
        self.concat_trajs = None

    def sample_initial_states(self, size, from_data=False, seed=None):
        assert self.env is not None
        np.random.seed(seed)
        if not from_data:
            if self.vectorized_env:
                old_size = self.env.num_envs
                # reset size
                self.env.num_envs = size
                self.env.observation_space = batch_space(
                    self.env.single_observation_space, n=self.env.num_envs)
                self.env.action_space = Tuple(
                    (self.env.single_action_space, ) * self.env.num_envs)
                S_inits = self.env.observation_space.sample()
                # recover size
                self.env.num_envs = old_size
                self.env.observation_space = batch_space(
                    self.env.single_observation_space, n=self.env.num_envs)
                self.env.action_space = Tuple(
                    (self.env.single_action_space, ) * self.env.num_envs)
            else:
                S_inits = np.vstack([self.env.reset() for _ in range(size)])
            return S_inits
        else:
            assert hasattr(self, '_initial_obs'), "please generate data or import trajectories first"
            selected_id = np.random.choice(a=self.n, size=size)
            return self._initial_obs[selected_id]

    def sample_states(self, size, seed=None):
        assert hasattr(self, '_obs'), "please generate data or import trajectories first"
        np.random.seed(seed)
        selected_id = np.random.choice(a=len(self._obs), size=size)
        return self._obs[selected_id]

    def gen_single_traj(self,
                 policy=None,
                 S_init=None,
                 S_init_kwargs={},
                 A_init=None,
                 burn_in=0,
                 evaluation=False,
                 seed=None):
        """
        Generate a single trajectory. 

        Args:
            policy (callable): the policy to generate the trajectory
            S_init (np.ndarray): initial states
            S_init_kwargs (dict): other kwargs input to env.reset()
            A_init (np.ndarray): initial action, dimension should equal to the number of candidate actions
            burn_in (int): length of burn-in period
            evaluation (bool): if True, use eval_env to generate trajectory
            seed (int): random seed for env

        Returns:
            observations_traj (np.ndarray)
            action_traj (np.ndarray)
            reward_traj (np.ndarray)
            T (int): length of the trajectory
            survival_prob_traj (np.ndarray)
            dropout_next_traj (np.ndarray)
            dropout_prob_traj (np.ndarray)

        TODO: organize the output into a dictionary
        """
        if policy is None:
            policy = self.obs_policy if self.obs_policy is not None else lambda S: self.env.action_space.sample(
            )
        if evaluation:
            env = self.eval_env
            max_T = env.T
        else:
            env = self.env
            max_T = self.max_T
        # initialize the state
        if seed is not None:
            env.seed(seed)
        if S_init is not None or not S_init_kwargs:
            S = env.reset(S_init, **S_init_kwargs)
        else:
            S = env.reset()
        S_traj = [S]
        A_traj = []
        reward_traj = []
        survival_prob_traj = []
        dropout_next_traj, dropout_prob_traj = [], []
        step = 1
        # while step <= env.T:
        while step <= max_T:
            self.low = np.minimum(self.low, np.array(S))
            self.high = np.maximum(self.high, np.array(S))
            if len(S_traj) == 1 and A_init is not None:
                A = A_init
            else:
                A = policy(S)
            if hasattr(A, '__iter__'):
                A = np.array(A).squeeze()
                assert len(
                    A
                ) == self.num_actions, f"len(A):{len(A)}, num_actions: {self.num_actions}, output of policy should match the number of actions"
                A = np.random.choice(range(self.num_actions), p=A).item()
            S_next, reward, done, env_infos = env.step(A)
            S_traj.append(S_next)
            A_traj.append(A)
            reward_traj.append(reward)
            survival_prob = env_infos.get('next_survival_prob', 1)
            survival_prob_traj.append(survival_prob)
            dropout_next = env_infos.get('dropout', None)
            dropout_next_traj.append(dropout_next)
            dropout_prob = env_infos.get('dropout_prob', None)
            dropout_prob_traj.append(dropout_prob)
            S = S_next
            if dropout_next:
                break
            step += 1
        T = len(reward_traj)
        # convert to numpy.ndarray
        S_traj = np.array(S_traj)[:T]
        A_traj = np.array(A_traj)
        reward_traj = np.array(reward_traj)
        survival_prob_traj = np.array(survival_prob_traj).reshape(-1)
        dropout_next_traj = np.array(dropout_next_traj).reshape(-1)
        dropout_prob_traj = np.array(dropout_prob_traj).reshape(-1)
        if burn_in is None:
            return [
                S_traj, 
                A_traj, 
                reward_traj, 
                T, 
                survival_prob_traj,
                dropout_next_traj, 
                dropout_prob_traj
            ]
        else:
            if T > burn_in:
                return [
                    S_traj[burn_in:], 
                    A_traj[burn_in:], 
                    reward_traj[burn_in:],
                    T - burn_in, 
                    survival_prob_traj[burn_in:],
                    dropout_next_traj[burn_in:], 
                    dropout_prob_traj[burn_in:]
                ]
            else:
                return []

    def gen_batch_trajs(self,
                        policy=None,
                        S_inits=None,
                        A_inits=None,
                        burn_in=0,
                        evaluation=False,
                        seed=None):
        """
        Generate a batch of trajectories. 

        Args: 
            policy (callable): the policy to generate trajectories
            S_inits (np.ndarray): initial states
            A_inits (np.ndarray): initial actions
            burn_in (int): the lenght of burn-in period
            evaluation (bool): if True, use eval_env
            seed (int): seed (int): random seed for env

        Returns:
            observations_traj (np.ndarray)
            action_traj (np.ndarray)
            reward_traj (np.ndarray)
            T (int)
            survival_prob_traj (np.ndarray)
            dropout_next_traj (np.ndarray)
            dropout_prob_traj (np.ndarray)

        TODO: organize the output into a dictionary
        """
        if A_inits is not None and len(A_inits.shape) == 1:
            A_inits = A_inits.reshape(-1, 1)
        if policy is None:
            policy = self.obs_policy if self.obs_policy is not None else lambda S: self.env.action_space.sample()
        if evaluation:
            env = self.eval_env
            max_T = env.T
        else:
            env = self.env
            max_T = self.max_T
        if seed is not None:
            env.seed(seed)
        # initialize the states
        if S_inits is not None:
            S = env.reset(S_inits)
        else:
            S = env.reset()
        survival_prob_traj = []
        dropout_next_traj, dropout_prob_traj = [], []
        step = 1
        # while step <= env.T:
        while step <= max_T:
            self.low = np.minimum(self.low, np.min(S, axis=0))
            self.high = np.maximum(self.high, np.max(S, axis=0))
            if step == 1 and A_inits is not None:
                A = A_inits  # (num_envs,1)
            else:
                A = policy(S)  # (num_envs,num_actions)
                if isinstance(A, tuple):
                    A = np.expand_dims(np.array(A), axis=1)
            if A.shape[1] > 1:
                assert A.shape[
                    1] == self.num_actions, "output of policy should match the number of actions"
                # sample an action based on the probability
                A = (A.cumsum(axis=1) > np.random.rand(
                    A.shape[0])[:, None]).argmax(axis=1)
            S, _, _, env_infos = env.step(actions=A)
            survival_prob = env_infos.get('next_survival_prob',
                                          np.ones(shape=(len(S), 1)))
            survival_prob_traj.append(survival_prob)
            dropout_next = env_infos.get(
                'dropout', np.zeros(shape=(len(S), 1), dtype=np.int8))
            dropout_next_traj.append(dropout_next)
            dropout_prob = env_infos.get('dropout_prob',
                                         np.zeros(shape=(len(S), 1)))
            dropout_prob_traj.append(dropout_prob)
            step += 1
        S_traj = env.states_history  # (num_envs, T, dim)
        A_traj = env.actions_history  # (num_envs, T)
        reward_traj = env.rewards_history  # (num_envs, T)
        survival_prob_traj = np.concatenate(survival_prob_traj,
                                            axis=1)  # (num_envs, T)
        dropout_next_traj = np.concatenate(dropout_next_traj,
                                           axis=1)  # (num_envs, T)
        dropout_prob_traj = np.concatenate(dropout_prob_traj,
                                           axis=1)  # (num_envs, T)

        S_traj_mask = env.states_history_mask  # (num_envs, T)
        # T denotes the actual length of the observed trajectories
        T = np.argmin(S_traj_mask, axis=1)
        T[T == 0] = self.max_T
        # output state, action, reward trajectory and T
        if burn_in is None:
            return [
                S_traj, A_traj, reward_traj, T, survival_prob_traj,
                dropout_next_traj, dropout_prob_traj, S_traj_mask
            ]
        else:
            if any(T > burn_in):
                observed_index = T > burn_in
                return [
                    S_traj[observed_index, burn_in:, :], 
                    A_traj[observed_index, burn_in:],
                    reward_traj[observed_index,burn_in:], 
                    T[observed_index] - burn_in,
                    survival_prob_traj[observed_index, burn_in:],
                    dropout_next_traj[observed_index, burn_in:],
                    dropout_prob_traj[observed_index, burn_in:], 
                    S_traj_mask[observed_index, burn_in:]
                ]
            else:
                return []

    def gen_masked_buffer(self,
                          policy=None,
                          n=None,
                          total_N=None,
                          S_inits=None,
                          S_inits_kwargs={},
                          burn_in=0,
                          seed=None):
        """
        Generate masked (observed) buffer data.

        Args:
            policy (callable): the policy to generate trajectories
            n (int): number of trajectories
            total_N (int): total number of observed tuples
            S_init (np.ndarray): initial states
            S_init_kwargs (dict): additional kwargs passed to env.reset()
            burn_in (int): length of burn-in period
            seed (int): random seed for env
        """
        self.burn_in = burn_in
        if n is None:
            n = self.n
        if S_inits is not None:
            assert len(
                S_inits) == n, "the number of initial states should match n"
        if not self.vectorized_env:
            count = 0
            incomplete_cnt = 0
            if total_N is None:
                for i in range(n):
                    trajs = self.gen_single_traj(
                        policy=policy,
                        burn_in=burn_in,
                        S_init=S_inits[i] if S_inits is not None else None,
                        S_init_kwargs={
                            k: v[i]
                            for k, v in S_inits_kwargs.items()
                        } if S_inits_kwargs else {},
                        A_init=None,
                        evaluation=False)
                    if not trajs:
                        continue
                    self.masked_buffer[(i)] = trajs
                    if self.masked_buffer[(i)][3] < self.max_T:
                        incomplete_cnt += 1
                    count += self.masked_buffer[(i)][3]
            else:
                i = 0
                incomplete_cnt = 0
                while count < total_N:
                    trajs = self.gen_single_traj(
                        policy=policy,
                        burn_in=burn_in,
                        S_init=S_inits[i] if S_inits is not None else None,
                        S_init_kwargs={
                            k: v[i]
                            for k, v in S_inits_kwargs.items()
                        } if S_inits_kwargs else {},
                        A_init=None,
                        evaluation=False)
                    if not trajs:
                        continue
                    self.masked_buffer[(i)] = trajs
                    if self.masked_buffer[(i)][3] < self.max_T:
                        incomplete_cnt += 1
                    count += self.masked_buffer[(i)][3]
                    i += 1
                self.n = i
            self.total_N = count
        else:
            count = 0
            incomplete_cnt = 0
            if total_N is None:
                self.concat_trajs = self.gen_batch_trajs(policy=policy,
                                                         burn_in=burn_in,
                                                         S_inits=S_inits,
                                                         A_inits=None,
                                                         seed=seed,
                                                         evaluation=False)
                S_traj, A_traj, reward_traj, T, survival_prob_traj, dropout_next_traj, dropout_prob_traj, S_traj_mask = self.concat_trajs
                for i in range(len(S_traj)):
                    observed_state_index = np.where(S_traj_mask[i] == 1)[0]
                    observed_index = observed_state_index[
                        observed_state_index < self.max_T - burn_in].tolist()
                    observed_state_index = observed_state_index.tolist()
                    self.masked_buffer[(i)] = [
                        S_traj[i][observed_state_index],
                        # S_traj[i][observed_index],
                        A_traj[i][observed_index],
                        reward_traj[i][observed_index],
                        T[i],
                        survival_prob_traj[i][observed_index],
                        dropout_next_traj[i][observed_index],
                        dropout_prob_traj[i][observed_index]
                    ]
                    self.full_buffer[(i)] = [
                        S_traj[i], A_traj[i], reward_traj[i], T[i],
                        survival_prob_traj[i], dropout_next_traj[i],
                        dropout_prob_traj[i]
                    ]
                    if T[i] < self.max_T:
                        incomplete_cnt += 1
                    count += T[i]
            else:
                i = 0
                incomplete_cnt = 0
                while count < total_N:
                    self.concat_trajs = self.gen_batch_trajs(policy=policy,
                                                             burn_in=burn_in,
                                                             S_inits=None,
                                                             A_inits=None,
                                                             seed=seed,
                                                             evaluation=False)
                    S_traj, A_traj, reward_traj, T, survival_prob_traj, dropout_next_traj, dropout_prob_traj, S_traj_mask = self.concat_trajs
                    if count + np.sum(T) <= total_N:
                        sub_n = len(S_traj)
                    else:
                        sub_n = np.where(
                            T.cumsum() > total_N - count)[0].min() + 1
                    # for j in range(i, i + sub_n):
                    for j in range(sub_n):
                        # print(f'S_traj_mask.shape:{S_traj_mask.shape}')
                        observed_state_index = np.where(S_traj_mask[j] == 1)[0]
                        observed_index = observed_state_index[
                            observed_state_index < self.max_T -
                            burn_in].tolist()
                        observed_state_index = observed_state_index.tolist()
                        self.masked_buffer[(j + i)] = [
                            S_traj[j][observed_state_index],
                            # S_traj[j][observed_index],
                            A_traj[j][observed_index],
                            reward_traj[j][observed_index],
                            T[j],
                            survival_prob_traj[j][observed_index],
                            dropout_next_traj[j][observed_index],
                            dropout_prob_traj[j][observed_index]
                        ]
                        self.full_buffer[(j + i)] = [
                            S_traj[j], A_traj[j], reward_traj[j], T[j],
                            survival_prob_traj[j], dropout_next_traj[j],
                            dropout_prob_traj[j]
                        ]
                        if T[j] < self.max_T:
                            incomplete_cnt += 1
                        count += T[j]
                    i += sub_n
                self.n = i
            self.total_N = count
        self.dropout_rate = incomplete_cnt / self.n
        self.missing_rate = 1 - self.total_N / (self.n * self.max_T)
        initial_obs = []
        all_obs = []
        for k in self.masked_buffer.keys():
            initial_obs.append(self.masked_buffer[k][0][0])
            all_obs.append(self.masked_buffer[k][0])
        
        self._initial_obs = np.vstack(initial_obs)
        self._obs = np.vstack(all_obs)

    def export_buffer(self):
        """Convert unscaled trajectories in self.masked_buffer to a dataframe.
        """
        obs_dim = np.array(self.masked_buffer[next(iter(
            self.masked_buffer))][0]).shape[1]
        X_cols = [f'X{i}' for i in range(1, obs_dim + 1)]
        df = pd.DataFrame([])
        for k in self.masked_buffer.keys():
            nrows = len(self.masked_buffer[k][0])
            action_list = self.masked_buffer[k][1]
            if len(action_list) < nrows:
                nonterminal_state_list = self.masked_buffer[k][0][:-1]
            else:
                nonterminal_state_list = self.masked_buffer[k][0]
            tmp = pd.DataFrame(self.masked_buffer[k][0], columns=X_cols)
            tmp['id'] = np.repeat(k, nrows)
            if len(action_list) < nrows:
                tmp['action'] = np.append(action_list, [None])
                tmp['reward'] = np.append(self.masked_buffer[k][2], [None])
                tmp['surv_prob'] = np.append(self.masked_buffer[k][4], [None])
                tmp['dropout_prob'] = np.append(self.masked_buffer[k][6], [None])
            else:
                tmp['action'] = action_list
                tmp['reward'] = self.masked_buffer[k][2]
                tmp['surv_prob'] = self.masked_buffer[k][4]
                tmp['dropout_prob'] = self.masked_buffer[k][6]
            df = df.append(tmp)
        df = df.reset_index(drop=True)
        return df

    def import_buffer(self,
                      data=None,
                      data_filename=None,
                      static_state_cols=None,
                      dynamic_state_cols=None,
                      action_col='action',
                      reward_col='reward',
                      id_col='id',
                      dropout_col=None,
                      subsample_id=None,
                      reward_transform=None,
                      burn_in=0,
                      mnar_nextobs_var=None,
                      mnar_noninstrument_var=None,
                      mnar_instrument_var=None,
                      **kwargs):
        """Import unscaled trajectories from a dataframe to masked_buffer.

        Args:
            data (pd.DataFrame): dataframe to be imported
            data_filename (str): path to the table
            static_state_cols (list): column names of static features
            dynamic_state_cols (list): column names of dynamic features
            action_col (str): name of action column
            reward_col (str): name of reward column
            id_col (str): name of reward column
            dropout_col (str): name of the binary column that indicates dropout event
            subsample_id (list): ids of subsample to be imported
            reward_transform (callable): transformation funtion of reward
            burn_in (int): length of burn-in period
            mnar_nextobs_var (str): column name of the next observation for MNAR inference
            mnar_noninstrument_var (list): column names of noninstrument variable for MNAR inference
            mnar_instrument_var (str): column name of instrument variable for MNAR inference
            kwargs (dict): kwargs passed to pd.read_csv()        
        """
        assert data is not None or data_filename is not None, "please provide data or data_filename."
        if data is None:
            print(f'Import from {data_filename} to buffer')
            data = pd.read_csv(data_filename, **kwargs)

        state_cols = static_state_cols + dynamic_state_cols
        self.state_cols = state_cols
        self._df = data
        self.num_actions = data[action_col].nunique()
        self.state_dim = len(state_cols)
        self.last_obs = None
        self.burn_in = burn_in
        if subsample_id is None:
            id_list = data[id_col].unique().tolist()
        else:
            id_list = subsample_id
        num_trajs = len(id_list)
        self.masked_buffer = {} # observed data
        self.misc_buffer = {} # miscellaneous info
        if state_cols is None:
            state_cols = [c for c in data.columns if c.startswith('X')]
        incomplete_cnt = 0
        self.total_N = 0
        self._initial_obs = []
        if 'custom_dropout_prob' not in data.columns and dropout_col in data.columns:
            print(f'use column {dropout_col} as dropout indicator')
        for i, id_ in enumerate(id_list):
            tmp = data.loc[data[id_col] == id_]
            S_traj = tmp[state_cols].values
            self._initial_obs.append(S_traj[0])
            A_traj = tmp[action_col].values
            reward_traj = tmp[reward_col].values
            if reward_transform is not None:
                reward_traj = reward_transform(reward_traj)
            T = len(A_traj)
            if 'custom_dropout_prob' not in tmp.columns:
                if 'surv_prob' in tmp.columns:
                    survival_prob_traj = tmp['surv_prob'].values.astype(float)
                    dropout_prob_traj = 1 - survival_prob_traj[
                        1:] / survival_prob_traj[:-1]
                    dropout_prob_traj = np.append(np.array([0]),
                                                  dropout_prob_traj)
                else:
                    survival_prob_traj = None
                    dropout_prob_traj = None
                if dropout_col in tmp.columns:
                    dropout_next_traj = tmp[dropout_col].values
                elif len(reward_traj) < self.max_T:
                    dropout_next_traj = np.zeros_like(reward_traj)
                    dropout_next_traj[-1] = 1
                else:
                    dropout_next_traj = np.zeros_like(reward_traj)
            else:
                dropout_prob_traj = tmp['custom_dropout_prob'].values.astype(float)
                if dropout_prob_traj[-1] is None:
                    dropout_prob_traj[-1] = -1
                survival_prob_traj = np.append(
                    1, (1 - dropout_prob_traj).cumprod())[:-1]

                # dropout_index = np.where(
                #     np.random.uniform(
                #         low=0, high=1, size=dropout_prob_traj.shape) <
                #     dropout_prob_traj)[0]
                # dropout_next_traj = np.zeros_like(reward_traj)
                # if len(dropout_index) > 0:
                #     dropout_index = dropout_index.min()
                #     dropout_next_traj[dropout_index] = 1
                # else:
                #     dropout_index = self.max_T
                # if dropout_prob_traj[-1] < 0:
                #     dropout_prob_traj[-1] = np.nan

                assert dropout_col in tmp.columns
                dropout_next_traj = tmp[dropout_col].values
                dropout_index = np.where(dropout_next_traj == 1)[0]
                if len(dropout_index) > 0:
                    dropout_index = dropout_index.min()
                else:
                    dropout_index = self.max_T
                T = min(dropout_index + 1, self.max_T)
                S_traj = S_traj[:T]
                A_traj = A_traj[:T]
                reward_traj = reward_traj[:T]
                survival_prob_traj = survival_prob_traj[:T]
                dropout_next_traj = dropout_next_traj[:T]
                dropout_prob_traj = dropout_prob_traj[:T]

            self.total_N += T
            # if T < self.max_T:
            #     incomplete_cnt += 1
            incomplete_cnt += sum(dropout_next_traj)

            if T > burn_in:
                self.masked_buffer[(i)] = [
                    S_traj[burn_in:], 
                    A_traj[burn_in:], 
                    reward_traj[burn_in:],
                    T,
                    survival_prob_traj[burn_in:] if survival_prob_traj is not None else None,
                    dropout_next_traj[burn_in:], 
                    dropout_prob_traj[burn_in:] if dropout_prob_traj is not None else None
                ]
            if mnar_nextobs_var is not None or mnar_noninstrument_var is not None is not None or mnar_instrument_var is not None:
                self.misc_buffer[(i)] = {}
            if mnar_nextobs_var is not None:
                mnar_nextobs_arr = tmp[mnar_nextobs_var].values
                if len(mnar_nextobs_arr.shape) == 1:
                    mnar_nextobs_arr = mnar_nextobs_arr.reshape(-1, 1)
                self.misc_buffer[(i)]['mnar_nextobs_arr'] = np.vstack([
                    mnar_nextobs_arr[burn_in + 1:],
                    np.zeros(shape=(1, mnar_nextobs_arr.shape[1])) * np.nan
                ]) # shift one step
            if mnar_noninstrument_var is not None:
                mnar_noninstrument_arr = tmp[mnar_noninstrument_var].values
                self.misc_buffer[(
                    i
                )]['mnar_noninstrument_arr'] = mnar_noninstrument_arr[burn_in:]
            if mnar_instrument_var is not None:
                mnar_instrument_arr = tmp[mnar_instrument_var].values
                self.misc_buffer[(
                    i)]['mnar_instrument_arr'] = mnar_instrument_arr[burn_in:]

        self.n = len(self.masked_buffer)
        self.dropout_rate = incomplete_cnt / self.n
        self.missing_rate = 1 - self.total_N / (self.n * self.max_T)
        self._initial_obs = np.array(self._initial_obs)
        print('Import finished!')

    def import_holdout_buffer(self,
                              data=None,
                              data_filename=None,
                              static_state_cols=None,
                              dynamic_state_cols=None,
                              action_col='action',
                              reward_col='reward',
                              id_col='id',
                              dropout_col=None,
                              reward_transform=None,
                              burn_in=0,
                              **kwargs):
        """Import unscaled trajectories from a dataframe to self.holdout_buffer. 
        This buffer is used to estimate behavior policy or learn optimal policy.

        Args:
            data (pd.DataFrame): dataframe to be imported
            data_filename (str): path to the table
            static_state_cols (list): column names of static features
            dynamic_state_cols (list): column names of dynamic features
            action_col (str): name of action column
            reward_col (str): name of reward column
            id_col (str): name of reward column
            dropout_col (str): name of the binary column that indicates dropout event
            subsample_id (list): ids of subsample to be imported
            reward_transform (callable): transformation funtion of reward
            burn_in (int): length of burn-in period
            kwargs (dict): kwargs passed to pd.read_csv()  
        """
        if data is None:
            print(f'Import from {data_filename} to buffer')
            data = pd.read_csv(data_filename, **kwargs)

        state_cols = static_state_cols + dynamic_state_cols
        if hasattr(self, 'state_cols'):
            assert self.state_cols == state_cols
        if hasattr(self, 'num_actions'):
            assert self.num_actions == data[action_col].nunique()
        if hasattr(self, 'state_dim'):
            assert self.state_dim == len(state_cols)
        if hasattr(self, 'burn_in'):
            assert self.burn_in == burn_in

        id_list = data[id_col].unique().tolist()
        num_trajs = len(id_list)
        self.holdout_buffer = {}
        if state_cols is None:
            state_cols = [c for c in data.columns if c.startswith('X')]
        for i, id_ in enumerate(id_list):
            tmp = data.loc[data[id_col] == id_]
            S_traj = tmp[state_cols].values
            A_traj = tmp[action_col].values
            reward_traj = tmp[reward_col].values
            if reward_transform is not None:
                reward_traj = reward_transform(reward_traj)
            T = len(A_traj)
            if 'surv_prob' in tmp.columns:
                survival_prob_traj = tmp['surv_prob'].values
                dropout_prob_traj = 1 - survival_prob_traj[
                    1:] / survival_prob_traj[:-1]
                dropout_prob_traj = np.append(np.array([0]), dropout_prob_traj)
            else:
                survival_prob_traj = None
                dropout_prob_traj = None

            dropout_next_traj = np.zeros_like(reward_traj)
            if dropout_col in tmp.columns:
                dropout_next_traj = tmp[dropout_col].values
            elif len(dropout_next_traj) < self.max_T:
                dropout_next_traj[-1] = 1

            if T > burn_in:
                self.holdout_buffer[(i)] = [
                    S_traj[burn_in:], A_traj[burn_in:], reward_traj[burn_in:],
                    T, survival_prob_traj[burn_in:]
                    if survival_prob_traj is not None else None,
                    dropout_next_traj[burn_in:], dropout_prob_traj[burn_in:]
                    if dropout_prob_traj is not None else None
                ]
        print('Import finished!')

    def train_state_model(self,
                          model_type="linear",
                          train_ratio=0.8,
                          scale_obs=False,
                          export_dir=None,
                          pkl_filename="state_model.pkl",
                          seed=None,
                          **kwargs):
        """Train state transition model.

        Args:
            model_type (str): select from "linear", "mlp" and "rf"
            train_ratio (float): proportion of training set
            scale_obs (bool): if True, scale observation
            export_dir (str): directory to export model
            pkl_filename (str): filename for trained model
            seed (int): random seed for data spliting
            kwargs (dict): passed to model
        """
        # prepare data
        obs_list, next_obs_list, action_list = [], [], []
        self.fitted_state_model = {}
        if pkl_filename is not None:
            self.state_model_filename = os.path.join(export_dir, pkl_filename)
            pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
        for i in self.masked_buffer.keys():
            S_traj, A_traj = self.masked_buffer[i][:2]
            obs_list.append(S_traj[:-1])
            next_obs_list.append(S_traj[1:])
            T = S_traj[:-1].shape[0]
            action_list.append(A_traj[:T].reshape(-1, 1))
        obs = np.vstack(obs_list)  # (total_T, S_dim)
        next_obs = np.vstack(next_obs_list)  # (total_T, S_dim)
        if scale_obs:
            if not hasattr(self.scaler, 'data_min_') or np.min(
                    self.scaler.data_min_) == -np.inf or np.max(
                        self.scaler.data_max_) == np.inf:
                self.scaler.fit(np.vstack([obs, next_obs]))
            obs = self.scaler.transform(obs)
            next_obs = self.scaler.transform(next_obs)
            self.fitted_state_model['scaler'] = self.scaler
        else:
            self.fitted_state_model['scaler'] = iden()
        actions = np.vstack(action_list)  # (total_T, 1)
        X, y = np.hstack([obs, actions]), next_obs
        data = {}
        for a in range(self.num_actions):
            X_a, y_a = X[X[:, -1] == a, :-1], y[X[:, -1] == a]
            X_train, X_test, y_train, y_test = train_test_split(
                X_a, y_a, test_size=1 - train_ratio, random_state=seed)
            data[a] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }

        if model_type.lower() == "linear":
            for a in range(self.num_actions):
                linear_reg = LinearRegression(fit_intercept=True)
                linear_reg.fit(X=data[a]['X_train'],
                               y=data[a]['y_train'],
                               **kwargs)
                mse_error = mean_squared_error(
                    y_true=data[a]['y_test'],
                    y_pred=linear_reg.predict(X=data[a]['X_test']))
                print(f'MSE on test set with action {a}: {mse_error}')
                self.fitted_state_model[a] = linear_reg
        elif model_type.lower() == "mlp":
            for a in range(self.num_actions):
                # create a validation set out of test set
                X_val_a, X_test_a, y_val_a, y_test_a = train_test_split(
                    data[a]['X_test'],
                    data[a]['y_test'],
                    test_size=0.5,
                    random_state=seed)
                mlp_model = MLPModule(input_dim=self.state_dim,
                                      output_dim=self.state_dim,
                                      hidden_sizes=[64, 64],
                                      hidden_nonlinearity=nn.ReLU(),
                                      lr=2e-3,
                                      loss=F.mse_loss)
                early_stop_callback = EarlyStopping(monitor='valid_loss',
                                                    patience=3,
                                                    verbose=False,
                                                    mode='min')
                trainer = pl.Trainer(max_epochs=250,
                                     callbacks=[early_stop_callback],
                                     **kwargs)
                train_dataloader = DataLoader(
                    TensorDataset(torch.Tensor(data[a]['X_train']),
                                  torch.Tensor(data[a]['y_train'])),
                    batch_size=min(len(data[a]['X_train']), 128),
                    drop_last=True)
                val_dataloader = DataLoader(TensorDataset(
                    torch.Tensor(X_val_a), torch.Tensor(y_val_a)),
                                            batch_size=min(len(X_val_a), 32),
                                            drop_last=True)
                test_dataloader = DataLoader(TensorDataset(
                    torch.Tensor(X_test_a), torch.Tensor(y_test_a)),
                                             batch_size=len(X_test_a),
                                             drop_last=True)
                trainer.fit(model=mlp_model,
                            train_dataloader=train_dataloader,
                            val_dataloaders=val_dataloader)
                eval_output = trainer.test(model=mlp_model,
                                           test_dataloaders=test_dataloader,
                                           verbose=False)
                mse_error = eval_output[0]['test_loss']
                print(f'MSE on test set with action {a}: {mse_error}')
                self.fitted_state_model[a] = mlp_model
        elif model_type.lower() == "rf":
            for a in range(self.num_actions):
                rf_reg = RandomForestRegressor(random_state=seed, **kwargs)
                rf_reg.fit(X=data[a]['X_train'], y=data[a]['y_train'])
                mse_error = mean_squared_error(
                    y_true=data[a]['y_test'],
                    y_pred=rf_reg.predict(X=data[a]['X_test']))
                print(f'MSE on test set with action {a}: {mse_error}')
                self.fitted_state_model[a] = rf_reg
            del rf_reg
            _ = gc.collect()
        else:
            raise NotImplementedError

        joblib.dump(value=self.fitted_state_model,
                    filename=self.state_model_filename,
                    compress=3)

    def train_reward_model(self,
                           model_type="linear",
                           train_ratio=0.8,
                           scale_obs=False,
                           export_dir=None,
                           pkl_filename="reward_model.pkl",
                           seed=None,
                           **kwargs):
        """Train reward model.

        Args:
            model_type (str): select from "linear", "mlp" and "rf"
            train_ratio (float): proportion of training set
            scale_obs (bool): if True, scale observation
            export_dir (str): directory to export model
            pkl_filename (str): filename for trained model
            kwargs (dict): passed to model
        """
        # prepare data
        obs_list, next_obs_list, action_list, reward_list = [], [], [], []
        self.fitted_reward_model = {}
        if pkl_filename is not None:
            self.reward_model_filename = os.path.join(export_dir, pkl_filename)
            pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
        for i in self.masked_buffer.keys():
            S_traj, A_traj, R_traj = self.masked_buffer[i][:3]
            obs_list.append(S_traj[:-1])
            next_obs_list.append(S_traj[1:])
            T = S_traj[:-1].shape[0]
            action_list.append(A_traj[:T].reshape(-1, 1))
            reward_list.append(R_traj[:T].reshape(-1, 1))
        obs = np.vstack(obs_list)  # (total_T, S_dim)
        next_obs = np.vstack(next_obs_list)  # (total_T, S_dim)
        if scale_obs:
            if not hasattr(self.scaler, 'data_min_') or np.min(
                    self.scaler.data_min_) == -np.inf or np.max(
                        self.scaler.data_max_) == np.inf:
                self.scaler.fit(np.vstack([obs, next_obs]))
            obs = self.scaler.transform(obs)
            next_obs = self.scaler.transform(next_obs)
            self.fitted_reward_model['scaler'] = self.scaler
        else:
            self.fitted_reward_model['scaler'] = iden()
        actions = np.vstack(action_list).squeeze()  # (total_T, )
        rewards = np.vstack(reward_list).squeeze()  # (total_T, )
        X, y = np.hstack([obs, next_obs]), rewards
        data = {}
        for a in range(self.num_actions):
            X_a, y_a = X[actions == a], y[actions == a]
            X_train, X_test, y_train, y_test = train_test_split(
                X_a, y_a, test_size=1 - train_ratio, random_state=seed)
            data[a] = {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }
        if model_type == "linear":
            for a in range(self.num_actions):
                linear_reg = LinearRegression(fit_intercept=True)
                linear_reg.fit(X=data[a]['X_train'],
                               y=data[a]['y_train'],
                               **kwargs)
                mse_error = mean_squared_error(
                    y_true=data[a]['y_test'],
                    y_pred=linear_reg.predict(X=data[a]['X_test']))
                print(f'MSE on test set with action {a}: {mse_error}')
                self.fitted_reward_model[a] = linear_reg
        elif model_type.lower() == "mlp":
            for a in range(self.num_actions):
                # create a validation set out of test set
                X_val_a, X_test_a, y_val_a, y_test_a = train_test_split(
                    data[a]['X_test'],
                    data[a]['y_test'].reshape(-1, 1),
                    test_size=0.5,
                    random_state=seed)
                mlp_model = MLPModule(input_dim=self.state_dim * 2,
                                      output_dim=1,
                                      hidden_sizes=[64, 64],
                                      hidden_nonlinearity=nn.ReLU(),
                                      lr=1e-3,
                                      loss=F.mse_loss)
                early_stop_callback = EarlyStopping(monitor='valid_loss',
                                                    patience=3,
                                                    verbose=False,
                                                    mode='min')
                trainer = pl.Trainer(max_epochs=250,
                                     callbacks=[early_stop_callback],
                                     **kwargs)
                train_dataloader = DataLoader(
                    TensorDataset(torch.Tensor(data[a]['X_train']),
                                  torch.Tensor(data[a]['y_train'])),
                    batch_size=min(len(data[a]['X_train']), 128),
                    drop_last=True)
                val_dataloader = DataLoader(TensorDataset(
                    torch.Tensor(X_val_a), torch.Tensor(y_val_a)),
                                            batch_size=min(len(X_val_a), 32),
                                            drop_last=True)
                test_dataloader = DataLoader(TensorDataset(
                    torch.Tensor(X_test_a), torch.Tensor(y_test_a)),
                                             batch_size=len(X_test_a),
                                             drop_last=True)
                trainer.fit(model=mlp_model,
                            train_dataloader=train_dataloader,
                            val_dataloaders=val_dataloader)
                eval_output = trainer.test(model=mlp_model,
                                           test_dataloaders=test_dataloader,
                                           verbose=False)
                mse_error = eval_output[0]['test_loss']
                print(f'MSE on test set with action {a}: {mse_error}')
                self.fitted_reward_model[a] = mlp_model
        elif model_type == "rf":
            for a in range(self.num_actions):
                rf_reg = RandomForestRegressor(random_state=seed, **kwargs)
                rf_reg.fit(X=data[a]['X_train'], y=data[a]['y_train'])
                mse_error = mean_squared_error(
                    y_true=data[a]['y_test'],
                    y_pred=rf_reg.predict(X=data[a]['X_test']))
                print(f'MSE on test set with action {a}: {mse_error}')
                self.fitted_reward_model[a] = rf_reg
            del rf_reg  # rf model tends to take up a large amount of memory
            _ = gc.collect()
        else:
            raise NotImplementedError

        joblib.dump(value=self.fitted_reward_model,
                    filename=self.reward_model_filename,
                    compress=3)

    def train_dropout_model(self,
                            model_type="linear",
                            train_ratio=0.8,
                            scale_obs=False,
                            missing_mechanism='mar',
                            dropout_obs_count_thres=1,
                            subsample_index=None,
                            export_dir=None,
                            pkl_filename="dropout_model.pkl",
                            seed=None,
                            include_reward=True,
                            instrument_var_index=None,
                            mnar_y_transform=None,
                            gamma_init=None,
                            bandwidth_factor=1.5,
                            verbose=True,
                            **kwargs):
        """
        Train dropout model.

        Args:
            model_type (str): select from "linear", "mlp" and "rf"
            train_ratio (float): proportion of training set
            scale_obs (bool): if True, scale observation before fitting the model
            missing_mechanism (str): "mnar" or "mar"
            dropout_obs_count_thres (int): number of observations that is not subject to dropout
            subsample_index (list): ids of subsample to train the model
            export_dir (str): directory to export model
            pkl_filename (str): filename for trained model
            seed (int): random seed to split data
            include_reward (bool): if True, include reward as feature in dropuot model
            instrument_var_index (int): index of the instrument variable
            mnar_y_transform (callable): input next_obs and reward, output Y term for the mnar dropout model 
            gamma_init (float): initial value for gamma in MNAR estimation
            bandwidth_factor (float): the constant used in bandwidth calculation
            kwargs (dict): passed to model
        """
        self.missing_mechanism = missing_mechanism
        self.dropout_obs_count_thres = max(dropout_obs_count_thres - 1, 0)  # -1 because this is the index
        self._include_reward = include_reward
        self._mnar_y_transform = mnar_y_transform
        self._instrument_var_index = instrument_var_index
        self._scale_obs = scale_obs
        obs_list, next_obs_list, action_list, reward_list = [], [], [], []
        dropout_next_list, dropout_prob_list = [], []
        # done_list = []
        mnar_nextobs_list, mnar_noninstrument_list, mnar_instrument_list = [], [], []
        self.fitted_dropout_model = {}
        if export_dir is not None and pkl_filename is not None:
            self.dropout_model_filename = os.path.join(export_dir,
                                                       pkl_filename)
            pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
        else:
            self.dropout_model_filename = None
        keys = self.masked_buffer.keys()
        if subsample_index is not None:
            keys = subsample_index
        for i in keys:
            S_traj, A_traj, R_traj, _, _, don_traj, dop_traj = self.masked_buffer[i][:7]
            if self.misc_buffer:
                mnar_nextobs_traj = self.misc_buffer[i].get(
                    'mnar_nextobs_arr', None)
                mnar_noninstrument_traj = self.misc_buffer[i].get(
                    'mnar_noninstrument_arr', None)
                mnar_instrument_traj = self.misc_buffer[i].get(
                    'mnar_instrument_arr', None)
            else:
                mnar_nextobs_traj = None
                mnar_noninstrument_traj = None
                mnar_instrument_traj = None
            if missing_mechanism == 'mar':
                T = A_traj.shape[0]
                obs_list.append(S_traj[self.dropout_obs_count_thres:T])
                reward_list.append(
                    np.append(0,
                              R_traj)[self.dropout_obs_count_thres:T].reshape(
                                  -1, 1))  # R_{t-1}
            elif missing_mechanism == 'mnar':
                T = A_traj.shape[0]
                
                if sum(don_traj) == 0: # no dropout
                    T -= 1
                    if T <= self.dropout_obs_count_thres:
                        continue

                obs_list.append(S_traj[self.dropout_obs_count_thres:T])
                if mnar_nextobs_traj is not None:
                    mnar_nextobs_list.append(
                        mnar_nextobs_traj[self.dropout_obs_count_thres:T])
                if mnar_noninstrument_traj is not None:
                    mnar_noninstrument_list.append(
                        mnar_noninstrument_traj[self.
                                                dropout_obs_count_thres:T])
                if mnar_instrument_traj is not None:
                    mnar_instrument_list.append(
                        mnar_instrument_traj[self.dropout_obs_count_thres:T])
                if len(S_traj) == T:
                    next_obs_list.append(
                        np.vstack([
                            S_traj[self.dropout_obs_count_thres + 1:],
                            [[np.nan] * self.state_dim]
                        ]))
                else:
                    next_obs_list.append(S_traj[self.dropout_obs_count_thres+1:T+1])
                reward_list.append(
                    R_traj[self.dropout_obs_count_thres:T].reshape(-1,
                                                                   1))  # R_{t}

            action_list.append(A_traj[self.dropout_obs_count_thres:T].reshape(
                -1, 1))
            dropout_next_list.append(
                don_traj[self.dropout_obs_count_thres:T].reshape(-1, 1))
            if dop_traj is not None:
                dropout_prob_list.append(
                    dop_traj[self.dropout_obs_count_thres:T].reshape(-1, 1))

            # dones = np.zeros(shape=T - self.dropout_obs_count_thres)
            # if T == self.max_T:
            #     dones[-1] = 1
            # done_list.append(dones.reshape(-1, 1))
        unscaled_obs = obs = np.vstack(obs_list)  # (total_T, S_dim)
        if next_obs_list:
            next_obs = np.vstack(next_obs_list)  # (total_T, S_dim)
        else:
            next_obs = None
        # we scale these arrays separately
        mnar_nextobs_arr = np.vstack(
            mnar_nextobs_list) if mnar_nextobs_list else None
        mnar_noninstrument_arr = np.vstack(
            mnar_noninstrument_list) if mnar_noninstrument_list else None

        mnar_instrument_arr = np.concatenate(
            mnar_instrument_list) if mnar_instrument_list else None
        if scale_obs:
            if not hasattr(self.scaler, 'data_min_') or not hasattr(
                    self.scaler, 'data_max_') or np.min(
                        self.scaler.data_min_) == -np.inf or np.max(
                            self.scaler.data_max_) == np.inf:
                if next_obs is not None:
                    self.scaler.fit(np.vstack([obs, next_obs]))
                else:
                    self.scaler.fit(obs)
            obs = self.scaler.transform(obs)
            if next_obs_list:
                next_obs = self.scaler.transform(next_obs)
            self.fitted_dropout_model['scaler'] = self.scaler
        else:
            self.fitted_dropout_model['scaler'] = iden()
        actions = np.vstack(action_list).squeeze()  # (total_T, )
        rewards = np.vstack(reward_list).squeeze()  # (total_T, )
        dropout_next = np.vstack(dropout_next_list).squeeze()  # (total_T, )
        # dones = np.vstack(done_list).squeeze()  # (total_T, )
        if dropout_prob_list:
            dropout_prob = np.vstack(
                dropout_prob_list).squeeze()  # (total_T, )
        else:
            dropout_prob = None

        if missing_mechanism == 'mar':
            if include_reward:
                X, y = np.hstack([obs, rewards.reshape(-1, 1)]), dropout_next
            else:
                X, y = obs, dropout_next
            data = {}
            obs_train_list = [] 
            probT_train_list = []
            action_train_list = []
            for a in range(self.num_actions):
                X_a, y_a = X[actions == a], y[actions == a]
                # assert len(np.unique(y_a)) > 1 # ensure at least one positive label
                if dropout_prob is not None:
                    probT_a = dropout_prob[actions == a]
                    if train_ratio < 1:
                        X_train, X_test, y_train, y_test, obs_train, obs_test, probT_train, probT_test = train_test_split(
                            X_a,
                            y_a,
                            obs[actions == a],
                            probT_a,
                            test_size=1 - train_ratio,
                            random_state=seed
                        )
                    else:
                        # use all data for training
                        X_train = X_test = X_a
                        y_train = y_test = y_a
                        obs_train = obs_test = obs[actions == a]
                        probT_train = probT_test = probT_a
                    obs_train_list.append(obs_train)
                    probT_train_list.append(probT_train)
                else:
                    if train_ratio < 1:
                        X_train, X_test, y_train, y_test, obs_train, obs_test = train_test_split(
                            X_a, y_a, obs[actions == a], test_size=1 - train_ratio, random_state=seed)
                    else:
                        X_train = X_test = X_a
                        y_train = y_test = y_a
                        obs_train = obs_test = obs[actions == a]
                    probT_train, probT_test = None, None
                    obs_train_list.append(obs_train)
                data[a] = {
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'prob_train': probT_train,
                    'prob_test': probT_test
                }
                action_train_list.append([a] * len(X_train))
            obs_train = np.vstack(obs_train_list)
            action_train = np.concatenate(action_train_list)
            if probT_train_list:
                probT_train = np.concatenate(probT_train_list)
            if model_type.lower() == "linear":
                for a in range(self.num_actions):
                    if np.sum(data[a]['y_train']) == 0: # all 0's
                        self.fitted_dropout_model[a] = 0.
                        continue
                    elif np.sum(data[a]['y_train']) == len(data[a]['y_train']): # all 1's
                        self.fitted_dropout_model[a] = 1.
                        continue
                    logit_reg = LogisticRegression(fit_intercept=True,
                                                   **kwargs)
                    logit_reg.fit(X=data[a]['X_train'],
                                  y=data[a]['y_train'],
                                  **kwargs)
                    test_acc = accuracy_score(
                        y_true=data[a]['y_test'],
                        y_pred=logit_reg.predict(X=data[a]['X_test']))
                    if verbose:
                        print(
                            f'Accuracy on test set with action {a}: {test_acc}'
                        )
                    try:
                        test_auroc = roc_auc_score(
                            y_true=data[a]['y_test'],
                            y_score=logit_reg.predict_proba(
                                X=data[a]['X_test'])[:, 1])
                        if verbose:
                            print(
                                f'AUROC on test set with action {a}: {test_auroc}'
                            )
                    except:
                        pass
                    if data[a]['prob_test'] is not None:
                        nan_index = np.isnan(data[a]['prob_test'])
                        test_mse = mean_squared_error(
                            y_true=data[a]['prob_test'][~nan_index],
                            y_pred=logit_reg.predict_proba(
                                X=data[a]['X_test'])[:, 1][~nan_index])
                        if verbose:
                            print(
                                f'MSE on test set with action {a}: {test_mse}')
                    self.fitted_dropout_model[a] = logit_reg
            elif model_type.lower() == "rf":
                for a in range(self.num_actions):
                    if np.sum(data[a]['y_train']) == 0: # all 0's
                        self.fitted_dropout_model[a] = 0.
                        continue
                    elif np.sum(data[a]['y_train']) == len(data[a]['y_train']): # all 1's
                        self.fitted_dropout_model[a] = 1.
                        continue
                    rf_clf = RandomForestClassifier(random_state=seed,
                                                    **kwargs)
                    rf_clf.fit(X=data[a]['X_train'],
                               y=data[a]['y_train'],
                               **kwargs)
                    test_acc = accuracy_score(
                        y_true=data[a]['y_test'],
                        y_pred=rf_clf.predict(X=data[a]['X_test']))
                    if verbose:
                        print(
                            f'Accuracy on test set with action {a}: {test_acc}'
                        )
                    try:
                        test_auroc = roc_auc_score(
                            y_true=data[a]['y_test'],
                            y_score=rf_clf.predict_proba(
                                X=data[a]['X_test'])[:, 1])
                        if verbose:
                            print(
                                f'AUROC on test set with action {a}: {test_auroc}'
                            )
                    except:
                        pass
                    if data[a]['prob_test'] is not None:
                        nan_index = np.isnan(data[a]['prob_test'])
                        test_mse = mean_squared_error(
                            y_true=data[a]['prob_test'][~nan_index],
                            y_pred=rf_clf.predict_proba(
                                X=data[a]['X_test'])[:, 1][~nan_index])
                        if verbose:
                            print(
                                f'MSE on test set with action {a}: {test_mse}')
                    self.fitted_dropout_model[a] = rf_clf
                del rf_clf  # rf model tends to take up a large amount of memory
                _ = gc.collect()
            else:
                raise NotImplementedError
            self._dropout_prob_mse = test_mse if data[a][
                'prob_test'] is not None else None
            if self.dropout_model_filename is not None:
                joblib.dump(value=self.fitted_dropout_model,
                            filename=self.dropout_model_filename,
                            compress=3)
        else:
            if include_reward:
                print('use the reward as outcome')
                # use reward as outcome
                y_arr = rewards
                y_dim = 1
            elif mnar_nextobs_arr is None:
                if mnar_y_transform is not None:
                    orig_next_obs = self.fitted_dropout_model[
                        'scaler'].inverse_transform(
                            next_obs)  # transform to its original scale
                    y_arr = mnar_y_transform(
                        np.hstack([orig_next_obs,
                                   rewards.reshape(-1, 1)]))
                    # print(f'y_arr:{y_arr}')
                # elif include_reward:
                #     y_arr = rewards
                else:
                    y_arr = next_obs
                if len(y_arr.shape) == 1 or (len(y_arr.shape) == 2
                                             and y_arr.shape[1] == 1):
                    y_dim = 1
                else:
                    y_dim = y_arr.shape[1]
                y_arr = y_arr.reshape(-1, y_dim)  # (n,y_dim)
            else:
                y_arr = mnar_nextobs_arr
                if len(y_arr.shape) == 1:
                    y_dim = 1
                else:
                    y_dim = y_arr.shape[1]
                y_arr = y_arr.reshape(-1, y_dim)
            print(f'y_dim: {y_dim}')
            if self._scale_obs:
                self._y_arr_scaler = MinMaxScaler()
                self._y_arr_scaler.fit(y_arr)
                y_arr = self._y_arr_scaler.transform(y_arr)
            if mnar_noninstrument_arr is None:
                if not isinstance(instrument_var_index,
                                  np.ndarray) and not hasattr(
                                      instrument_var_index, '__iter__'):
                    non_instrument_index = [
                        i for i in range(obs.shape[1])
                        if i != instrument_var_index
                    ]
                else:
                    non_instrument_index = [
                        i for i in range(obs.shape[1])
                        if i not in set(instrument_var_index)
                    ]
                u_dim = len(non_instrument_index)
                u_arr = obs[:,
                            non_instrument_index].reshape(len(obs),
                                                          u_dim)  # (n, u_dim)
            else:
                if len(mnar_noninstrument_arr.shape) == 1:
                    u_dim = 1
                else:
                    u_dim = mnar_noninstrument_arr.shape[1]
                u_arr = mnar_noninstrument_arr.reshape(-1, u_dim)
            if self._scale_obs:
                self._u_arr_scaler = MinMaxScaler()
                self._u_arr_scaler.fit(u_arr)
                u_arr = self._u_arr_scaler.transform(u_arr)
            if mnar_instrument_arr is None:
                assert instrument_var_index is not None, 'please provide the name of the instrument variable, otherwise there is identifibility issue.'
                z_arr = obs[:, instrument_var_index]  # (n,)
                # discretize Z
                L = y_dim + 3  # try y_dim + 1, y_dim + 2, or y_dim + 3
                print(f'L: {L}')
                z_min, z_max = z_arr.min(), z_arr.max()
                self.instrument_disc_bins = np.quantile(
                    a=z_arr, q=np.linspace(0, 1, L + 1))  # divide by quantile
                # make sure bins are non-overlap
                bins_nonoverlap = []
                for i in range(len(self.instrument_disc_bins)):
                    if not bins_nonoverlap or bins_nonoverlap[
                            -1] != self.instrument_disc_bins[i]:
                        bins_nonoverlap.append(self.instrument_disc_bins[i])
                self.instrument_disc_bins = bins_nonoverlap
                L = len(self.instrument_disc_bins) - 1
                assert L >= y_dim + 1
                # handle the boundary
                self.instrument_disc_bins[0] -= 0.1
                self.instrument_disc_bins[-1] += 0.1
                if verbose:
                    print(f'bins:{self.instrument_disc_bins}')
                z_arr = np.digitize(z_arr, bins=self.instrument_disc_bins)
                z_arr_count = Counter(z_arr)
                print(f'Counter(z_arr): {z_arr_count}')
                # make sure every level of Z has samples in it
                for i in range(1, L + 1):
                    assert z_arr_count[i] != 0
            else:
                z_arr = mnar_instrument_arr.reshape(
                    -1)  # assume z is already discretized
                L = len(np.unique(z_arr))
                print(f'L:{L}')
                # discretized Z start from 1 instead of 0
                if min(z_arr) <= 0:
                    z_arr -= min(z_arr) - 1

            if dropout_prob is not None:
                probT_arr = dropout_prob.astype('float')
                logitT_arr = np.log(1 / probT_arr - 1)
            else:
                probT_arr, logitT_arr = None, None
            # delta_arr = 1 - np.logical_or(dropout_next, dones) # indicator of observed sample
            # done=True does not necessarily indicate dropout
            delta_arr = 1 - dropout_next
            print('count of dropout indicator:', sum(delta_arr == 0))

            if verbose:
                print(f'observed proportion: {np.mean(delta_arr)}')

            delta_train = None
            while delta_train is None or np.mean(delta_train) == 1:
                if probT_arr is not None:
                    if train_ratio < 1:
                        u_train, u_test, z_train, z_test, obs_train, obs_test, y_train, y_test, delta_train, delta_test, probT_train, probT_test = train_test_split(
                            u_arr,
                            z_arr,
                            obs,
                            y_arr,
                            delta_arr,
                            probT_arr,
                            test_size=1 - train_ratio,
                            random_state=seed)
                    else:
                        u_train = u_test = u_arr
                        z_train = z_test = z_arr
                        obs_train = obs_test = obs
                        y_train = y_test = y_arr
                        delta_train = delta_test = delta_arr
                        probT_train = probT_test = probT_arr
                else:
                    if train_ratio < 1:
                        u_train, u_test, z_train, z_test, obs_train, obs_test, y_train, y_test, delta_train, delta_test = train_test_split(
                            u_arr,
                            z_arr,
                            obs,
                            y_arr,
                            delta_arr,
                            test_size=1 - train_ratio,
                            random_state=seed)
                    else:
                        u_train = u_test = u_arr
                        z_train = z_test = z_arr
                        obs_train = obs_test = obs
                        y_train = y_test = y_arr
                        delta_train = delta_test = delta_arr
                    probT_train, probT_test = None, None
            # train model
            mnar_clf = ExpoTiltingClassifierMNAR()
            if verbose:
                print(f'observed proportion (train): {np.mean(delta_train)}')
            fit_start = time.time()
            bounds = None

            if gamma_init is not None:
                bounds = ((gamma_init - 1.5, gamma_init + 1.5), ) # can set custom search range

            mnar_clf.fit(L=L,
                         z=z_train,
                         u=u_train,
                         y=y_train,
                         delta=delta_train,
                         seed=seed,
                         gamma_init=gamma_init,
                         bounds=bounds,
                         verbose=verbose,
                         bandwidth_factor=bandwidth_factor)
            self.mnar_gamma = mnar_clf.gamma_hat
            if verbose:
                print(
                    f'fitting mnar-ipw model takes {time.time()-fit_start} secs.'
                )

            # the following part is only used for simulation
            if True:
                prob_pred_test = 1 - \
                    mnar_clf.predict_proba(u=u_test, z=z_test, y=y_test)

                if verbose:
                    print(f'gamma_hat: {mnar_clf.gamma_hat}')
                    print(
                        f'max_prob_pred_test: {prob_pred_test[delta_test==1].max()}'
                    )
                    if probT_test is not None:
                        print(
                            f'max_probT_test: {probT_test[delta_test==1].max()}'
                        )
                try:
                    test_auroc = roc_auc_score(y_true=1 - delta_test,
                                               y_score=prob_pred_test)
                    if verbose:
                        print(f'AUROC on test set: {test_auroc}')
                except:
                    pass

                if probT_test is not None:
                    test_mse = mean_squared_error(
                        y_true=probT_test[delta_test == 1],
                        y_pred=prob_pred_test[delta_test == 1])
                    if verbose:
                        print(f'MSE on test set: {test_mse}')
                    self._dropout_prob_mse = test_mse

            self.fitted_dropout_model['model'] = mnar_clf
            # save the model
            if self.dropout_model_filename is not None:
                mnar_clf.save(self.dropout_model_filename)

        # add some additional tracking
        if verbose and probT_train is not None:
            prob_pred_df = pd.DataFrame({
                'X1': obs_train[:, 0],
                'X2': obs_train[:, 1],
                'dropout_prob': probT_train
            })
            if missing_mechanism == 'mnar':
                prob_pred_train = 1 - \
                    mnar_clf.predict_proba(u=u_train, z=z_train, y=y_train)
            else:
                prob_pred_train = np.zeros_like(probT_train)
                for a in range(self.num_actions):
                    if isinstance(self.fitted_dropout_model[a], float):
                        prob_pred_train[action_train == a] = self.fitted_dropout_model[a]
                    else:
                        prob_pred_train[action_train == a] = self.fitted_dropout_model[a].predict_proba(X=data[a]['X_train'])[:, 1]
            prob_pred_df['dropout_prob_est'] = prob_pred_train
            if missing_mechanism == 'mnar':
                train_mse = mean_squared_error(
                    y_true=probT_train[delta_train == 1],
                    y_pred=prob_pred_train[delta_train == 1])
            else:
                nan_index = np.isnan(probT_train)
                train_mse = mean_squared_error(y_true=probT_train[~nan_index], y_pred=prob_pred_train[~nan_index])                
            logitT_train = np.log(1 / np.maximum(probT_train, 1e-8) - 1)
            logit_pred_train = np.log(1 /
                                        np.maximum(prob_pred_train, 1e-8) -
                                        1)
            print(
                'true obs prob (0.0/0.25/0.5/0.75/1.0 quantile): {0:.2f}/{1:.2f}/{2:.2f}/{3:.2f}/{4:.2f}'
                .format(np.nanmin(probT_train),
                        np.nanquantile(probT_train, 0.25),
                        np.nanquantile(probT_train, 0.5),
                        np.nanquantile(probT_train, 0.75),
                        np.nanmax(probT_train)))
            print(
                'true obs logit (0.0/0.25/0.5/0.75/1.0 quantile): {0:.2f}/{1:.2f}/{2:.2f}/{3:.2f}/{4:.2f}'
                .format(np.nanmin(logitT_train),
                        np.nanquantile(logitT_train, 0.25),
                        np.nanquantile(logitT_train, 0.5),
                        np.nanquantile(logitT_train, 0.75),
                        np.nanmax(logitT_train)))
            print(
                'est obs prob (0.0/0.25/0.5/0.75/1.0 quantile): {0:.2f}/{1:.2f}/{2:.2f}/{3:.2f}/{4:.2f}'
                .format(np.nanmin(prob_pred_train),
                        np.nanquantile(prob_pred_train, 0.25),
                        np.nanquantile(prob_pred_train, 0.5),
                        np.nanquantile(prob_pred_train, 0.75),
                        np.nanmax(prob_pred_train)))
            print(
                'est obs logit (0.0/0.25/0.5/0.75/1.0 quantile): {0:.2f}/{1:.2f}/{2:.2f}/{3:.2f}/{4:.2f}'
                .format(np.nanmin(logit_pred_train),
                        np.nanquantile(logit_pred_train, 0.25),
                        np.nanquantile(logit_pred_train, 0.5),
                        np.nanquantile(logit_pred_train, 0.75),
                        np.nanmax(logit_pred_train)))
            if False:
                fig, ax = plt.subplots(nrows=2,
                                    ncols=2,
                                    figsize=(5 * 2, 4 * 2))
                sns.scatterplot(x='X1',
                                y='X2',
                                hue='dropout_prob',
                                data=prob_pred_df,
                                ax=ax[0, 0],
                                s=50)
                ax[0, 0].set_title(f'true dropout prob')
                ax[0, 0].legend(bbox_to_anchor=(1.4, 1), title="prob")
                sns.scatterplot(x='X1',
                                y='X2',
                                hue='dropout_prob_est',
                                data=prob_pred_df,
                                ax=ax[0, 1],
                                s=50)
                if missing_mechanism == 'mnar':
                    ax[0, 1].set_title(
                        f'est dropout prob, L={L}, gamma_hat={round(mnar_clf.gamma_hat[0],2)}, MSE={round(train_mse,3)}'
                    )
                else:
                    ax[0, 1].set_title(f'est dropout prob, MSE={round(train_mse,3)}')                    
                ax[0, 1].legend(bbox_to_anchor=(1.4, 1), title="prob")
                sns.histplot(data=prob_pred_df,
                            x='dropout_prob',
                            bins=25,
                            ax=ax[1, 0])
                sns.histplot(data=prob_pred_df,
                            x='dropout_prob_est',
                            bins=25,
                            ax=ax[1, 1])
                plt.tight_layout()
                if missing_mechanism == 'mnar':
                    figure_name = f"prob_est_dist_missing{round(self.dropout_rate,1)}_instrument{L}_gamma{round(mnar_clf.gamma_hat[0],2)}.png"
                else:
                    figure_name = f"prob_est_dist_missing{round(self.dropout_rate,1)}_mar.png"
                plt.savefig(os.path.join(export_dir, figure_name))
                plt.close()

    def estimate_missing_prob(self,
                      missing_mechanism=None,
                      subsample_index=None):
        if missing_mechanism is not None:
            missing_mechanism = self.missing_mechanism
        assert hasattr(self, 'fitted_dropout_model'
                       ), 'please run function train_dropout_model() first'
        self.propensity_pred = {}
        keys = self.masked_buffer.keys()
        if subsample_index is not None:
            keys = subsample_index
        dropout_prob_concat = []
        surv_prob_concat = []
        for i in keys:
            S_traj, A_traj, R_traj, _, surv_prob_traj, don_traj, dop_traj = self.masked_buffer[
                i][:7]
            if self.misc_buffer:
                mnar_nextobs_arr = self.misc_buffer[i].get(
                    'mnar_nextobs_arr', None)
                mnar_noninstrument_arr = self.misc_buffer[i].get(
                    'mnar_noninstrument_arr', None)
                mnar_instrument_arr = self.misc_buffer[i].get(
                    'mnar_instrument_arr', None)
                if mnar_nextobs_arr is not None and mnar_noninstrument_arr is not None and mnar_instrument_arr is not None:
                    assert mnar_nextobs_arr.shape[
                        0] == mnar_noninstrument_arr.shape[
                            0] == mnar_instrument_arr.shape[0]
            else:
                mnar_nextobs_arr = None
                mnar_noninstrument_arr = None
                mnar_instrument_arr = None

            if missing_mechanism == 'mar':
                traj_len = len(A_traj)
                obs = S_traj[self.dropout_obs_count_thres:traj_len]
                actions = A_traj[self.dropout_obs_count_thres:traj_len]
                # rewards = R_traj[:traj_len].reshape(-1, 1)
                rewards = np.append(0, R_traj[:(
                    traj_len -
                    1)])[self.dropout_obs_count_thres:traj_len].reshape(-1, 1)
                obs = self.fitted_dropout_model['scaler'].transform(obs)
                if self._include_reward:
                    X_mat = np.hstack([obs, rewards])
                else:
                    X_mat = obs
                # don't forget to cast dtype as float
                dropout_prob_pred = np.zeros_like(
                    don_traj[self.dropout_obs_count_thres:traj_len],
                    dtype=float)
                for a in range(self.num_actions):
                    if any(actions == a) == False:
                        continue
                    if isinstance(self.fitted_dropout_model[a], float):
                        dropout_prob_pred[actions == a] = self.fitted_dropout_model[a]
                    else:
                        dropout_prob_pred[actions == a] = self.fitted_dropout_model[a].predict_proba(
                                    X=X_mat[actions == a])[:, 1]
            elif missing_mechanism == 'mnar':
                traj_len = A_traj.shape[0]
                obs = S_traj[self.dropout_obs_count_thres:traj_len]
                actions = A_traj[self.dropout_obs_count_thres:traj_len]
                if len(S_traj) == traj_len:
                    next_obs = np.vstack([
                        S_traj[self.dropout_obs_count_thres + 1:],
                        [[np.nan] * self.state_dim]
                    ])
                else:
                    next_obs = S_traj[self.dropout_obs_count_thres + 1:]
                obs = self.fitted_dropout_model['scaler'].transform(obs)
                next_obs = self.fitted_dropout_model['scaler'].transform(
                    next_obs)
                orig_next_obs = self.fitted_dropout_model[
                    'scaler'].inverse_transform(
                        next_obs)  # transform to its original scale
                rewards = R_traj[self.
                                 dropout_obs_count_thres:traj_len].reshape(
                                     -1, 1)
                if self._include_reward:
                    # use reward as outcome
                    y_arr = rewards
                    y_dim = 1
                elif mnar_nextobs_arr is None:
                    if self._mnar_y_transform is not None:
                        y_arr = self._mnar_y_transform(
                            np.hstack([orig_next_obs, rewards]))
                    # elif self._include_reward:
                    #     y_arr = rewards
                    else:
                        y_arr = next_obs
                    if len(y_arr.shape) == 1 or (len(y_arr.shape) == 2
                                                 and y_arr.shape[1] == 1):
                        y_dim = 1
                    else:
                        y_dim = y_arr.shape[1]
                    y_arr = y_arr.reshape(-1, y_dim)
                else:
                    y_arr = mnar_nextobs_arr[self.
                                             dropout_obs_count_thres:traj_len]
                    if len(y_arr.shape) == 1:
                        y_dim = 1
                    else:
                        y_dim = y_arr.shape[1]
                    y_arr = y_arr.reshape(-1, y_dim)
                if self._scale_obs:
                    assert hasattr(self, '_y_arr_scaler'), 'scaler for outcome not found'
                    y_arr = self._y_arr_scaler.transform(y_arr)
                if mnar_noninstrument_arr is None:
                    if not isinstance(self._instrument_var_index,
                                      np.ndarray) and not hasattr(
                                          self._instrument_var_index,
                                          '__iter__'):
                        non_instrument_index = [
                            i for i in range(obs.shape[1])
                            if i != self._instrument_var_index
                        ]
                    else:
                        non_instrument_index = [
                            i for i in range(obs.shape[1])
                            if i not in set(self._instrument_var_index)
                        ]
                    u_dim = len(non_instrument_index)
                    u_arr = obs[:,
                                non_instrument_index].reshape(len(obs), u_dim)
                else:
                    u_arr = mnar_noninstrument_arr[
                        self.dropout_obs_count_thres:traj_len]
                    if len(u_arr.shape) == 1:
                        u_dim = 1
                    else:
                        u_dim = u_arr.shape[1]
                    u_arr = u_arr.reshape(-1, u_dim)
                if self._scale_obs:
                    assert hasattr(self, '_u_arr_scaler'), 'scaler for non-instrumental variables is not found'
                    u_arr = self._u_arr_scaler.transform(u_arr)
                if mnar_instrument_arr is None:
                    z_arr = obs[:, self._instrument_var_index]
                    z_arr = np.digitize(z_arr, bins=self.instrument_disc_bins)
                else:
                    z_arr = mnar_instrument_arr[
                        self.dropout_obs_count_thres:traj_len].reshape(-1)
                    if min(z_arr) <= 0:
                        z_arr -= min(z_arr) - 1

                mnar_clf = self.fitted_dropout_model['model']
                L = mnar_clf.L
                dropout_prob_pred = 1 - \
                    mnar_clf.predict_proba(u=u_arr, z=z_arr, y=y_arr)

            dropout_prob_pred = np.append(np.array([0] * self.dropout_obs_count_thres),
                                 dropout_prob_pred)

            surv_prob_pred = (1 - dropout_prob_pred).cumprod()
            self.propensity_pred[i] = [dropout_prob_pred, surv_prob_pred]
            dropout_prob_concat.append(dropout_prob_pred)
            surv_prob_concat.append(surv_prob_pred)

        dropout_prob_concat = np.concatenate(dropout_prob_concat)
        surv_prob_concat = np.concatenate(surv_prob_concat)

    def estimate_behavior_policy(
            self,
            model_type="linear",
            train_ratio=0.8,
            export_dir=None,
            pkl_filename="behavior_policy_model.pkl",
            seed=None,
            verbose=True,
            use_holdout_data=True,
            **kwargs):
        """
        Estimate behavior policy.

        Args:
            model_type (str): select from "linear" and "rf"
            train_ratio (float): proportion of training set
            export_dir (str): directory to export model
            pkl_filename (str): filename for trained model
            seed (int): random seed to split data
            verbose (bool): if True, output intermediate outcome
            kwargs (dict): passed to model
        """
        action_list, f_list = [], []
        if export_dir is not None and pkl_filename is not None:
            self.behavior_model_filename = os.path.join(
                export_dir, pkl_filename)
            pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
        else:
            self.behavior_model_filename = None
        if use_holdout_data:
            training_buffer = self.holdout_buffer
        else:
            training_buffer = self.masked_buffer
        keys = training_buffer.keys()
        for k in keys:
            S_traj, A_traj, reward_traj = training_buffer[k][:3]
            T = A_traj.shape[0]
            action_list.append(A_traj.reshape(-1, 1))
            X_mat = self.Q(S=S_traj, A=A_traj, predictor=True)
            f_list.append(X_mat)
        actions = np.vstack(action_list).squeeze()
        X, y = np.vstack(f_list), actions
        data = {}
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=1 -
                                                            train_ratio,
                                                            random_state=seed)
        data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        if model_type.lower() == "linear":
            logit_reg = LogisticRegression(fit_intercept=True,
                                           max_iter=1000,
                                           **kwargs)
            logit_reg.fit(X=data['X_train'],
                          y=data['y_train'],
                          sample_weight=None)
            test_acc = accuracy_score(
                y_true=data['y_test'],
                y_pred=logit_reg.predict(X=data['X_test']))
            test_f1 = f1_score(y_true=data['y_test'],
                               y_pred=logit_reg.predict(X=data['X_test']),
                               average='macro')
            if verbose:
                print(f'Accuracy on test set: {test_acc}')
                print(f'F1 score on test set: {test_f1}')
            try:
                test_auroc = roc_auc_score(
                    y_true=data['y_test'],
                    y_score=logit_reg.predict_proba(X=data['X_test']),
                    multi_class='ovr')
                if verbose:
                    print(f'AUROC on test set: {test_auroc}')
            except:
                pass
            self.fitted_behavior_model = logit_reg
            del logit_reg
            _ = gc.collect()
        elif model_type.lower() == "rf":
            rf_clf = RandomForestClassifier(random_state=seed, **kwargs)
            rf_clf.fit(X=data['X_train'], y=data['y_train'])
            test_acc = accuracy_score(y_true=data['y_test'],
                                      y_pred=rf_clf.predict(X=data['X_test']))
            test_f1 = f1_score(y_true=data['y_test'],
                               y_pred=rf_clf.predict(X=data['X_test']),
                               average='macro')
            if verbose:
                print(f'Accuracy on test set: {test_acc}')
                print(f'F1 score on test set: {test_f1}')
            try:
                test_auroc = roc_auc_score(
                    y_true=data['y_test'],
                    y_score=rf_clf.predict_proba(X=data['X_test']),
                    multi_class='ovr')
                if verbose:
                    print(f'AUROC on test set: {test_auroc}')
            except:
                pass
            self.fitted_behavior_model = rf_clf
            del rf_clf  # rf model tends to take up a large amount of memory
            _ = gc.collect()
        else:
            raise NotImplementedError
        
        if self.behavior_model_filename is not None:
            joblib.dump(value=self.fitted_behavior_model,
                        filename=self.behavior_model_filename,
                        compress=3)

    def evaluate_pointwise_Q(
        self,
        policy,
        eval_size=20,
        eval_horizon=None,
        pointwise_eval_size=20,
        seed=None,
        S_inits=None,
        S_inits_kwargs={}
        ):
        """Evaluate pointwise Q and value

        Args:
            policy (callable): target policy to be evaluated
            eval_size (int): number of initial states to evaluate the policy
            eval_horizon (int): horizon of Monte Carlo approxiamation
            pointwise_eval_size (int): Monte Carlo size to estimate point-wise value
            seed (int): random seed passed to gen_batch_trajs()
            S_inits (np.ndarray): initial states to evaluate the policy
            S_inits_kwargs (dict): additional kwargs passed to env.reset()

        Returns:
            S_inits (np.ndarray): array of initial states to evaluate the policy
            true_value_avg (list): corresponding point-wise V(s)
            true_Q_dict (dict): corresponding point-wise Q(s,a), key is the action name
        """
        if S_inits is not None:
            if len(S_inits) == 1:
                S_inits = np.expand_dims(S_inits, axis=0)
            eval_size = len(S_inits)
        elif self.vectorized_env:
            S_inits = np.vstack([
                self.eval_env.single_observation_space.sample()
                for _ in range(eval_size)
            ])

        if self.vectorized_env:
            total_eval_size = pointwise_eval_size * eval_size
            if S_inits is not None:
                S_inits = np.repeat(S_inits,
                                    repeats=pointwise_eval_size,
                                    axis=0)
            old_num_envs = self.eval_env.num_envs
            old_horizon = self.eval_env.T
            # reset num_envs
            if eval_horizon is not None:
                self.eval_env.T = eval_horizon
            self.eval_env.num_envs = total_eval_size
            self.eval_env.observation_space = batch_space(
                self.eval_env.single_observation_space,
                n=self.eval_env.num_envs)
            self.eval_env.action_space = Tuple(
                (self.eval_env.single_action_space, ) * self.eval_env.num_envs)

            print('eval V: generate trajectories')
            trajectories = self.gen_batch_trajs(
                policy=policy,
                seed=seed,
                S_inits=S_inits if S_inits is not None else None,
                A_inits=None,
                burn_in=0,
                evaluation=True)
            rewards_history = self.eval_env.rewards_history
            print('eval V: calculate value')
            true_value = np.matmul(
                rewards_history,
                self.gamma**np.arange(start=0,
                                      stop=rewards_history.shape[1]).reshape(
                                          -1, 1))  # (total_eval_size, 1)
            true_value = true_value.reshape(eval_size, pointwise_eval_size)
            true_value_avg = true_value.mean(axis=1).tolist()
            self.eval_env.reset()
            del rewards_history
            _ = gc.collect()
            # get true Q for each action
            true_Q_dict = {}
            for a in range(self.eval_env.single_action_space.n):
                print('eval Q: generate trajectories')
                A_inits = np.repeat(a, repeats=len(S_inits))
                batch = self.gen_batch_trajs(
                    policy=policy,
                    seed=seed,
                    S_inits=S_inits if S_inits is not None else None,
                    A_inits=A_inits,
                    burn_in=0,
                    evaluation=True)
                rewards_history = self.eval_env.rewards_history * \
                    self.eval_env.states_history_mask[:,
                                                      :self.eval_env.rewards_history.shape[1]]
                print('eval Q: calculate value')
                true_Q = np.matmul(
                    rewards_history,
                    self.gamma**np.arange(
                        start=0, stop=rewards_history.shape[1]).reshape(-1, 1))
                true_Q = true_Q.reshape(eval_size, pointwise_eval_size)
                true_Q_avg = true_Q.mean(axis=1).tolist()
                true_Q_dict[a] = true_Q_avg

            # recover num_envs
            self.eval_env.T = old_horizon
            self.eval_env.num_envs = old_num_envs
            self.eval_env.observation_space = batch_space(
                self.eval_env.single_observation_space,
                n=self.eval_env.num_envs)
            self.eval_env.action_space = Tuple(
                (self.eval_env.single_action_space, ) * self.eval_env.num_envs)
            del trajectories, rewards_history
            _ = gc.collect()
            return S_inits[::pointwise_eval_size], true_value_avg, true_Q_dict
        else:
            old_horizon = self.eval_env.T
            if eval_horizon is not None:
                self.eval_env.T = eval_horizon
            print('eval_V: generate trajectories')
            self.eval_env.T = old_horizon
            true_value_list = []
            if S_inits is None and S_inits_kwargs:
                S_inits_sample = []
            for i in range(eval_size):
                rewards_histories_list = []
                for j in range(pointwise_eval_size):
                    traj = self.gen_single_traj(
                        policy=policy,
                        seed=None,
                        S_init=S_inits[i] if S_inits is not None else None,
                        S_init_kwargs={
                            k: v[i]
                            for k, v in S_inits_kwargs.items()
                        } if S_inits_kwargs else {},
                        A_init=None,
                        burn_in=0,
                        evaluation=True)
                    rewards_history = self.eval_env.rewards_history
                    rewards_histories_list.append(rewards_history)
                rewards_histories_arr = np.array(rewards_histories_list) # (pointwise_eval_size, eval_T)
                true_value = np.matmul(
                    rewards_histories_arr,
                    self.gamma**np.arange(
                        start=0, stop=rewards_histories_arr.shape[1]).reshape(
                            -1, 1)).reshape(-1)  # (pointwise_eval_size,)
                true_value_list.append(true_value)
                if S_inits is None and S_inits_kwargs:
                    S_inits_sample.append(traj[0][0])
            true_value_arr = np.array(true_value_list) # (eval_size,pointwise_eval_size)
            if S_inits is None and S_inits_kwargs:
                S_inits_sample = np.array(S_inits_sample)
                print(f'S_inits_sample: {S_inits_sample}')
            true_value_avg = true_value_arr.mean(
                axis=1).tolist()  # (eval_size,)
            self.eval_env.reset()
            del rewards_history, rewards_histories_list, rewards_histories_arr
            _ = gc.collect()
            # get true Q for each action
            true_Q_dict = {}
            for a in range(self.num_actions):
                print('eval_Q: generate trajectories')
                true_Q_list = []
                for i in range(eval_size):
                    rewards_histories_list = []
                    for j in range(pointwise_eval_size):
                        traj = self.gen_single_traj(
                            policy=policy,
                            seed=None,  # seed
                            S_init=S_inits[i] if S_inits is not None else None,
                            A_init=a,
                            burn_in=0,
                            evaluation=True)
                        rewards_history = self.eval_env.rewards_history
                        rewards_histories_list.append(rewards_history)
                    rewards_histories_arr = np.array(rewards_histories_list) # (pointwise_eval_size, eval_T)
                    true_Q = np.matmul(
                        rewards_histories_arr,
                        self.gamma**np.arange(
                            start=0,
                            stop=rewards_histories_arr.shape[1]).reshape(
                                -1))  # (pointwise_eval_size,)
                    true_Q_list.append(true_Q)
                true_Q_arr = np.array(true_Q_list) # (eval_size,pointwise_eval_size)
                true_Q_avg = true_Q_arr.mean(axis=1).tolist()  # (eval_size,)
                true_Q_dict[a] = true_Q_avg
            self.eval_env.T = old_horizon
            del rewards_history, rewards_histories_list, rewards_histories_arr
            _ = gc.collect()
            if S_inits is None and S_inits_kwargs:
                S_inits = S_inits_sample
            return S_inits, true_value_avg, true_Q_dict

    def get_target_value(self, target_policy, S_inits=None, MC_size=None):
        """Wrapper function of V_int
        
        Args:
            target_policy (callable): target policy to be evaluated.
            S_inits (np.ndarray): initial states for policy evaluation. If both S_inits and MC_size are not 
                specified, use initial observations from data.
            MC_size (int): sample size for Monte Carlo approximation.

        Returns:
            est_V (float): integrated value of target policy
        """
        raise NotImplementedError

