import numpy as np
import pandas as pd
import os
import gc
import pickle
import joblib
import pathlib
import time
from collections import Counter
from functools import reduce, partial
from itertools import product
from scipy.stats import norm
from scipy.interpolate import BSpline
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
# RL environment
from gym.spaces import Tuple
from gym.vector.utils.spaces import batch_space
# visualization
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    normcdf,
    iden,
    MinMaxScaler,
    MLPModule,
    ExpoTiltingClassifierMNAR
)
from base import SimulationBase


class FittedQEval(SimulationBase):

    def __init__(self,
                 env=None,
                 n=500,
                 horizon=None,
                 scale="NormCdf",
                 product_tensor=True,
                 discount=0.8,
                 eval_env=None,
                 basis_scale_factor=1.):
        """
        Args:
            env (gym.Env): dynamic environment
            n (int): the number of subjects (trajectories)
            horizon (int): the maximum length of trajectories
            scale (str): scaler to transform state features onto [0,1], 
                select from "NormCdf", "Identity", "MinMax", or a path to a fitted scaler
            product_tensor (bool): if True, use product tensor to construct basis
            discount (float): discount factor
            eval_env (gym.Env): dynamic environment to evaluate the policy, if not specified, use env
            basis_scale_factor (float): a multiplier to basis in order to avoid extremely small value
        """

        super().__init__(
            env=env, 
            n=n, 
            horizon=horizon, 
            discount=discount, 
            eval_env=eval_env
        )
        
        # scaler to transform state features onto [0,1]
        if scale == "NormCdf":
            self.scaler = normcdf()
        elif scale == "Identity":
            self.scaler = iden()
        elif scale == "MinMax":
            self.scaler = MinMaxScaler(
                min_val=self.env.low,
                max_val=self.env.high) if env is not None else MinMaxScaler()
        else:
            # a path to a fitted scaler
            assert os.path.exists(scale)
            with open(scale, 'rb') as f:
                self.scaler = pickle.load(f)
        
        self.para_dim = None # the dimension of parameter built in basis spline
        self.product_tensor = product_tensor
        self.basis_scale_factor = basis_scale_factor
        self.bspline = None
        self.knot = None # quantile knots for basis spline

    def export_buffer(self, eval_Q=False):
        """Convert unscaled trajectories in self.masked_buffer to a dataframe.

        Args:
            eval_Q (bool): if True, also output the estimated Q-value
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
            if eval_Q:
                Q_list = [
                    self.Q(S=S, A=A, predictor=False, double=True)
                    for S, A in zip(nonterminal_state_list, action_list)
                ]  # same length as action_list
            if len(action_list) < nrows:
                tmp['action'] = np.append(action_list, [None])
                tmp['reward'] = np.append(self.masked_buffer[k][2], [None])
                tmp['surv_prob'] = np.append(self.masked_buffer[k][4], [None])
                tmp['dropout_prob'] = np.append(self.masked_buffer[k][6],
                                                [None])
                if eval_Q:
                    tmp['est_Q'] = np.append(Q_list, [None])
            else:
                tmp['action'] = action_list
                tmp['reward'] = self.masked_buffer[k][2]
                tmp['surv_prob'] = self.masked_buffer[k][4]
                tmp['dropout_prob'] = self.masked_buffer[k][6]
                if eval_Q:
                    tmp['est_Q'] = Q_list
            df = df.append(tmp)
        df = df.reset_index(drop=True)
        return df

    def B_spline(self, L=10, d=3, knots=None):
        """
        Construct B-Spline basis.

        Args:
            L (int): number of basis function (degree of freedom)
            d (int): B-spline degree
            knots (str or np.ndarray): location of knots
        """
        obs_concat = []
        if self.masked_buffer:
            for i in self.masked_buffer.keys():
                obs_concat.extend(self.masked_buffer[i][0])
        else:
            for i in self.holdout_buffer.keys():
                obs_concat.extend(self.holdout_buffer[i][0])        
        obs_concat = np.array(obs_concat)
        if not hasattr(self.scaler, 'data_min_') or not hasattr(
                self.scaler, 'data_max_') or np.min(
                    self.scaler.data_min_) == -np.inf or np.max(
                        self.scaler.data_max_) == np.inf:
            self.scaler.fit(obs_concat)
        scaled_obs_concat = self.scaler.transform(obs_concat)

        if isinstance(knots, str) and knots == 'equivdist':
            upper = scaled_obs_concat.max(axis=0)
            lower = scaled_obs_concat.min(axis=0)
            knots = np.linspace(start=lower - d * (upper - lower) / (L - d),
                                stop=upper + d * (upper - lower) / (L - d),
                                num=L + d + 1)
            self.knot = knots
        elif isinstance(knots, np.ndarray):
            assert len(knots) == L + d + 1
            if len(knots.shape) == 1:
                knots = np.tile(knots.reshape(-1, 1), reps=(1, self.state_dim))
            self.knot = knots
        elif isinstance(knots, str) and knots == 'quantile':
            base_knots = np.quantile(a=scaled_obs_concat,
                                     q=np.linspace(0, 1, L - d + 1),
                                     axis=0)  # (L+1, state_dim)
            upper = base_knots.max(axis=0)
            lower = base_knots.min(axis=0)
            left_extrapo = np.linspace(lower - d * (upper - lower) / (L - d),
                                       lower,
                                       num=d + 1)[:-1]
            right_extrapo = np.linspace(upper,
                                        upper + d * (upper - lower) / (L - d),
                                        num=d + 1)[1:]
            self.knot = np.concatenate(
                [left_extrapo, base_knots, right_extrapo])
        else:
            raise NotImplementedError

        self.bspline = []

        self.para_dim = 1 if self.product_tensor else 0
        for i in range(self.state_dim):
            tmp = []
            for j in range(L):
                cof = [0] * L
                cof[j] = 1
                spf = BSpline(t=self.knot.T[i], c=cof, k=d, extrapolate=True)
                tmp.append(spf)
            self.bspline.append(tmp)
            if self.product_tensor:
                self.para_dim *= len(self.bspline[i])
            else:
                self.para_dim += len(self.bspline[i])
            print(
                "Building %d-th basis spline (total %d state dimemsion) which has %d basis "
                % (i, self.state_dim, len(self.bspline[i])))

        self.para = {}
        for i in range(self.num_actions):
            self.para[i] = np.random.normal(loc=0,
                                            scale=0.1,
                                            size=self.para_dim)
        self.para_2 = self.para.copy()  # for double Q-learning

    def _stretch_para(self):
        """Flatten parameters into a vector."""
        self.all_para = []
        for i in self.para.values():
            self.all_para.extend(i)
        self.all_para = np.array(self.all_para)

    ##################################
    ## calculate Q and value function 
    ##################################

    def _predictor(self, S):
        """
        Return value of basis functions given states and actions. 

        Args:
            S (np.ndarray): array of states, dimension (n, state_dim)

        Returns:  
            output (np.ndarray): array of basis values, dimension (n, para_dim)
        """
        S = np.array(S)  # (n,S_dim)
        if len(S.shape) == 1:
            S = np.expand_dims(S, axis=0)  # (n,S_dim)
        S = self.scaler.transform(S)
        if self.bspline:
            S = S.T  # (S_dim,n)
            if self.product_tensor:
                output = np.vstack(
                    list(
                        map(partial(np.prod, axis=0),
                            (product(*[
                                np.array([func(s) for func in f])
                                for f, s in zip(self.bspline, S)
                            ],
                                     repeat=1)))))  # ((L-d)^S_dim, n)
                output *= self.basis_scale_factor  #
            else:
                output = np.concatenate([
                    np.array([func(s) for func in f])
                    for f, s in zip(self.bspline, S)
                ])  # ((L-d)*S_dim, n)
            output = output.T # (n, para_dim)
            output *= self.basis_scale_factor
            return output
        else:
            raise NotImplementedError

    def Q(self, S, A, predictor=False, double=False):
        """
        Return the value of Q-function given states and actions. 

        Args:
            S (np.ndarray) : array of states, dimension (n, state_dim)
            A (np.ndarray) : array of actions, dimension (n, )
            predictor (bool) : if True, return the value of each basis function
            double (bool) : if True, use double Q-functions

        Returns:
            Q_est (np.ndarray): Q-function values, dimension (n, )
        """
        S = np.array(S)  # (n,S_dim)
        if len(S.shape) == 1:
            S = np.expand_dims(S, axis=0)  # (n,S_dim)
            scaler_output = True
        else:
            scaler_output = False

        output = self._predictor(S=S)
        if predictor:
            return output
        
        A = np.array(A).astype(np.int8).squeeze()  # (n,)
        Q_est = np.zeros(shape=(len(output), ))
        if double:
            for a in range(self.num_actions):
                Q_est += np.matmul(output, self.para_2[a]) * (a == A)
        else:
            for a in range(self.num_actions):
                Q_est = Q_est + np.matmul(output, self.para[a]) * (a == A)
        return Q_est if not scaler_output else Q_est.item()

    def V(self, S, policy):
        """
        Return the value of states under some policy. 

        Args:
            S (np.ndarray) : array of states, dimension (n,state_dim)
            policy (callable) : the policy to be evaluated

        Returns:
            V_est (np.ndarray): array of values, dimension (n,)
        """
        S = np.array(S)
        if len(S.shape) == 1:
            S = np.expand_dims(S, axis=0)  # (n,S_dim)
            scaler_output = True
        else:
            scaler_output = False
        U_mat = self._U(S=S, policy=policy)
        est_beta = []
        for i in self.para.values():
            est_beta.extend(i)
        est_beta = np.array(est_beta)
        V_est = np.matmul(U_mat, est_beta.reshape(-1, 1))
        return V_est if not scaler_output else V_est.item()

    def V_int(self, policy, MC_size=None, S_inits=None):
        """
        Return the integrated value function under a given policy. 

        Args:
            policy (callable) : the policy to be evaluated
            MC_size (int) : Monte Carlo size to calculate the integral
            S_inits (np.ndarray) : initial states to evaluate the values

        Returns:
            V_int_est (float): integrated value
        """
        if S_inits is None:
            if MC_size is None:
                S_inits = self._initial_obs
            else:
                S_inits = self.sample_initial_states(size=MC_size, from_data=True)
        return np.mean(self.V(policy=policy, S=S_inits))

    def update_op_policy(
            self,
            policy,
            ipw=False,
            estimate_missing_prob=False,
            drop_last_TD=True,
            prob_lbound=1e-3):
        """Update parameters to estimate Q-function under a fixed policy Q(pi)

        Args:
            policy (callable): the target policy to evaluate
            ipw (bool): if True, use inverse weighting to adjust for bias induced by informative dropout
            estimate_missing_prob (bool): if True, use estimated probability. Otherwise, use ground truth (only for simulation)
            drop_last_TD (bool): if True, drop the temporal difference error at the terminal step
            prob_lbound (float): lower bound of survival probability to avoid explosion of inverse weight
        """
        if estimate_missing_prob:
            assert hasattr(
                self, 'propensity_pred'
            ), 'please call function self.estimate_missing_prob() first'
        target_dict, f_dict, ipw_dict = {}, {}, {}
        total_T = 0
        obs_list, next_obs_list = [], []
        action_list, reward_list = [], []
        surv_prob_list, inverse_wt_list = [], []
        max_inverse_wt, min_inverse_wt = 0, 100
        # obtain predictor and reponse
        for i in self.masked_buffer.keys():
            S_traj = self.masked_buffer[i][0]
            A_traj = self.masked_buffer[i][1]
            reward_traj = self.masked_buffer[i][2]
            survival_prob = self.masked_buffer[i][4]
            if len(S_traj) < 2:
                continue
            T = S_traj.shape[0] - 1
            total_T += T
            obs_list.append(S_traj[:-1])
            next_obs_list.append(S_traj[1:])
            action_list.append(A_traj[:T].reshape(-1, 1))
            reward_list.append(reward_traj[:T].reshape(-1, 1))
            if estimate_missing_prob:
                prob = self.propensity_pred[i][1].reshape(-1, 1)
                assert len(prob) == T
            else:
                prob = survival_prob[:T].reshape(-1, 1)
            surv_prob_list.append(prob)
            if not ipw:
                inverse_wt = np.ones(shape=(T, 1))
            else:
                inverse_wt = 1 / np.clip(a=prob, a_min=prob_lbound,
                                         a_max=1)  # bound ipw
            inverse_wt_list.append(inverse_wt)
        obs = np.vstack(obs_list)  # (total_T, S_dim)
        next_obs = np.vstack(next_obs_list)  # (total_T, S_dim)
        actions = np.vstack(action_list)  # (total_T, 1)
        rewards = np.vstack(reward_list)  # (total_T, 1)
        surv_probs = np.vstack(surv_prob_list)  # (total_T, 1)
        inverse_wts = np.vstack(inverse_wt_list)  # (total_T, 1)
        policy_next_actions = policy(next_obs)  # (total_T, *)
        is_stochastic = policy_next_actions.shape[1] > 1
        f = self._predictor(S=obs)  # (total_T, para_dim)
        if not is_stochastic:
            targets = rewards + self.gamma * self.Q(
                S=next_obs, A=policy_next_actions).reshape(-1, 1)
        else:
            targets = rewards
            for a in range(self.num_actions):
                targets += self.gamma * policy_next_actions[:, a].reshape(
                    -1, 1) * self.Q(S=next_obs,
                                    A=np.tile(a, reps=(total_T, 1))).reshape(
                                        -1, 1)
        for a in range(self.num_actions):
            target_dict[a] = targets[(actions == a).squeeze()].squeeze()
            f_dict[a] = f[(actions == a).squeeze()]
            ipw_dict[a] = inverse_wts[(actions == a).squeeze()].squeeze()

        max_inverse_wt = np.max(inverse_wts)
        min_inverse_wt = np.min(inverse_wts)
        print('MaxInverseWeight')
        print(max_inverse_wt)
        print('MinInverseWeight')
        print(min_inverse_wt)

        self.para_2 = self.para.copy()
        for a in range(self.num_actions):
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X=np.array(f_dict[a]),
                    y=np.array(target_dict[a]),
                    sample_weight=np.array(ipw_dict[a]))
            self.para[a] = reg.coef_

    def opt_policy(self, S, epsilon=0.0, double=True):
        """
        Get the optimal action based on Q-function. 

        Args:
            S (np.ndarray): array of states, dimension (n, state_dim)
            epsilon (float): the probability to use random policy for exploration

        Returns:
            opt_policy (np.ndarray): array of selected optimal actions, dimension (n,)
        """
        S = np.array(S)  # (n,S_dim)
        if len(S.shape) == 1:
            S = np.expand_dims(S, axis=0)  # (n,S_dim)
            scaler_output = True
        else:
            scaler_output = False
        Q_vals = np.vstack([
            self.Q(S=S,
                   A=np.tile(i, reps=S.shape[0]),
                   predictor=False,
                   double=double) for i in range(self.num_actions)
        ]).T
        opt_action = np.argmax(Q_vals, axis=1)
        random_action = np.random.choice(self.num_actions, S.shape[0])
        opt_policy = np.where(
            np.random.uniform(0, 1, size=S.shape[0]) < epsilon, random_action,
            opt_action)
        return opt_policy.item() if scaler_output else opt_policy

    ############################
    ## inference on beta 
    ############################
    def _Xi(self, S, A):
        """
        Return Xi given states and actions. 

        Args:
            S (np.ndarray) : An array of states, dimension (n, state_dim)
            A (np.ndarray) : An array of actions, dimension (n, )

        Returns:
            xi (np.ndarray): An array of Xi values, dimension (n, para_dim * num_actions)
        """
        S = np.array(S)  # (n, S_dim)
        if len(S.shape) == 1:
            S = np.expand_dims(S, axis=0)  # (n, S_dim)
        nrows = S.shape[0]
        predictor = self._predictor(S=S)  # (n, para_dim)

        A = np.array(A).astype(np.int8).reshape(nrows)  # (n,)
        xi = np.tile(predictor,
                     reps=self.num_actions)  # (n, para_dim * num_actions)
        action_mask = np.repeat(np.eye(self.num_actions)[A],
                                repeats=self.para_dim,
                                axis=1)  # (n, para_dim * num_actions)
        return xi * action_mask

    def _U(self, S, policy):
        """
        Return U given states and policy. 

        Args:
            S (np.ndarray) : array of states, dimension (n, state_dim)
            policy (callable) : policy function that outputs actions with dimension (n, *)

        Returns
            U (np.ndarray): array of U values, dimension (n, para_dim * num_actions)
        """
        S = np.array(S)
        if len(S.shape) == 1:
            S = np.expand_dims(S, axis=0)  # (n,S_dim)
        if self.vectorized_env:
            action = policy(S)
        else:
            action = np.array([policy(s) for s in S])
        is_stochastic = action.shape[1] > 1
        if not is_stochastic:
            return self._Xi(S=S, A=action)
        else:
            predictor = self._predictor(S=S)
            U = np.tile(predictor,
                        reps=self.num_actions)  # (n, para_dim * num_actions)
            policy_mask = np.repeat(action, repeats=self.para_dim,
                                    axis=1)  # (n, para_dim * num_actions)
            return U * policy_mask  # (n, para_dim * num_actions)

    def _beta_hat(self,
                  policy,
                  block=False,
                  ipw=False,
                  estimate_missing_prob=False,
                  weight_curr_step=True,
                  prob_lbound=1e-3,
                  ridge_factor=0.,
                  grid_search=False,
                  verbose=True,
                  subsample_index=None):
        """
        Calculate beta_hat. 

        Args:
            policy (callable): the target policy to evaluate that outputs actions with dimension (n, *)
            block (bool): if True, use data in the next block
            ipw (bool): if True, use inverse probability weighting to adjust for missing data
            estimate_missing_prob (bool): if True, use estimated missing probability, otherwise, use ground truth (only for simulation)
            weight_curr_step (bool): if True, use the probability of dropout at current step
            prob_lbound (float): lower bound of dropout/survival probability to avoid extreme inverse weight
            ridge_factor (float): ridge penalty parameter
            grid_search (bool): if True, use grid search to select the optimal ridge_factor
            verbose (bool): if True, print intermediate tracking
            subsample_index (list): indices to subsample data for bootstrapping

        Returns:
            est_beta (np.ndarray): array of beta_hat, dimension (para_dim*num_actions,1)
        """
        if hasattr(self, 'est_beta'):
            return self.est_beta.copy()
        if not ipw:
            estimate_missing_prob = False
        if estimate_missing_prob:
            assert hasattr(
                self, 'propensity_pred'
            ), 'please call function self.estimate_missing_prob() first'
        mat1 = np.zeros((self.para_dim * self.num_actions,
                         self.para_dim * self.num_actions))
        mat2 = np.zeros((self.para_dim * self.num_actions, 1))

        total_T = 0
        obs_list, next_obs_list = [], []
        action_list, reward_list = [], []
        surv_prob_list, inverse_wt_list = [], []
        dropout_prob_list = []
        max_inverse_wt = 1
        min_inverse_wt = 1 / prob_lbound if prob_lbound else 0
        # timewise_inverse_wt = np.zeros(shape=self.max_T - self.burn_in - 1)
        if block:
            training_buffer = self.next_block
        else:
            training_buffer = self.masked_buffer
        keys = training_buffer.keys()
        if subsample_index is not None:
            keys = subsample_index
        if verbose:
            print('Start calculating beta...')
        dropout_prob_concat = []
        for i in keys:
            S_traj = training_buffer[i][0]
            A_traj = training_buffer[i][1]
            reward_traj = training_buffer[i][2]
            dropout_next_traj = training_buffer[i][5]
            dop_traj = training_buffer[i][6]
            if len(S_traj) < 2:
                continue
            if estimate_missing_prob:
                survival_prob = self.propensity_pred[i][1]
                dropout_prob = self.propensity_pred[i][0]
                if training_buffer[i][4] is not None:
                    assert len(survival_prob) == len(training_buffer[i][4])
            else:
                survival_prob = training_buffer[i][4]
                dropout_prob = training_buffer[i][6]
            T = len(S_traj) - 1
            total_T += T
            obs_list.append(S_traj[:-1])
            next_obs_list.append(S_traj[1:])
            action_list.append(np.array(A_traj[:T]).reshape(-1, 1))
            reward_list.append(np.array(reward_traj[:T]).reshape(-1, 1))
            if survival_prob is not None:
                prob = np.array(survival_prob[:T]).reshape(-1, 1)
            if dropout_prob is not None:
                dropout_prob = np.array(dropout_prob[:T]).reshape(-1, 1)
            if dropout_prob is not None:
                dropout_prob_concat.append(dropout_prob)
            if not ipw:
                inverse_wt = np.ones(shape=(T, 1))
            else:
                if weight_curr_step:
                    inverse_wt = 1 / np.clip(a=1 - dropout_prob,
                                             a_min=prob_lbound,
                                             a_max=1)  # bound ipw
                else:
                    inverse_wt = 1 / np.clip(
                        a=prob, a_min=prob_lbound, a_max=1)  # bound ipw
            inverse_wt_list.append(inverse_wt)
            # timewise_inverse_wt += np.append(
            #     inverse_wt, [0] * (self.max_T - self.burn_in - 1 - T))
        obs = np.vstack(obs_list)  # (total_T, S_dim)
        next_obs = np.vstack(next_obs_list)  # (total_T, S_dim)
        actions = np.vstack(action_list)  # (total_T, 1)
        rewards = np.vstack(reward_list)  # (total_T, 1)
        inverse_wts = np.vstack(inverse_wt_list)  # (total_T, 1)
        inverse_wts_mat = np.diag(
            v=inverse_wts.squeeze()).astype(float)  # (total_T, total_T)
        Xi_mat = self._Xi(S=obs, A=actions)
        self._Xi_mat = Xi_mat
        # timewise_inverse_wt = timewise_inverse_wt / self.n
        if dropout_prob_concat:
            dropout_prob_concat = np.concatenate(dropout_prob_concat)
        U_mat = self._U(S=next_obs, policy=policy)
        mat1 = reduce(np.matmul,
                      [Xi_mat.T, inverse_wts_mat, Xi_mat - self.gamma * U_mat])
        mat2 = reduce(np.matmul, [Xi_mat.T, inverse_wts_mat, rewards])
        max_inverse_wt = np.max(inverse_wts)
        min_inverse_wt = np.min(inverse_wts)

        if verbose:
            print('MaxInverseWeight')
            print(max_inverse_wt)
            print('MinInverseWeight')
            print(min_inverse_wt)

        self.max_inverse_wt = max_inverse_wt
        self.total_T = total_T
        if ipw:
            self.total_T_ipw = self.n * (self.max_T - self.burn_in - 1)
        else:
            self.total_T_ipw = total_T

        self._stretch_para()
        self.residual = (mat2 - np.matmul(mat1, self.all_para.reshape(
            -1, 1))) / self.total_T_ipw

        if grid_search:
            ridge_params = np.power(10, np.arange(-9, 1).astype(float))
            best_gcv = np.inf
            for r in ridge_params:
                est_beta = np.matmul(
                    np.linalg.pinv(
                        np.diag([r] * mat1.shape[0]) +
                        mat1 / self.total_T_ipw), mat2 / self.total_T_ipw)
                U_mat, S_vec, Vh_mat = np.linalg.svd(mat1 / self.total_T_ipw)
                edof = np.sum(S_vec**2 / (S_vec**2 + r))
                n_ = mat2.shape[0]
                dev = np.mean(
                    ((mat2 / self.total_T_ipw) -
                     np.matmul(mat1 / self.total_T_ipw, est_beta))**2)

                gcv = (n_ * dev) / (n_ - 0.5 * edof)**2
                if gcv < best_gcv:
                    best_param = r
                    best_gcv = gcv
                    best_edof = edof

            self.Sigma_hat = mat1 / self.total_T_ipw
            self.inv_Sigma_hat = np.linalg.pinv(
                self.Sigma_hat +
                np.diag([best_param] * mat1.shape[0]))  # generalized inverse
            self.vector = mat2 / self.total_T_ipw
            self.est_beta = np.matmul(self.inv_Sigma_hat, self.vector)

            self._ridge_param_cv = {
                'best_param': best_param,
                'best_gcv': best_gcv,
                'best_edof': best_edof
            }
        else:
            self.Sigma_hat = np.diag(
                [ridge_factor] * mat1.shape[0]) + mat1 / self.total_T_ipw
            self.Sigma_hat = self.Sigma_hat.astype(float)
            self.vector = mat2 / self.total_T_ipw
            self.inv_Sigma_hat = np.linalg.pinv(self.Sigma_hat)
            self.est_beta = np.matmul(self.inv_Sigma_hat, self.vector)

        if verbose:
            print('Finish estimating beta!')

        self.residual = (mat2 -
                         np.matmul(mat1, self.est_beta)) / self.total_T_ipw

        del training_buffer
        self._stretch_para()
        return self.est_beta.copy()

    def _store_para(self, est_beta):
        """Store the estimated beta in self.para
        
        Args:
            est_beta (np.ndarray): vector of estimated beta
        """
        for i in range(self.num_actions):
            self.para[i] = self.est_beta[i * self.para_dim:(i + 1) *
                                         self.para_dim].reshape(-1)

    def _Omega_hat(self,
                   policy,
                   block=False,
                   ipw=False,
                   estimate_missing_prob=False,
                   weight_curr_step=True,
                   prob_lbound=1e-3,
                   ridge_factor=1e-9,
                   grid_search=False,
                   verbose=True,
                   subsample_index=None):
        """
        Calculate Omega_hat. 

        Args:
            policy (callable): the target policy to evaluate that outputs actions with dimension (n, *)
            block (bool): if True, use data in the next block
            ipw (bool): if True, use inverse weighting to adjust for missing data
            estimate_missing_prob (bool): if True, use estimated missing probability, otherwise, use ground truth (only for simulation)
            weight_curr_step (bool): if True, use the probability of dropout at current step
            prob_lbound (float): lower bound of dropout/survival probability to avoid explosion inverse weight
            ridge_factor (float): ridge penalty parameter
            grid_search (bool): if True, use grid search to select the optimal ridge_factor
            verbose (bool): if True, print intermediate tracking
            subsample_index (list): indices to subsample data for bootstrapping

        Returns:
            Omega (np.ndarray): array of Omega_hat, dimension (para_dim * num_actions, para_dim * num_actions)
        """
        if not ipw:
            estimate_missing_prob = False
        if estimate_missing_prob:
            assert hasattr(
                self, 'propensity_pred'
            ), 'please call function self.estimate_missing_prob() first'
        self._beta_hat(policy=policy,
                       block=block,
                       ipw=ipw,
                       estimate_missing_prob=estimate_missing_prob,
                       weight_curr_step=weight_curr_step,
                       prob_lbound=prob_lbound,
                       ridge_factor=ridge_factor,
                       grid_search=grid_search,
                       verbose=verbose,
                       subsample_index=subsample_index)
        self._store_para(self.est_beta)
        total_T = 0
        if not block:
            training_buffer = self.masked_buffer
        else:
            training_buffer = self.next_block
        print('Start calculating Omega...')
        obs_list, next_obs_list = [], []
        action_list, reward_list = [], []
        surv_prob_list, inverse_wt_list = [], []
        for i in training_buffer.keys():
            S_traj = training_buffer[i][0]
            A_traj = training_buffer[i][1]
            reward_traj = training_buffer[i][2]
            survival_prob = training_buffer[i][4]
            dropout_prob = training_buffer[i][6]
            if len(S_traj) < 2:
                continue
            T = S_traj.shape[0] - 1
            total_T += S_traj.shape[0] - 1
            obs_list.append(S_traj[:-1])
            next_obs_list.append(S_traj[1:])
            action_list.append(A_traj[:T].reshape(-1, 1))
            reward_list.append(reward_traj[:T].reshape(-1, 1))
            if estimate_missing_prob:
                surv_prob = self.propensity_pred[i][1][:T].reshape(-1, 1)
                dropout_prob = self.propensity_pred[i][0][:T].reshape(-1, 1)
            else:
                surv_prob = survival_prob[:T].reshape(
                    -1, 1) if survival_prob is not None else None
                dropout_prob = dropout_prob[:T].reshape(
                    -1, 1) if dropout_prob is not None else None
            surv_prob_list.append(surv_prob)
            if not ipw:
                inverse_wt = np.ones(shape=(T, 1))
            else:
                if weight_curr_step:
                    inverse_wt = 1 / np.clip(a=1 - dropout_prob,
                                             a_min=prob_lbound,
                                             a_max=1)  # bound ipw
                else:
                    inverse_wt = 1 / np.clip(
                        a=surv_prob, a_min=prob_lbound, a_max=1)  # bound ipw
            inverse_wt_list.append(inverse_wt)

        obs = np.vstack(obs_list)  # (total_T, S_dim)
        next_obs = np.vstack(next_obs_list)  # (total_T, S_dim)
        actions = np.vstack(action_list)  # (total_T, 1)
        rewards = np.vstack(reward_list)  # (total_T, 1)
        surv_probs = np.vstack(surv_prob_list)  # (total_T, 1)
        inverse_wts = np.vstack(inverse_wt_list)  # (total_T, 1)
        inverse_wts_mat = np.diag(
            v=inverse_wts.squeeze()).astype(float)  # (total_T, total_T)
        Xi = self._Xi(S=obs, A=actions)  # (n, para_dim*num_actions)
        proj_td = np.matmul(
            inverse_wts_mat *
            (rewards +
             self.gamma * self.V(S=next_obs, policy=policy).reshape(-1, 1) -
             self.Q(S=obs, A=actions).reshape(-1, 1)), Xi
        )  # (n, para_dim*num_actions), np.sqrt(inverse_wts_mat) or inverse_wts_mat?
        output = np.matmul(
            proj_td.T, proj_td)  # (para_dim*num_actions, para_dim*num_actions)

        self.Omega = output / self.total_T_ipw

        print('Finish calculating Omega!')
        del training_buffer
        print(f'pseudo sample size: {inverse_wts.sum()}')
        return self.Omega.copy()

    #####################################
    ##  inference on point-wise value 
    #####################################

    def _sigma(self,
               S,
               policy,
               block=False,
               ipw=False,
               estimate_missing_prob=False,
               weight_curr_step=True,
               prob_lbound=1e-3,
               ridge_factor=1e-9,
               grid_search=False,
               verbose=True,
               subsample_index=None):
        """
        Calculate sigma.

        Args:
            S (np.ndarray) : array of states, dimension (n, state_dim)
            policy (callable): the target policy to evaluate that outputs actions with dimension (n, *)
            block (bool): if True, use data in the next block
            ipw (bool): if True, use inverse weighting to adjust for missing data
            estimate_missing_prob (bool): if True, use estimated missing probability, otherwise, use ground truth (only for simulation)
            weight_curr_step (bool): if True, use the probability of dropout at current step
            prob_lbound (float): lower bound of dropout/survival probability to avoid explosion inverse weight
            ridge_factor (float): ridge penalty parameter
            grid_search (bool): if True, use grid search to select the optimal ridge_factor
            verbose (bool): if True, print intermediate tracking
            subsample_index (list): indices to subsample data for bootstrapping

        Returns:
            sigma (np.ndarray): array of sigma, dimension (n,1)
        """
        if estimate_missing_prob:
            assert hasattr(
                self, 'propensity_pred'
            ), 'please call function self.estimate_missing_prob() first'
        S = np.array(S)  # (n,S_dim)
        if len(S.shape) == 1:
            S = np.expand_dims(S, axis=0)  # (n,S_dim)
            scaler_output = True
        else:
            scaler_output = False
        self._Omega_hat(policy=policy,
                        block=block,
                        ipw=ipw,
                        estimate_missing_prob=estimate_missing_prob,
                        weight_curr_step=weight_curr_step,
                        prob_lbound=prob_lbound,
                        ridge_factor=ridge_factor,
                        grid_search=grid_search,
                        verbose=verbose,
                        subsample_index=subsample_index)
        U_mat = self._U(S=S, policy=policy)  # (n, S_dim)
        sigma_sq_list = []
        for u in U_mat:
            sigma_sq = reduce(np.matmul, [
                u.reshape(1, -1), self.inv_Sigma_hat, self.Omega,
                self.inv_Sigma_hat.T,
                u.reshape(-1, 1)
            ])
            sigma_sq_list.extend(sigma_sq)
        sigma_sq_arr = np.array(sigma_sq_list)
        return sigma_sq_arr.item() if scaler_output else sigma_sq_arr

    def inference(self,
                  S,
                  policy,
                  alpha=0.05,
                  block=False,
                  ipw=False,
                  estimate_missing_prob=False,
                  weight_curr_step=True,
                  prob_lbound=1e-3,
                  ridge_factor=1e-9,
                  grid_search=False,
                  subsample_index=None,
                  verbose=True):
        """
        Calculate confidence interval for point-wise value. 

        Args:
            S (np.ndarray) : array of states, dimension (n, state_dim)
            policy (callable): the target policy to evaluate that outputs actions with dimension (n, *)
            alpha (float): significance level, default is 0.05
            block (bool): if True, use data in the next block
            ipw (bool): if True, use inverse weighting to adjust for missing data
            estimate_missing_prob (bool): if True, use estimated missing probability, otherwise, use ground truth (only for simulation)
            weight_curr_step (bool): if True, use the probability of dropout at current step
            prob_lbound (float): lower bound of dropout/survival probability to avoid explosion inverse weight
            ridge_factor (float): ridge penalty parameter
            grid_search (bool): if True, use grid search to select the optimal ridge_factor
            verbose (bool): if True, print intermediate tracking
            subsample_index (list): indices to subsample data for bootstrapping

        Returns:
            lower_bound (np.ndarray): lower bound of value estimation, dimension (n,1)
            upper_bound (np.ndarray): upper bound of value estimation, dimension (n,1)
        """
        if estimate_missing_prob:
            assert hasattr(
                self, 'propensity_pred'
            ), 'please call function self.estimate_missing_prob() first'
        S = np.array(S)  # (n,S_dim)
        if len(S.shape) == 1:
            S = np.expand_dims(S, axis=0)  # (n,S_dim)
            scaler_output = True
        else:
            scaler_output = False
        total_T = self.total_T_ipw  # self.total_T_ipw, self.total_T
        sigma_sq = self._sigma(S=S,
                               policy=policy,
                               block=block,
                               ipw=ipw,
                               estimate_missing_prob=estimate_missing_prob,
                               weight_curr_step=weight_curr_step,
                               prob_lbound=prob_lbound,
                               ridge_factor=ridge_factor,
                               grid_search=grid_search,
                               subsample_index=subsample_index,
                               verbose=verbose)  # estimate the beta
        V_est = self.V(S=S, policy=policy)
        if not scaler_output:
            V_est = V_est.reshape(-1, 1)
        lower_bound = V_est - norm.ppf(1 - alpha / 2) * (sigma_sq**
                                                         0.5) / (total_T**0.5)
        upper_bound = V_est + norm.ppf(1 - alpha / 2) * (sigma_sq**
                                                         0.5) / (total_T**0.5)
        if scaler_output:
            return lower_bound.item(), upper_bound.item()
        return lower_bound, upper_bound

    #######################################
    ##   inference on integrated value 
    #######################################

    def _sigma_int(
            self,
            policy,
            block=False,
            ipw=False,
            estimate_missing_prob=False,
            weight_curr_step=True,
            prob_lbound=1e-3,
            ridge_factor=1e-9,
            grid_search=False,
            MC_size=None,
            S_inits=None,
            verbose=True,
            subsample_index=None):
        """Get sigma_hat for integrated value.

        Args:
            policy (callable): target policy to be evaluated
            block (bool): if True, use the current block for policy evaluation. Otherwise, use all the data.
            ipw (bool): if True, use inverse probability weighting to adjust for missing data
            estimate_missing_prob (bool): if True, estimate the probability. Otherwise, use the true probability.
            weight_curr_step (bool): if True, wight by the inverse probability of observing the next state. 
                Otherwise, use survival probability. Default is True.
            prob_lbound (float): lower bound of probability to avoid extreme inverse weight
            ridge_factor (float): weight of ridge penalty
            grid_search (bool): if True, use grid search to find optimal ridge_factor
            MC_size (int): sample size of Monte Carlo approximation
            S_inits (np.ndarray): initial states for policy evaluation
            verbose (bool): if True, output intermediate results
            subsample_index (list): ids of subsample to estimate beta

        Returns:
            V_int_sigma_sq (float): sigma_hat for integrated value
        """
        # print("start calculating Omega...")
        if not hasattr(self, 'Omega') or self.Omega is None:
            self._Omega_hat(policy=policy,
                            block=block,
                            ipw=ipw,
                            estimate_missing_prob=estimate_missing_prob,
                            weight_curr_step=weight_curr_step,
                            prob_lbound=prob_lbound,
                            ridge_factor=ridge_factor,
                            grid_search=grid_search,
                            verbose=verbose,
                            subsample_index=subsample_index)
        
        print("start extracting U....")
        if S_inits is None:
            if MC_size is not None:
                S_inits = self.sample_initial_states(size=MC_size, from_data=True)
            else:
                S_inits = self._initial_obs
        U = self._U(S=S_inits, policy=policy)  # (MC_size, para_dim * num_actions)
        U_int = np.mean(U, axis=0).reshape(-1, 1)  # (para_dim * num_actions, 1)

        # get sigma_sq
        print("start obtaining squared sigma....")
        V_int_sigma_sq = reduce(np.matmul, [
            U_int.T, self.inv_Sigma_hat, self.Omega, self.inv_Sigma_hat.T,
            U_int
        ])
        self.V_int_sigma_sq = V_int_sigma_sq.item()

        beta_hat_sigma_sq = np.diag(
            reduce(np.matmul,
                   [self.inv_Sigma_hat, self.Omega, self.inv_Sigma_hat.T]))

        return self.V_int_sigma_sq

    def inference_int(self,
                      policy,
                      alpha=0.05,
                      ipw=False,
                      estimate_missing_prob=False,
                      weight_curr_step=True,
                      prob_lbound=1e-3,
                      ridge_factor=1e-9,
                      grid_search=False,
                      MC_size=10000,
                      S_inits=None,
                      verbose=True,
                      subsample_index=None,
                      block=False):
        """Make inference on integrated value.

        Args:
            policy (callable): target policy to be evaluated
            alpha (float): significance level
            ipw (bool): if True, use IPW
            estimate_missing_prob (bool): if True, estimate the probability. Otherwise, use the true probability.
            weight_curr_step (bool): if True, wight by the inverse probability of observing the next state. 
                Otherwise, use survival probability. Default is True.
            prob_lbound (float): lower bound of prob
            ridge_factor (float): weight of ridge penalty
            grid_search (bool): if True, use grid search to find optimal ridge_factor
            MC_size (int): sample size for Monte Carlo approximation
            S_inits (np.ndarray): initial states for policy evaluation
            verbose (bool): if True, output intermediate results
            subsample_index (list): ids of subsample to estimate beta
            block (bool): if True, use the current block for policy evaluation. Otherwise, use all the data.

        Returns:
            inference_summary (dict)
        """

        V_int_sigma_sq = self._sigma_int(
            policy=policy,
            block=block,
            ipw=ipw,
            estimate_missing_prob=estimate_missing_prob,
            weight_curr_step=weight_curr_step,
            prob_lbound=prob_lbound,
            ridge_factor=ridge_factor,
            grid_search=grid_search,
            MC_size=MC_size,
            S_inits=S_inits,
            verbose=verbose,
            subsample_index=subsample_index)
        print("start getting V value ...")
        start = time.time()
        V = self.V_int(policy=policy, MC_size=MC_size, S_inits=S_inits)
        print("Finshed! %d secs elapsed" % (time.time() - start))
        std = (V_int_sigma_sq**0.5) / (self.total_T_ipw**0.5)
        print(f'total_T_ipw: {self.total_T_ipw}')
        lower_bound = V - norm.ppf(1 - alpha / 2) * std
        upper_bound = V + norm.ppf(1 - alpha / 2) * std
        inference_summary = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'value': V,
            'std': std
        }
        return inference_summary

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
        if MC_size is None and S_inits is None:
            S_inits = self._initial_obs
        est_V = self.V_int(policy=target_policy, MC_size=MC_size, S_inits=S_inits)
        return est_V

    def evaluate_policy(self,
                        policy,
                        eval_size=20,
                        seed=None,
                        S_inits=None,
                        lower_b=None,
                        upper_b=None,
                        calc_value_est=True,
                        mask_unobserved=True):
        """
        Evaluate given policy (Deprecated)

        Args:
            policy (callable): target policy to be evaluated
            eval_size (int): Monte Carlo size to estimate point-wise value
            seed (int): random seed passed to gen_single_traj() or gen_batch_trajs()
            S_inits (np.ndarray): initial states for policy evaluation
            lower_b (float): lower bound of value
            upper_b (float): upper bound of value
            calc_value_est (bool): if True, also output estimated value
            mask_unobserved (bool): if True, mask unobserved states

        Returns:
            true_V (list): true value for each state
            action_percent: percentage of action=1 (only apply for binary action)
            est_V (list): estimated value for each state
            proportion of value within the bounds (float)
        """
        if not self.vectorized_env:
            true_V = []
            action_percent = []
            est_V = []
            count = 0
            if S_inits is not None and np.shape(S_inits) == (self.env.dim, ):
                S_inits = np.tile(A=S_inits, reps=(eval_size, 1))
            # evaluation on n subjects
            for i in range(eval_size):
                trajectories = self.gen_single_traj(
                    policy=policy,
                    seed=seed,
                    S_init=S_inits[i] if S_inits is not None else None,
                    evaluation=True)
                S, A, reward, T = trajectories[:4]
                est_true_Value = sum(
                    np.array([self.gamma**j
                              for j in range(T)]) * np.array(reward))
                true_V.append(est_true_Value)
                action_percent.append(np.mean(A))
                if calc_value_est:
                    est_V.append(np.mean(self.V(S=S[0], policy=policy)))
                if lower_b or upper_b is not None:
                    if est_true_Value >= lower_b and est_true_Value <= upper_b:
                        count += 1
            if lower_b or upper_b is not None:
                return true_V, action_percent, est_V, count / eval_size
            else:
                return true_V, action_percent, est_V, []
        else:
            old_num_envs = self.eval_env.num_envs
            # reset num_envs
            self.eval_env.num_envs = eval_size
            self.eval_env.observation_space = batch_space(
                self.eval_env.single_observation_space,
                n=self.eval_env.num_envs)
            self.eval_env.action_space = Tuple(
                (self.eval_env.single_action_space, ) * self.eval_env.num_envs)

            count = 0
            if S_inits is not None and np.shape(S_inits) == (self.env.dim, ):
                S_inits = np.tile(A=S_inits, reps=(eval_size, 1))
            trajectories = self.gen_batch_trajs(
                policy=policy,
                seed=seed,
                S_inits=S_inits if S_inits is not None else None,
                evaluation=True)
            S, A, reward, T = trajectories[:4]
            # set unobserved (mask=0) reward to 0
            if mask_unobserved:
                rewards_history = self.eval_env.rewards_history * self.eval_env.states_history_mask[:, :
                                                                                                    -1]
            else:
                rewards_history = self.eval_env.rewards_history
            # recover num_envs
            self.eval_env.num_envs = old_num_envs
            self.eval_env.observation_space = batch_space(
                self.eval_env.single_observation_space,
                n=self.eval_env.num_envs)
            self.eval_env.action_space = Tuple(
                (self.eval_env.single_action_space, ) * self.eval_env.num_envs)

            true_value = np.matmul(
                rewards_history,
                self.gamma**np.arange(start=0,
                                      stop=rewards_history.shape[1]).reshape(
                                          -1, 1))
            true_V = true_value.squeeze().tolist()
            action_percent = np.mean(A, axis=1).tolist()
            est_V = []
            if calc_value_est:
                for i in range(len(S)):
                    est_V.append(self.V(S=S[i][0], policy=policy))
            if lower_b or upper_b is not None:
                if true_value >= lower_b and true_value <= upper_b:
                    count += 1
        if lower_b or upper_b is not None:
            return true_V, action_percent, est_V, count / eval_size
        else:
            return true_V, action_percent, est_V, []