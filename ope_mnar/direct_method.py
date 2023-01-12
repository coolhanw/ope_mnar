import numpy as np
import pandas as pd
import os
import pickle
import time
import collections
from termcolor import colored
from functools import reduce, partial
from itertools import product
from scipy.stats import norm
from scipy.interpolate import BSpline
# ML model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
# RL environment
from gym.spaces import Tuple
from gym.vector.utils.spaces import batch_space
# visualization
import seaborn as sns
import matplotlib.pyplot as plt

from utils import normcdf, iden, MinMaxScaler, SimpleReplayBuffer
from base import SimulationBase
from batch_rl.dqn import QNetwork

__all__ = ['LSTDQ', 'FQE']

class LSTDQ(SimulationBase):
    """
    Shi, Chengchun, et al. "Statistical inference of the value function for reinforcement learning in infinite-horizon settings." 
    Journal of the Royal Statistical Society. Series B: Statistical Methodology (2021).
    """

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

        super().__init__(env=env,
                         n=n,
                         horizon=horizon,
                         discount=discount,
                         eval_env=eval_env)

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

        self.para_dim = None  # the dimension of parameter built in basis spline
        self.product_tensor = product_tensor
        self.basis_scale_factor = basis_scale_factor
        self.bspline = None
        self.knot = None  # quantile knots for basis spline

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
                    self.Q(S=S, A=A, predictor=False)
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
            output = output.T  # (n, para_dim)
            output *= self.basis_scale_factor
            return output
        else:
            raise NotImplementedError

    def Q(self, S, A, predictor=False):
        """
        Return the value of Q-function given states and actions. 

        Args:
            S (np.ndarray) : array of states, dimension (n, state_dim)
            A (np.ndarray) : array of actions, dimension (n, )
            predictor (bool) : if True, return the value of each basis function

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
                S_inits = self.sample_initial_states(size=MC_size,
                                                     from_data=True)
        return np.mean(self.V(policy=policy, S=S_inits))

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
        min_inverse_wt = 1 / prob_lbound if prob_lbound is not None else 0
        
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

        obs = np.vstack(obs_list)  # (total_T, S_dim)
        next_obs = np.vstack(next_obs_list)  # (total_T, S_dim)
        actions = np.vstack(action_list)  # (total_T, 1)
        rewards = np.vstack(reward_list)  # (total_T, 1)
        inverse_wts = np.vstack(inverse_wt_list).astype(float)  # (total_T, 1)
        del training_buffer
        print(f'pseudo sample size: {inverse_wts.sum()}')

        Xi_mat = self._Xi(S=obs, A=actions) # (n, para_dim*num_actions)
        self._Xi_mat = Xi_mat

        if dropout_prob_concat:
            dropout_prob_concat = np.concatenate(dropout_prob_concat)
        U_mat = self._U(S=next_obs, policy=policy)

        mat1 = np.matmul(Xi_mat.T, inverse_wts * (Xi_mat - self.gamma * U_mat))
        mat2 = np.matmul(Xi_mat.T, inverse_wts * rewards)
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

            self._ridge_param_cv = {
                'best_param': best_param,
                'best_gcv': best_gcv,
                'best_edof': best_edof
            }
            ridge_factor = best_param

        self.Sigma_hat = np.diag([ridge_factor] * mat1.shape[0]) + mat1 / self.total_T_ipw
        self.Sigma_hat = self.Sigma_hat.astype(float)
        self.vector = mat2 / self.total_T_ipw
        self.inv_Sigma_hat = np.linalg.pinv(self.Sigma_hat)
        self.est_beta = np.matmul(self.inv_Sigma_hat, self.vector)
        # store the estimated beta in self.para
        self._store_para(self.est_beta)
        if verbose:
            print('Finish estimating beta!')        

        if verbose:
            print('Start calculating Omega...')
        proj_td = inverse_wts * (rewards +
             self.gamma * self.V(S=next_obs, policy=policy).reshape(-1, 1) -
             self.Q(S=obs, A=actions).reshape(-1, 1)) * Xi_mat  # (n, para_dim*num_actions)
        output = np.matmul(
            proj_td.T, proj_td)  # (para_dim*num_actions, para_dim*num_actions)
        self.Omega = output / self.total_T_ipw
        if verbose:
            print('Finish calculating Omega!')

    def estimate_Q(self, 
                  target_policy,
                  ipw=False,
                  estimate_missing_prob=False,
                  weight_curr_step=True,
                  prob_lbound=1e-3,
                  ridge_factor=0.,
                  L=10, 
                  d=3, 
                  knots=None,
                  grid_search=False,
                  verbose=True,
                  subsample_index=None):
        """Main function for estimating the parameters for the Q-function."""
        
        self.target_policy = target_policy

        self.B_spline(L=L, d=d, knots=knots)

        self._beta_hat(
            policy=target_policy,
            ipw=ipw,
            estimate_missing_prob=estimate_missing_prob,
            weight_curr_step=weight_curr_step,
            prob_lbound=prob_lbound,
            ridge_factor=ridge_factor,
            grid_search=grid_search,
            verbose=verbose,
            subsample_index=subsample_index
        )

    def _store_para(self, est_beta):
        """Store the estimated beta in self.para
        
        Args:
            est_beta (np.ndarray): vector of estimated beta
        """
        if est_beta is None:
            est_beta = self.est_beta
        for i in range(self.num_actions):
            self.para[i] = est_beta[i * self.para_dim:(i + 1) *
                                         self.para_dim].reshape(-1)

    def _Omega_hat(self,
                   policy,
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
        
        if not hasattr(self, 'est_beta') or self.est_beta is None:
            self._beta_hat(policy=policy,
                        ipw=ipw,
                        estimate_missing_prob=estimate_missing_prob,
                        weight_curr_step=weight_curr_step,
                        prob_lbound=prob_lbound,
                        ridge_factor=ridge_factor,
                        grid_search=grid_search,
                        verbose=verbose,
                        subsample_index=subsample_index)

        if not hasattr(self, 'Omega') or self.Omega is None:
            total_T = 0
            training_buffer = self.masked_buffer
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
            inverse_wts = np.vstack(inverse_wt_list).astype(float)  # (total_T, 1)
            inverse_wts_mat = np.diag(v=inverse_wts.squeeze()).astype(
                float)  # (total_T, total_T)
            del training_buffer
            
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

    #####################################
    ##  inference on point-wise value
    #####################################

    def _sigma(self,
               S,
               policy,
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

    def _sigma_int(self,
                   policy,
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
        if not hasattr(self, 'Omega') or self.Omega is None:
            self._Omega_hat(policy=policy,
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
                S_inits = self.sample_initial_states(size=MC_size,
                                                     from_data=True)
            else:
                S_inits = self._initial_obs
        U = self._U(S=S_inits,
                    policy=policy)  # (MC_size, para_dim * num_actions)
        U_int = np.mean(U, axis=0).reshape(-1,
                                           1)  # (para_dim * num_actions, 1)

        # get sigma_sq
        print("start obtaining squared sigma....")
        V_int_sigma_sq = reduce(np.matmul, [
            U_int.T, self.inv_Sigma_hat, self.Omega, self.inv_Sigma_hat.T,
            U_int
        ])
        self.V_int_sigma_sq = V_int_sigma_sq.item()

        # beta_hat_sigma_sq = np.diag(
        #     reduce(np.matmul,
        #            [self.inv_Sigma_hat, self.Omega, self.inv_Sigma_hat.T]))

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
                      subsample_index=None
                    ):
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

        Returns:
            inference_summary (dict)
        """

        V_int_sigma_sq = self._sigma_int(
            policy=policy,
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

    def get_state_values(self, target_policy=None, S_inits=None):
        if target_policy is None:
            target_policy = self.target_policy
        if S_inits is None:
            S_inits = self._initial_obs
        return self.V(S_inits)

    def get_value(self, target_policy=None, S_inits=None, MC_size=None):
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
        if target_policy is None:
            target_policy = self.target_policy
        est_V = self.V_int(policy=target_policy,
                           MC_size=MC_size,
                           S_inits=S_inits)
        return est_V

    def get_value_interval(self, target_policy=None, alpha=0.05, S_inits=None, MC_size=None):
        """Main function for getting value confidence interval."""

        print("Calculating the U matrix")
        if S_inits is None:
            if MC_size is not None:
                S_inits = self.sample_initial_states(size=MC_size,
                                                     from_data=True)
            else:
                S_inits = self._initial_obs
        if target_policy is None:
            target_policy = self.target_policy
        U = self._U(S=S_inits, policy=target_policy)  # (MC_size, para_dim * num_actions)
        U_int = np.mean(U, axis=0).reshape(-1,1)  # (para_dim * num_actions, 1)

        print("Getting value point estimates")
        V = self.V_int(policy=target_policy, MC_size=MC_size, S_inits=S_inits)

        print("Calculating sigma-squared")
        V_int_sigma_sq = reduce(np.matmul, [
            U_int.T, self.inv_Sigma_hat, self.Omega, self.inv_Sigma_hat.T,
            U_int
        ])
        V_int_sigma_sq = V_int_sigma_sq.item()
        std = (V_int_sigma_sq**0.5) / (self.total_T_ipw**0.5)
       
        lower_bound = V - norm.ppf(1 - alpha / 2) * std
        upper_bound = V + norm.ppf(1 - alpha / 2) * std
        print("Finish value inference!")

        inference_summary = {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'value': V,
            'std': std
        }
        return inference_summary

    def validate_Q(self, grid_size=10, visualize=False):
        self.grid = []
        self.idx2states = collections.defaultdict(list)

        obs_list, action_list = [], []
        for i in self.masked_buffer.keys():
            S_traj = self.masked_buffer[i][0]
            A_traj = self.masked_buffer[i][1]
            T = len(S_traj) - 1

            obs_list.append(S_traj[:-1])
            action_list.append(np.array(A_traj[:T]).reshape(-1, 1))

        states = np.vstack(obs_list)  # (total_T, S_dim)
        actions = np.vstack(action_list)  # (total_T, 1)
        Q_est = self.Q(S=states, A=actions).squeeze()

        # generate trajectories under the target policy
        init_states = states # self._initial_obs
        init_actions = actions # self._init_actions
        old_num_envs = self.eval_env.num_envs
        eval_size = len(init_states)

        self.eval_env.num_envs = eval_size
        self.eval_env.observation_space = batch_space(
            self.eval_env.single_observation_space,
            n=self.eval_env.num_envs)
        self.eval_env.action_space = Tuple(
            (self.eval_env.single_action_space, ) * self.eval_env.num_envs)

        trajectories = self.gen_batch_trajs(
            policy=self.target_policy,
            seed=None,
            S_inits=init_states,
            A_inits=init_actions,
            burn_in=0,
            evaluation=True)
        rewards_history = self.eval_env.rewards_history * \
                    self.eval_env.states_history_mask[:,
                                                      :self.eval_env.rewards_history.shape[1]]
        Q_ref = np.matmul(
                    rewards_history,
                    self.gamma**np.arange(
                        start=0, stop=rewards_history.shape[1]).reshape(-1, 1)).squeeze()
        # recover eval_env
        self.eval_env.num_envs = old_num_envs
        self.eval_env.observation_space = batch_space(
            self.eval_env.single_observation_space,
            n=self.eval_env.num_envs)
        self.eval_env.action_space = Tuple(
            (self.eval_env.single_action_space, ) * self.eval_env.num_envs)

        discretized_states = np.zeros_like(states)
        for i in range(self.state_dim):
            disc_bins = np.linspace(start=self.low[i] - 0.1, stop=self.high[i] + 0.1, num=grid_size + 1)
            # disc_bins = np.quantile(a=states[:,i], q=np.linspace(0, 1, grid_size + 1))
            # disc_bins[0] -= 0.1
            # disc_bins[-1] += 0.1
            self.grid.append(disc_bins)
            discretized_states[:,i] = np.digitize(states[:,i], bins=disc_bins) - 1
        discretized_states = list(map(tuple, discretized_states.astype('int')))
        for ds, s, a, q, qr in zip(discretized_states, states, actions.squeeze(), Q_est, Q_ref):
            self.idx2states[ds].append(np.concatenate([s, [a], [q], [qr]]))

        # only for 2D state, binary action
        Q_mat = np.zeros(shape=(self.num_actions, grid_size, grid_size))
        Q_ref_mat = np.zeros(shape=(self.num_actions, grid_size, grid_size))
        for k, v in self.idx2states.items():
            v = np.array(v)
            if any(v[:,self.state_dim] == 0):
                Q_mat[0][k[0]][k[1]] = np.mean(v[v[:,self.state_dim] == 0,self.state_dim+1])
                Q_ref_mat[0][k[0]][k[1]] = np.mean(v[v[:,self.state_dim] == 0,self.state_dim+2])
            if any(v[:,self.state_dim] == 1):
                Q_mat[1][k[0]][k[1]] = np.mean(v[v[:,self.state_dim] == 1,self.state_dim+1])
                Q_ref_mat[1][k[0]][k[1]] = np.mean(v[v[:,self.state_dim] == 1,self.state_dim+2])

        if visualize:

            fig, ax = plt.subplots(2, self.num_actions, figsize=(5*self.num_actions,8))
            for a in range(self.num_actions):
                sns.heatmap(
                    Q_mat[a], 
                    cmap="YlGnBu",
                    linewidth=1,
                    ax=ax[0,a]
                )
                ax[0,a].invert_yaxis()
                ax[0,a].set_title(f'estimated Q (action={a})')
            for a in range(self.num_actions):
                sns.heatmap(
                    Q_ref_mat[a], 
                    cmap="YlGnBu",
                    linewidth=1,
                    ax=ax[1,a]
                )
                ax[1,a].invert_yaxis()
                ax[1,a].set_title(f'empirical Q (action={a})')
            plt.savefig('Q_func_heatplot.png')



class FQE(SimulationBase):
    """
    Le, Hoang, Cameron Voloshin, and Yisong Yue. "Batch policy learning under constraints." 
    International Conference on Machine Learning. PMLR, 2019.
    """

    def __init__(self,
                 env=None,
                 n=500,
                 horizon=None,
                 discount=0.8,
                 eval_env=None,
                 device=None):
        """
        Args:
            env (gym.Env): dynamic environment
            n (int): the number of subjects (trajectories)
            horizon (int): the maximum length of trajectories
            discount (float): discount factor
            eval_env (gym.Env): dynamic environment to evaluate the policy, if not specified, use env
        """

        super().__init__(env=env,
                         n=n,
                         horizon=horizon,
                         discount=discount,
                         eval_env=eval_env)

        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def estimate_Q(
            self,
            target_policy,
            max_iter=200,
            tol=0.001,
            use_RF=False,
            hidden_sizes=[256,256],
            # num_actions=2,
            lr=0.001,
            batch_size=128,
            epoch=20000,
            patience=10,
            seed=0,
            scaler="Standard",
            print_freq=10,
            verbose=0,
            # other RF arguments
            **kwargs):

        self.replay_buffer = SimpleReplayBuffer(trajs=self.masked_buffer,
                                                seed=seed)
        self.target_policy = target_policy
        self.use_RF = use_RF
        self.lr = lr

        if scaler == "NormCdf":
            self.scaler = normcdf()
        elif scaler == "Identity":
            self.scaler = iden()
        elif scaler == "MinMax":
            self.scaler = MinMaxScaler(
                min_val=self.env.low,
                max_val=self.env.high
            ) if self.env is not None else MinMaxScaler()
        elif scaler == "Standard":
            self.scaler = StandardScaler(with_mean=True, with_std=True)
        else:
            # a path to a fitted scaler
            assert os.path.exists(scaler)
            with open(scaler, 'rb') as f:
                self.scaler = pickle.load(f)

        if use_RF:
            self.model = RandomForestRegressor(**kwargs,
                                               n_jobs=-1,
                                               verbose=0,
                                               random_state=seed)
        else:
            # NN
            self.model = QNetwork(state_dim=self.state_dim,
                                  num_actions=self.num_actions,
                                  hidden_sizes=hidden_sizes)
            self.target_model = QNetwork(state_dim=self.state_dim,
                                  num_actions=self.num_actions,
                                  hidden_sizes=hidden_sizes)
            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
            #                                                  step_size=100,
            #                                                  gamma=0.99)

        # state features shuold be scaled first
        self._train(max_iter=max_iter,
                    tol=tol,
                    batch_size=batch_size,
                    epoch=epoch,
                    patience=patience,
                    print_freq=print_freq,
                    verbose=verbose)

    def _train(self,
               max_iter,
               tol,
               batch_size,
               epoch,
               patience,
               print_freq=10,
               verbose=0):
        state = self.replay_buffer.states
        action = self.replay_buffer.actions
        reward = self.replay_buffer.rewards
        next_state = self.replay_buffer.next_states
        if state.ndim == 1:
            state = state.reshape((-1, 1))
            next_state = next_state.reshape((-1, 1))
        pi_next_state_prob = self.target_policy.get_action_prob(next_state) # the input should be on the original scale
        old_target_Q = reward / (1 - self.gamma)

        # standardize the input
        state_concat = np.vstack([state, next_state])
        self.scaler.fit(state_concat)
        state = self.scaler.transform(state)
        next_state = self.scaler.transform(next_state)

        for itr in range(max_iter):

            if self.use_RF:
                if itr > 0:
                    target_Q = reward + self.gamma * np.sum(
                        self.model.predict(next_state) * pi_next_state_prob,
                        axis=1)
                    _targets = self.model.predict(state)
                    _targets[range(len(action)), action.astype(int)] = target_Q
                else:
                    target_Q = reward / (1 - self.gamma)
                    _targets = np.zeros(shape=(len(action), self.num_actions))
                    _targets[range(len(action)), action.astype(int)] = target_Q
                
                self.model.fit(state, _targets)
            else:
                self.model.train()
                # reset optimizer and scheduler
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                step_size=100,
                                                                gamma=0.99)
                min_loss = float('inf')
                self.losses = []
                wait_count = 0
                for ep in range(epoch):
                    transitions = self.replay_buffer.sample(batch_size)
                    state, action, reward, next_state = transitions[:4]
                    pi_next_state = self.target_policy.get_action(next_state) # input should on the original scale
                    pi_next_state_prob = self.target_policy.get_action_prob(
                        next_state)  # input should on the original scale
                    # standardize the input
                    state = self.scaler.transform(state)
                    next_state = self.scaler.transform(next_state)
                    
                    state = torch.Tensor(state).to(self.device)
                    action = torch.LongTensor(action[:, np.newaxis]).to(self.device)
                    reward = torch.Tensor(reward[:, np.newaxis]).to(self.device)
                    next_state = torch.Tensor(next_state).to(self.device)
                    pi_next_state_prob = torch.Tensor(pi_next_state_prob).to(self.device)

                    # Compute the target Q value
                    if itr > 0:
                        with torch.no_grad():
                            target_Q = reward + self.gamma * torch.sum(
                                self.target_model(next_state) * pi_next_state_prob,
                                dim=1, keepdim=True) # notice, use target network here
                    else:
                        target_Q = reward / (1 - self.gamma)

                    # Get current Q estimate
                    current_Q = self.model(state).gather(dim=1, index=action)

                    # Compute Q loss, closely related to HuberLoss
                    Q_loss = F.smooth_l1_loss(input=current_Q, target=target_Q)

                    # Optimize the Q
                    self.optimizer.zero_grad()
                    Q_loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

                    self.losses.append(Q_loss.detach().numpy())

                    mean_loss = self.losses[0]
                    if ep % 5 == 0 and ep >= 10:
                        if ep >= 50:
                            mean_loss = np.mean(self.losses[(ep - 50):ep])
                        else:
                            mean_loss = np.mean(self.losses[(ep - 10):ep])
                        if mean_loss / min_loss > 1:
                            wait_count += 1
                        if mean_loss < min_loss:
                            min_loss = mean_loss
                            wait_count = 0
                    if patience is not None and wait_count >= patience:
                        break
                
                # update target network
                self._target_update()

                # evaluate on the whole data
                reward = self.replay_buffer.rewards
                next_state = self.replay_buffer.next_states
                pi_next_state_prob = self.target_policy.get_action_prob(next_state)
                next_state = self.scaler.transform(next_state)
                reward = torch.Tensor(reward).to(self.device)
                next_state = torch.Tensor(next_state).to(self.device)
                pi_next_state_prob = torch.Tensor(pi_next_state_prob).to(self.device)
                with torch.no_grad():
                    target_Q = reward + self.gamma * torch.sum(
                        self.model(next_state) * pi_next_state_prob,
                        dim=1, keepdim=True) # notice, use target network here
                # convert to numpy array for comparison
                target_Q = target_Q.detach().numpy()

            target_diff = abs(target_Q - old_target_Q).mean() / (abs(old_target_Q).mean() + 1e-6)
            
            if verbose >= 1 and itr % print_freq == 0:
                action_init_states = self.target_policy.get_action(
                    self._initial_obs)
                values = self.Q(self._initial_obs, action_init_states) # call model.eval() inside
                est_value = np.mean(values)
                print(
                    colored(
                        "[iteration {}] V = {:.2f} with diff = {:.4f} and std_init_Q = {:.1f}"
                        .format(itr, est_value, target_diff,
                                np.std(values)), 'green'))
            if itr > 0 and target_diff < tol:
                break
            
            old_target_Q = target_Q.copy()

    def _target_update(self):
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param.data)

    def Q(self, S, A=None):
        if S.ndim == 1:
            S = np.expand_dims(S, 1)
        S = self.scaler.transform(S)
        if self.use_RF:
            Q_values = self.model.predict(S)
            if A is not None:
                return np.take_along_axis(arr=Q_values, indices=A.reshape(-1,1), axis=1)
            else:
                return Q_values
        else:
            S = torch.Tensor(S).to(self.device)

            self.model.eval()
            Q_values = self.model(S)

            if A is not None:
                A = torch.LongTensor(A.reshape(-1,1)).to(self.device)
                return Q_values.gather(dim=1, index=A).detach().numpy()
            else:
                return Q_values.detach().numpy()

    def V(self, S, policy=None):
        if S.ndim == 1:
            S = np.expand_dims(S, 1)
        if policy is None:
            policy = self.target_policy
        action_probs = policy.get_action_prob(S)
        Q_values = self.Q(S, A=None)  # (n, num_actions)
        return np.sum(Q_values * action_probs, axis=1)

    def get_state_values(self, target_policy=None, S_inits=None):
        if target_policy is None:
            target_policy = self.target_policy
        if S_inits is None:
            S_inits = self._initial_obs
        return self.V(S_inits)

    def get_value(self, target_policy=None, S_inits=None, MC_size=None):
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
        if target_policy is None:
            target_policy = self.target_policy
        est_V = self.V_int(policy=target_policy,
                           MC_size=MC_size,
                           S_inits=S_inits)
        return est_V

    def validate_Q(self, grid_size=10, visualize=False):
        self.grid = []
        self.idx2states = collections.defaultdict(list)
        states = self.replay_buffer.states
        actions = self.replay_buffer.actions
        Q_est = self.Q(S=states, A=actions).squeeze()

        # generate trajectories under the target policy
        init_states = states # self._initial_obs
        init_actions = actions # self._init_actions
        old_num_envs = self.eval_env.num_envs
        eval_size = len(init_states)

        self.eval_env.num_envs = eval_size
        self.eval_env.observation_space = batch_space(
            self.eval_env.single_observation_space,
            n=self.eval_env.num_envs)
        self.eval_env.action_space = Tuple(
            (self.eval_env.single_action_space, ) * self.eval_env.num_envs)

        trajectories = self.gen_batch_trajs(
            policy=self.target_policy.policy_func,
            seed=None,
            S_inits=init_states,
            A_inits=init_actions,
            burn_in=0,
            evaluation=True)
        rewards_history = self.eval_env.rewards_history * \
                    self.eval_env.states_history_mask[:,
                                                      :self.eval_env.rewards_history.shape[1]]
        Q_ref = np.matmul(
                    rewards_history,
                    self.gamma**np.arange(
                        start=0, stop=rewards_history.shape[1]).reshape(-1, 1)).squeeze()
        # recover eval_env
        self.eval_env.num_envs = old_num_envs
        self.eval_env.observation_space = batch_space(
            self.eval_env.single_observation_space,
            n=self.eval_env.num_envs)
        self.eval_env.action_space = Tuple(
            (self.eval_env.single_action_space, ) * self.eval_env.num_envs)

        discretized_states = np.zeros_like(states)
        for i in range(self.state_dim):
            disc_bins = np.linspace(start=self.low[i] - 0.1, stop=self.high[i] + 0.1, num=grid_size + 1)
            # disc_bins = np.quantile(a=states[:,i], q=np.linspace(0, 1, grid_size + 1))
            # disc_bins[0] -= 0.1
            # disc_bins[-1] += 0.1
            self.grid.append(disc_bins)
            discretized_states[:,i] = np.digitize(states[:,i], bins=disc_bins) - 1
        discretized_states = list(map(tuple, discretized_states.astype('int')))
        for ds, s, a, q, qr in zip(discretized_states, states, actions, Q_est, Q_ref):
            self.idx2states[ds].append(np.concatenate([s, [a], [q], [qr]]))

        # only for 2D state, binary action
        Q_mat = np.zeros(shape=(self.num_actions, grid_size, grid_size))
        Q_ref_mat = np.zeros(shape=(self.num_actions, grid_size, grid_size))
        for k, v in self.idx2states.items():
            v = np.array(v)
            if any(v[:,self.state_dim] == 0):
                Q_mat[0][k[0]][k[1]] = np.mean(v[v[:,self.state_dim] == 0,self.state_dim+1])
                Q_ref_mat[0][k[0]][k[1]] = np.mean(v[v[:,self.state_dim] == 0,self.state_dim+2])
            if any(v[:,self.state_dim] == 1):
                Q_mat[1][k[0]][k[1]] = np.mean(v[v[:,self.state_dim] == 1,self.state_dim+1])
                Q_ref_mat[1][k[0]][k[1]] = np.mean(v[v[:,self.state_dim] == 1,self.state_dim+2])

        if visualize:

            fig, ax = plt.subplots(2, self.num_actions, figsize=(5*self.num_actions,8))
            for a in range(self.num_actions):
                sns.heatmap(
                    Q_mat[a], 
                    cmap="YlGnBu",
                    linewidth=1,
                    ax=ax[0,a]
                )
                ax[0,a].invert_yaxis()
                ax[0,a].set_title(f'estimated Q (action={a})')
            for a in range(self.num_actions):
                sns.heatmap(
                    Q_ref_mat[a], 
                    cmap="YlGnBu",
                    linewidth=1,
                    ax=ax[1,a]
                )
                ax[1,a].invert_yaxis()
                ax[1,a].set_title(f'empirical Q (action={a})')
            plt.savefig('Q_func_heatplot.png')


