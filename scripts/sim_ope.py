"""
Examine the point-estimator of the integrated value.
"""

import os
import numpy as np
import pandas as pd
from numpy.linalg import inv
from functools import reduce
from scipy import integrate
from scipy.stats import norm
import argparse
import pathlib
import time
import gc
from collections import defaultdict, Counter

try:
    from ope_mnar.utils import SimEnv, VectorSimEnv
    from ope_mnar.main import train_Q_func, get_target_value_multi
except:
    import sys
    sys.path.append(os.path.expanduser('~/Projects/ope_mnar/ope_mnar'))
    sys.path.append(os.path.expanduser('~/Projects/ope_mnar'))
    from ope_mnar.utils import SimEnv, VectorSimEnv
    from ope_mnar.main import train_Q_func, get_target_value_multi

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='linear') # 'linear', 'SAVE'
parser.add_argument('--max_episode_length', type=int, default=25) # 10, 25
parser.add_argument('--discount', type=float, default=0.8)
parser.add_argument('--num_trajs', type=int, default=500) # 250, 500
parser.add_argument('--burn_in', type=int, default=0)
parser.add_argument('--max_itr', type=int, default=100)
parser.add_argument('--mc_size', type=int, default=250)
parser.add_argument('--eval_policy_mc_size', type=int, default=10000)
parser.add_argument('--eval_horizon', type=int, default=250)
parser.add_argument('--dropout_scheme', type=str, default='3.19')
parser.add_argument('--dropout_rate', type=float, default=0.9)
parser.add_argument('--dropout_obs_count_thres', type=int, default=2, help="the number of observations that is not subject to dropout")
parser.add_argument('--scale_state',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=False,
                    help="If True, state features are already scaled to [0,1]")
parser.add_argument('--ipw',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
parser.add_argument('--estimate_missing_prob',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
parser.add_argument('--weight_curr_step',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
parser.add_argument('--env_model_type', type=str, default='linear')
parser.add_argument('--vectorize_env',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
args = parser.parse_args()


if __name__ == '__main__':
    log_dir = os.path.expanduser('~/Projects/ope_mnar/output')
    env_class = args.env
    default_scaler = "MinMax" # "NormCdf", "MinMax"
    T = args.max_episode_length
    n = args.num_trajs
    total_N = None
    num_actions = 2
    state_dim = 2
    gamma = args.discount
    mc_size = args.mc_size
    dropout_scheme = args.dropout_scheme
    dropout_rate = args.dropout_rate
    burn_in = args.burn_in
    scale_state = args.scale_state
    vectorize_env = args.vectorize_env
    ipw = args.ipw
    weight_curr_step = args.weight_curr_step # if False, use survival probability
    adaptive_dof = False
    estimate_missing_prob = args.estimate_missing_prob
    if not ipw:
        method = 'cc'
    elif ipw and estimate_missing_prob:
        method = 'ipw_propF'
    elif ipw and not estimate_missing_prob:
        method = 'ipw_propT'
    dropout_obs_count_thres = args.dropout_obs_count_thres
    instrument_var_index = None
    mnar_y_transform = None
    bandwidth_factor = None
    if env_class.lower() in ['linear','save']:
        if dropout_scheme == '0':
            missing_mechanism = None
        elif dropout_scheme in ['3.19','3.20']:
            missing_mechanism = 'mnar'
            instrument_var_index = 1
            if dropout_scheme == '3.19':
                bandwidth_factor = 7.5
            elif dropout_scheme == '3.20':
                bandwidth_factor = 2.5
        else :
            missing_mechanism = 'mar' 

    gamma_true = None # 1.5, None
    if missing_mechanism and missing_mechanism.lower() == 'mnar':
        initialize_with_gammaT = False
    
    prop = 'propT' if not estimate_missing_prob else 'propF'
    train_opt_policy = False
    eval_policy_mc_size = args.eval_policy_mc_size # 10000
    eval_horizon = args.eval_horizon # 500
    solve_iteratively = False
    max_itr = args.max_itr
    drop_last_TD = True
    folder_suffix = '_scaled' if scale_state else '_unscaled'
    folder_suffix += f'_missing{dropout_rate}'

    error_bound = 0.005
    scale = "Identity" if scale_state else default_scaler
    low = 0 if scale_state else  -norm.ppf(0.999) # -np.inf
    high = 1 if scale_state else norm.ppf(0.999) # np.inf
    eval_seed = 123
    dropout_model_type = 'linear'
    product_tensor = True
    prob_lbound = 1e-2
    
    grid_search = False
    basis_scale_factor = 100
    basis_type = 'spline'
    spline_degree = 3

    if not scale_state:
        # environment used in SAVE paper
        if env_class.upper() == 'SAVE':
            def vec_state_trans_model(obs, action, rng=np.random):
                if not isinstance(obs, np.ndarray):
                    obs = np.array(obs)
                if len(obs.shape) == 1:
                    obs = obs.reshape(1, -1)
                if not isinstance(action, np.ndarray):
                    action = np.array(action)
                if not action.shape or len(action.shape) == 1:
                    action = action.reshape(-1, 1)
                S_next = (2 * action - 1) * np.matmul(obs, np.array([[0.75,0],[0,-0.75]]))
                S_next += rng.normal(loc=0, scale=0.5, size=S_next.shape)
                return S_next.squeeze() 

            def vec_reward_model(obs, action, next_obs, rng=np.random):
                if not isinstance(obs, np.ndarray):
                    obs = np.array(obs)
                if len(obs.shape) == 1:
                    obs = obs.reshape(1, -1)
                if not isinstance(action, np.ndarray):
                    action = np.array(action)
                if not action.shape or len(action.shape) == 1:
                    action = action.reshape(-1, 1)
                if not isinstance(next_obs, np.ndarray):
                    next_obs = np.array(next_obs)
                if len(next_obs.shape) == 1:
                    next_obs = next_obs.reshape(1, -1)
                obs_nobs_action = np.concatenate([obs, next_obs, action], axis=1)
                weight = np.array([0, 0, 2, 1, -1/2]).reshape(-1, 1)
                reward = np.matmul(obs_nobs_action, weight)
                bias = np.zeros_like(reward) + 1/4
                reward += bias
                noise = rng.normal(loc=0., scale=0., size=reward.shape)
                reward += noise
                return reward
        # environment used in our paper
        else:
            def state_trans_model(obs, action, rng=np.random):
                if not isinstance(obs, np.ndarray):
                    obs = np.array(obs)
                trans_mat = np.array([[(2 * action - 1), 0],
                                    [0, (1 - 2 * action)]])
                bias = np.array([0,0])
                noise = rng.normal(loc=0, scale=0.5, size=trans_mat.shape[0])
                S_next = np.dot(obs, trans_mat) + bias + noise
                return S_next

            def vec_state_trans_model(obs, action, rng=np.random):
                if not isinstance(obs, np.ndarray):
                    obs = np.array(obs)
                if len(obs.shape) == 1:
                    obs = obs.reshape(1, -1)
                if not isinstance(action, np.ndarray):
                    action = np.array(action)
                if not action.shape or len(action.shape) == 1:
                    action = action.reshape(-1, 1)
                
                obs_action = np.concatenate([obs, action, obs*action], axis=1)
                trans_mat = np.array([[-1, 0], [0, 1], [0, 0], [2, 0], [0, -2]])
                S_next = np.matmul(obs_action, trans_mat).astype('float')
                S_next += rng.normal(loc=0.0, scale=0.5, size=S_next.shape)
                return S_next.squeeze()

            def reward_model(obs, action, next_obs, rng=np.random):
                if not isinstance(obs, np.ndarray):
                    obs = np.array(obs)
                if not isinstance(next_obs, np.ndarray):
                    next_obs = np.array(next_obs)
                next_weight = np.array([2, 1])
                weight = np.array([0, 0.5])
                bias = - (2 * action - 1) / 4
                noise = rng.normal(loc=0.0, scale=0.01)
                reward = np.dot(next_obs, next_weight) + np.dot(obs, weight) + bias
                return reward

            def vec_reward_model(obs, action, next_obs, rng=np.random):
                if not isinstance(obs, np.ndarray):
                    obs = np.array(obs)
                if len(obs.shape) == 1:
                    obs = obs.reshape(1, -1)
                if not isinstance(action, np.ndarray):
                    action = np.array(action)
                if not action.shape or len(action.shape) == 1:
                    action = action.reshape(-1, 1)
                if not isinstance(next_obs, np.ndarray):
                    next_obs = np.array(next_obs)
                if len(next_obs.shape) == 1:
                    next_obs = next_obs.reshape(1, -1)
                obs_nobs_action = np.concatenate([obs, next_obs, action], axis=1)
                weight = np.array([0, 0.5, 2, 1, -1/2]).reshape(-1, 1)
                reward = np.matmul(obs_nobs_action, weight)
                bias = np.zeros_like(reward) + 1/4
                reward += bias
                noise = rng.normal(loc=0.0, scale=0.01, size=reward.shape)
                reward += noise
                return reward
    else:
        def state_trans_model(obs, action, rng=np.random):
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
            obs = np.clip(a=obs, a_min=0.001, a_max=0.999)
            obs_orig = norm.ppf(obs)
            trans_mat = np.array([[1, 0], [0, 1]])
            bias = (2 * action - 1) * np.sign(obs_orig) / 10
            noise = rng.normal(loc=0, scale=0.01, size=trans_mat.shape[0])
            S_next_orig = np.dot(obs_orig, trans_mat) + bias + noise
            S_next = norm.cdf(S_next_orig)
            return np.clip(a=S_next, a_min=0.001, a_max=0.999)

        def vec_state_trans_model(obs, action, rng=np.random):
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
            if len(obs.shape) == 1:
                obs = obs.reshape(1, -1)
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            if not action.shape or len(action.shape) == 1:
                action = action.reshape(-1, 1)
            obs = np.clip(a=obs, a_min=0.001, a_max=0.999)
            obs_orig = norm.ppf(obs)
            obs_action = np.concatenate([obs_orig, 2 * action - 1], axis=1)
            trans_mat = np.array([[1, 0], [0, 1], [0, 0]])
            S_next_orig = np.matmul(obs_action, trans_mat).astype('float')
            S_next_orig += (2 * action - 1) / 10 * np.sign(obs_orig)
            S_next_orig += rng.normal(loc=0.0, scale=0.01, size=S_next_orig.shape)
            S_next = norm.cdf(S_next_orig)
            return np.clip(a=S_next, a_min=0.001, a_max=0.999).squeeze() 

        def reward_model(obs, action, next_obs, rng=np.random):
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
            if not isinstance(next_obs, np.ndarray):
                next_obs = np.array(next_obs)
            weight = np.array([2, 1])
            bias = 0
            noise = rng.normal(loc=0., scale=0.01)
            reward = np.dot(next_obs, weight) + bias
            return reward

        def vec_reward_model(obs, action, next_obs, rng=np.random):
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
            if len(obs.shape) == 1:
                obs = obs.reshape(1, -1)
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            if not action.shape or len(action.shape) == 1:
                action = action.reshape(-1, 1)
            if not isinstance(next_obs, np.ndarray):
                next_obs = np.array(next_obs)
            if len(next_obs.shape) == 1:
                next_obs = next_obs.reshape(1, -1)
            obs_nobs_action = np.concatenate([obs, next_obs, action], axis=1)
            weight = np.array([0, 0, 2, 1, 0]).reshape(-1, 1)
            reward = np.matmul(obs_nobs_action, weight)
            bias = np.zeros_like(reward)
            reward += bias
            noise = rng.normal(loc=0.0, scale=0.01, size=reward.shape)
            reward += noise
            return reward

    def dropout_model(obs_history, action_history, reward_history):
        if len(action_history) < dropout_obs_count_thres:
            return 0
        intercept = 3.5  # 3
        if dropout_scheme == '0':
            return 0
        elif dropout_scheme == '3.19':
            if not scale_state:
                if T == 25:
                    if dropout_rate == 0.9:
                        if dropout_obs_count_thres == 5:
                            logit = intercept + 2.4 + 0.8 * obs_history[-2][0] - 1.5 * reward_history[-1]
                        elif dropout_obs_count_thres == 2:
                            logit = intercept + 3.2 + 0.8 * obs_history[-2][0] - 1.5 * reward_history[-1]
                        elif dropout_obs_count_thres == 1:
                            logit = intercept + 4 + 0.8 * obs_history[-2][0] - 1.5 * reward_history[-1]
                    else:
                        raise NotImplementedError
            else:
                logit = intercept + 6.5 + 1. * obs_history[-2][0] - 4.5 * reward_history[-1]
            prob = 1 / (np.exp(logit) + 1)
        else:
            raise NotImplementedError
        return prob

    def vec_dropout_model(obs_history, action_history, reward_history):
        if not isinstance(obs_history, np.ndarray):
            obs_history = np.array(obs_history)
        if not isinstance(action_history, np.ndarray):
            action_history = np.array(action_history)
        if not isinstance(reward_history, np.ndarray):
            reward_history = np.array(reward_history)
        # dropout only happens after a threshold
        # if action_history.shape[1] <= dropout_obs_count_thres: # burn_in
        if action_history.shape[1] < dropout_obs_count_thres:
            return np.zeros(shape=obs_history.shape[0]).reshape(-1,1)
        intercept = 3.5  # 3
        if dropout_scheme == '0':
            prob = np.zeros(shape=obs_history.shape[0])
        elif dropout_scheme == '3.19':
            if not scale_state:
                if T == 25:
                    if dropout_rate == 0.6:
                        logit = intercept + 7.0 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-1]
                    elif dropout_rate == 0.7:
                        logit = intercept + 5.0 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-1]
                    elif dropout_rate == 0.8:
                        logit = intercept + 3.6 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-1]
                    elif dropout_rate == 0.9:
                        if dropout_obs_count_thres == 5:
                            logit = intercept + 2.4 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-1]
                        elif dropout_obs_count_thres == 2:
                            logit = intercept + 3.2 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-1]
                        elif dropout_obs_count_thres == 1:
                            logit = intercept + 4 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-1]
                    else:
                        raise NotImplementedError
                elif T == 10:
                    if dropout_rate == 0.6:
                        if dropout_obs_count_thres == 5:
                            logit = intercept + 2 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-1]
                        elif dropout_obs_count_thres == 3:
                            logit = intercept + 3 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-1]
                        elif dropout_obs_count_thres == 2:
                            logit = intercept + 4 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-1]
                    elif dropout_rate == 0.7:
                        logit = intercept + 0. + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-1]
                    elif dropout_rate == 0.8:
                        logit = intercept - 0.8 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-1]
                    elif dropout_rate == 0.9:
                        if dropout_obs_count_thres == 6:
                            logit = intercept - 2. + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-1]
                        elif dropout_obs_count_thres == 5:
                            logit = intercept - 1.2 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-1]
                        elif dropout_obs_count_thres == 4:
                            logit = intercept - 0.9 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-1]
                        elif dropout_obs_count_thres == 3:
                            logit = intercept - 0.6 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-1]
                        elif dropout_obs_count_thres == 2:
                            logit = intercept - 0.2 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-1]
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
            else:
                logit = intercept + 6.5 + 1. * obs_history[:,-2, 0] - 4.5 * reward_history[:,-1]
            prob = 1 / (np.exp(logit) + 1)
        elif dropout_scheme == '3.19-mar':
            if not scale_state:
                if T == 25:
                    if dropout_rate == 0.6:
                        logit = intercept + 4.5 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-2]
                    elif dropout_rate == 0.7:
                        logit = intercept + 3.2 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-2]
                    elif dropout_rate == 0.8:
                        logit = intercept + 2.5 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-2]
                    elif dropout_rate == 0.9:
                        if dropout_obs_count_thres == 5:
                            logit = intercept + 1.5 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-2]
                        elif dropout_obs_count_thres == 2:
                            logit = intercept + 2.3 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-2]
                        elif dropout_obs_count_thres == 1:
                            logit = intercept + 2.8 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-2]
                    else:
                        raise NotImplementedError
                elif T == 10:
                    if dropout_rate == 0.6:
                        if dropout_obs_count_thres == 5:
                            logit = intercept + 0. + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-2]
                        elif dropout_obs_count_thres == 2:
                            logit = intercept + 2.5 + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-2]
                    elif dropout_rate == 0.7:
                        logit = intercept - 1. + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-2]
                    elif dropout_rate == 0.8:
                        logit = intercept - 2. + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-2]
                    elif dropout_rate == 0.9:
                        logit = intercept - 3. + 0.8 * obs_history[:,-2, 0] - 1.5 * reward_history[:,-2]
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
            else:
                logit = intercept + 6.5 + 1. * obs_history[:,-2, 0] - 4.5 * reward_history[:,-2]
            prob = 1 / (np.exp(logit) + 1)
        elif dropout_scheme == '3.20':
            if not scale_state:
                if T == 25:
                    if dropout_rate == 0.6:
                        logit = intercept + 8 - 0.5 * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-1]
                    elif dropout_rate == 0.7:
                        logit = intercept + 5.5 - 0.5 * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-1]
                    elif dropout_rate == 0.8:
                        logit = intercept + 4.2 - 0.5 * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-1]
                    elif dropout_rate == 0.9:
                        if dropout_obs_count_thres == 5:
                            logit = intercept + 3 - 0.5 * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-1] 
                        elif dropout_obs_count_thres == 2:
                            logit = intercept + 3.7 - 0.5 * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-1]  
                        elif dropout_obs_count_thres == 1:
                            logit = intercept + 4.5 - 0.5 * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-1]  
                    else:
                        raise NotImplementedError
                elif T == 10:
                    if dropout_rate == 0.6:
                        if dropout_obs_count_thres == 5:
                            logit = intercept + 1.8 - 1. * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-1]   
                        elif dropout_obs_count_thres == 2:
                            logit = intercept + 6 - 1. * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-1]   
                    elif dropout_rate == 0.7:
                        logit = intercept + 0.5 - 1. * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-1]
                    elif dropout_rate == 0.8:
                        logit = intercept - 0.5 - 1. * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-1]
                    elif dropout_rate == 0.9:
                        if dropout_obs_count_thres == 6:
                            logit = intercept - 2. - 1. * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-1]
                        elif dropout_obs_count_thres == 5:
                            logit = intercept - 1. - 1. * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-1]
                        elif dropout_obs_count_thres == 4:
                            logit = intercept - 0.4 - 1. * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-1]
                        elif dropout_obs_count_thres == 3:
                            logit = intercept - 0. - 1. * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-1]
                        elif dropout_obs_count_thres == 2:
                            logit = intercept + 0.2 - 1. * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-1]
                        elif dropout_obs_count_thres == 1:
                            logit = intercept + 0.5 - 1. * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-1]
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
            else:
                logit = intercept + 6.5 + 0.5 * np.power(obs_history[:,-2, 0],2) - 4.5 * reward_history[:,-1]
            prob = 1 / (np.exp(logit) + 1)
        elif dropout_scheme == '3.20-mar':
            if not scale_state:
                if T == 25:
                    if dropout_rate == 0.6:
                        logit = intercept + 8 - 0.5 * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-2]
                    elif dropout_rate == 0.7:
                        logit = intercept + 5.5 - 0.5 * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-2]
                    elif dropout_rate == 0.8:
                        logit = intercept + 4.2 - 0.5 * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-2]
                    elif dropout_rate == 0.9:
                        if dropout_obs_count_thres == 5:
                            logit = intercept + 3 - 0.5 * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-2]  
                        elif dropout_obs_count_thres == 2:
                            logit = intercept + 3.7 - 0.5 * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-2]  
                        elif dropout_obs_count_thres == 1:
                            logit = intercept + 4.5 - 0.5 * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-2]  
                    else:
                        raise NotImplementedError
                elif T == 10:
                    if dropout_rate == 0.6:
                        if dropout_obs_count_thres == 5:
                            logit = intercept + 1.3 - 1. * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-2]   
                        elif dropout_obs_count_thres == 2:
                            logit = intercept + 6 - 1. * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-2]   
                    elif dropout_rate == 0.7:
                        logit = intercept + 0.5 - 1. * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-2]
                    elif dropout_rate == 0.8:
                        logit = intercept - 0.5 - 1. * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-2]
                    elif dropout_rate == 0.9:
                        logit = intercept - 2. - 1. * np.power(obs_history[:,-2, 0],2) - 1.5 * reward_history[:,-2]
                    else:
                        raise NotImplementedError
                else:
                    raise NotImplementedError
            else:
                logit = intercept + 6.5 + 0.5 * np.power(obs_history[:,-2, 0],2) - 4.5 * reward_history[:,-2]
            prob = 1 / (np.exp(logit) + 1)
        else:
            raise NotImplementedError
        return prob.reshape(-1,1)
 
    def dropout_model_0(obs_history, action_history, reward_history):
        return 0

    def vec_dropout_model_0(obs_history, action_history, reward_history):
        return np.zeros(shape=(obs_history.shape[0], 1))

    if env_class.lower() == 'linear' and vectorize_env:
        env_kwargs = {
            'dim': state_dim,
            'num_actions': num_actions,
            'low': low,
            'high': high, 
            'vec_state_trans_model': vec_state_trans_model,
            'vec_reward_model': vec_reward_model,
            'vec_dropout_model': vec_dropout_model_0  # no dropout
        }

        env_dropout_kwargs = {
            'dim': state_dim,
            'num_actions': num_actions,
            'low': low,
            'high': high, 
            'vec_state_trans_model': vec_state_trans_model,
            'vec_reward_model': vec_reward_model,
            'vec_dropout_model': vec_dropout_model
        }
        env = VectorSimEnv(num_envs=n, T=T, **env_kwargs)
        env_dropout = VectorSimEnv(num_envs=n, T=T, **env_dropout_kwargs)
    elif env_class.lower() == 'linear':
        env_kwargs = {
            'dim': state_dim,
            'num_actions': num_actions,
            'low': low,
            'high': high, 
            'state_trans_model': state_trans_model,
            'reward_model': reward_model,
            'dropout_model': dropout_model_0  # no dropout
        }

        env_dropout_kwargs = {
            'dim': state_dim,
            'num_actions': num_actions,
            'low': low,
            'high': high, 
            'state_trans_model': state_trans_model,
            'reward_model': reward_model,
            'dropout_model': dropout_model
        }
        env = SimEnv(T=T, **env_kwargs)
        env_dropout = SimEnv(T=T, **env_dropout_kwargs)
    else:
        raise NotImplementedError

    if adaptive_dof:
        dof = max(4, int(np.sqrt((n * T)**(3/7)))) # degree of freedom
    else:
        dof = 7
    ridge_factor = 1e-3
    folder_suffix += f'_ridge{ridge_factor}'
    if env_class.upper() == 'SAVE':
        folder_suffix += '_SAVE'
    if basis_scale_factor != 1:
        folder_suffix += f'_scale{int(basis_scale_factor)}'
    if basis_type != 'spline':
        folder_suffix += f'_{basis_type}'
    if scale_state or scale == default_scaler:
        knots = np.linspace(start=-spline_degree/(dof-spline_degree), stop=1+spline_degree/(dof-spline_degree), num=dof+spline_degree+1) # handle the boundary
    else:
        knots = 'equivdist' # 'equivdist', None

    print('Configuration:')
    print(f'T : {T}')
    print(f'n : {n}')
    print(f'total_N : {total_N}')
    print(f'spline degree of freedom : {dof}')
    print(f'gamma : {gamma}')
    print(f'dropout_scheme : {dropout_scheme}')
    print(f'ipw : {ipw}')
    print(f'estimate_missing_prob : {estimate_missing_prob}')
    print(f'eval_policy_mc_size : {eval_policy_mc_size}')
    print(f'eval_horizon : {eval_horizon}')
    print(f'scale_state : {scale_state}')
    print(f'Logged to folder: T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}')

    # eval_S_inits
    np.random.seed(seed=eval_seed)
    if env_class.lower() in ['linear','save']:
        if scale_state:
            eval_S_inits = np.random.uniform(low=0,
                                    high=1,
                                    size=(eval_policy_mc_size, env.dim))
        else:
            eval_S_inits = np.random.normal(loc=0,
                                    scale=1,
                                    size=(eval_policy_mc_size, env.dim))
        eval_kargs = {}

    default_key = 'C'
    if eval_S_inits is not None:
        eval_S_inits = np.clip(eval_S_inits, a_min=low, a_max=high)
        eval_S_inits_dict = {default_key: eval_S_inits}
    else:
        eval_S_inits_dict = {default_key: eval_kargs}

    ## policy
    if env_class.lower() in ['linear','save'] and scale_state:
        action_thres = 0.5
        def target_policy(S):
            if S[0] > action_thres and S[1] > action_thres:
                return [0.,1.]
            else:
                return [1.,0.]

        def vec_target_policy(S):
            if len(S.shape) == 1:
                S = S.reshape(1, -1)
            return np.where(((S[:, 0] > action_thres) & (S[:, 1] > action_thres)).reshape(-1, 1),
                            np.repeat([[0.,1.]], repeats=S.shape[0], axis=0),
                            np.repeat([[1.,0.]], repeats=S.shape[0], axis=0))
    else:
        if env_class.lower() == 'save':
            # target policy in SAVE paper
            def target_policy(S):
                if S[0] > 0 and S[1] > 0:
                    return [1, 0]
                else:
                    return [0, 1]

            def vec_target_policy(S):
                if len(S.shape) == 1:
                    S = S.reshape(1, -1)
                return np.where(((S[:, 0] > 0) & (S[:, 1] > 0)).reshape(-1, 1),
                                np.repeat([[1, 0]], repeats=S.shape[0], axis=0),
                                np.repeat([[0, 1]], repeats=S.shape[0], axis=0))
        elif env_class.lower() == 'linear':
            # our target policy
            def target_policy(S):
                if S[0] + S[1] > 0:
                    return [0, 1]
                else:
                    return [1, 0]

            def vec_target_policy(S):
                if len(S.shape) == 1:
                    S = S.reshape(1, -1)
                return np.where((S[:, 0] + S[:, 1] > 0).reshape(-1, 1),
                                np.repeat([[0, 1]], repeats=S.shape[0], axis=0),
                                np.repeat([[1, 0]], repeats=S.shape[0], axis=0))

    policy = vec_target_policy if vectorize_env else target_policy

    train_dir = os.path.join(log_dir,f'{env_class}_est_Q_func{folder_suffix}/T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}')
    value_dir = os.path.join(log_dir,f'{env_class}_est_value{folder_suffix}/T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}')
    true_value_dir = os.path.join(log_dir,f'{env_class}_est_value{folder_suffix}')
    pathlib.Path(train_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(value_dir).mkdir(parents=True, exist_ok=True)

    for itr in range(mc_size):
        np.random.seed(itr)
        # if the observational space of the environemnt is bounded, the initial states will only be sampled from uniform distribution
        # if we still want a normal distribution, pass random initial states manually.
        if env_class.lower() in ['linear','save']:
            if scale_state:
                train_S_inits = np.random.uniform(low=0,
                                        high=1,
                                        size=(n, env.dim))
            else:
                train_S_inits = np.random.normal(loc=0,
                                        scale=1,
                                        size=(n, env.dim))
            train_S_inits = np.clip(train_S_inits, a_min=low, a_max=high)
        else:
            train_S_inits = None
        if ipw:
            suffix = f'ipw_{prop}_itr_{itr}'
        else:
            suffix = f'itr_{itr}'
        filename_train = f'train_with_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{suffix}'
        filename_value_int = f'value_int_with_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{suffix}'
        filename_value = f'value_with_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{suffix}'
        filename_data = f'train_data_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{suffix}.csv'
        filename_log = f'train_logger_with_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{suffix}.csv'  
        filename_true_value = f'{env_class}_true_value_T_{eval_horizon}_gamma{gamma}_size{eval_policy_mc_size}'
        
        print('Train Q-function...')
        start = time.time()
        converge = train_Q_func(
            T=T,
            n=n,
            env=None,
            basis_type=basis_type,
            L=dof,
            d=spline_degree,
            knots=knots,
            total_N=total_N,
            burn_in=burn_in,
            use_vector_env=vectorize_env, 
            iterative=solve_iteratively,
            error_bound=error_bound,
            max_itr=max_itr,
            target_policy=policy,
            export_dir=train_dir,
            scale=scale,
            product_tensor=product_tensor,
            discount=gamma,
            seed=itr, 
            S_inits=None, # train_S_inits, None
            S_inits_kwargs={},
            ipw=ipw,
            weight_curr_step=weight_curr_step,
            estimate_missing_prob=estimate_missing_prob,
            dropout_obs_count_thres=dropout_obs_count_thres,
            missing_mechanism=missing_mechanism,
            instrument_var_index=instrument_var_index,
            mnar_y_transform=mnar_y_transform,
            gamma_init=None if missing_mechanism=='mnar' and not initialize_with_gammaT else gamma_true,
            bandwidth_factor=bandwidth_factor,
            drop_last_TD=drop_last_TD,
            ridge_factor=ridge_factor,
            grid_search=grid_search,
            basis_scale_factor=basis_scale_factor,
            dropout_model_type=dropout_model_type,
            dropout_scale_obs=False, # True, False
            dropout_include_reward=True, # True, False
            model_suffix=suffix,
            prob_lbound=prob_lbound,
            eval_env=env,
            filename_data=filename_data,
            filename_log=filename_log,
            filename_train=filename_train,
            **env_dropout_kwargs)
        
        end = time.time()
        print('Finished! Elapsed time: %.3f mins' % ((end - start) / 60))
        
        if not converge:
            print('Not converged!')

        # evaluate the value only if the learning converged
        print('Evaluate target policy...')
        start = time.time()
        if converge:
            value_dict = get_target_value_multi(
                T=T,
                n=n,
                env=None, 
                eval_T = eval_horizon,
                vf_mc_size=eval_policy_mc_size,
                target_policy=policy,
                use_vector_env=vectorize_env, 
                import_dir=train_dir,
                filename_train=filename_train,
                filename_true_value=filename_true_value,
                export_dir=value_dir,
                value_import_dir=true_value_dir,
                figure_name=f'est_value_scatterplot_{method}_itr_{itr}.png',
                scale=scale,
                product_tensor=product_tensor,
                discount=gamma,
                eval_env=env,
                filename_value=filename_value,
                eval_S_inits_dict=eval_S_inits_dict,
                eval_seed=eval_seed, # itr
                **env_dropout_kwargs)
            
            print({
                'MeanSE':
                value_dict[eval_horizon][default_key]['MeanSE'],
                'actual_value_int':
                value_dict[eval_horizon][default_key]['actual_value_int'],
                'est_value_int':
                value_dict[eval_horizon][default_key]['est_value_int']
            })
        end = time.time()
        print('Finished! Elapsed time: %.3f mins \n' % ((end - start) / 60))
        _ = gc.collect()
