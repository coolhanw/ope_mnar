"""
Examine the coverage of CI for the integrated value.
"""

import os
import numpy as np
import pickle
from numpy.linalg import inv
from scipy.stats import norm, truncnorm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from ope_mnar.utils import VectorSimEnv
    from ope_mnar.main import eval_V_int_CI_multi, eval_V_int_CI_bootstrap_multi
except:
    import sys
    sys.path.append(os.path.expanduser('~/Projects/ope_mnar/ope_mnar'))
    sys.path.append(os.path.expanduser('~/Projects/ope_mnar'))
    from ope_mnar.utils import VectorSimEnv
    from ope_mnar.main import eval_V_int_CI_multi, eval_V_int_CI_bootstrap_multi

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str,
                    default='linear')  # 'linear', 'SAVE'
parser.add_argument('--max_episode_length', type=int, default=25)
parser.add_argument('--discount', type=float, default=0.8)
parser.add_argument('--num_trajs', type=int, default=500) # 250, 500
parser.add_argument('--burn_in', type=int, default=0)
parser.add_argument('--mc_size', type=int, default=250)
parser.add_argument('--eval_policy_mc_size', type=int,
                    default=10000)  # use 100 for test purpose
parser.add_argument('--eval_horizon', type=int, default=250)
parser.add_argument('--scale_state',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=False,
                    help="If True, state features are already scaled to [0,1]")
parser.add_argument('--dropout_scheme', type=str, default='3.19') 
parser.add_argument('--dropout_rate', type=float, default=0.9)
parser.add_argument('--dropout_obs_count_thres', type=int, default=2)
parser.add_argument('--ipw',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
parser.add_argument('--estimate_missing_prob',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
parser.add_argument('--bootstrap',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=False)
parser.add_argument('--vectorize_env',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
parser.add_argument('--basis_type', type=str, default='spline')
parser.add_argument('--log_suffix', type=str, default='')  
args = parser.parse_args()

if __name__ == '__main__':
    log_dir = os.path.expanduser('~/Projects/ope_mnar/output')
    env_class = args.env
    default_scaler = "MinMax"  # "NormCdf", "MinMax"

    T = args.max_episode_length
    n = args.num_trajs
    total_N = None  # T*n, None
    num_actions = 2
    state_dim = 2
    adaptive_dof = False
    dropout_rate = args.dropout_rate  # 0.9
    gamma = args.discount
    mc_size = args.mc_size
    burn_in = args.burn_in
    # alpha = args.alpha
    alpha_list = [0.05] # [0.05, 0.1, 0.2, 0.3, 0.4] 
    vectorize_env = args.vectorize_env
    dropout_scheme = args.dropout_scheme
    dropout_obs_count_thres = args.dropout_obs_count_thres
    ipw = args.ipw
    estimate_missing_prob = args.estimate_missing_prob
    weight_curr_step = True
    instrument_var_index = None
    mnar_y_transform = None
    bandwidth_factor = None
    if env_class.lower() in ['linear', 'save']:
        if dropout_scheme == '0':
            missing_mechanism = None
        elif dropout_scheme in ['3.19', '3.20']:
            missing_mechanism = 'mnar'
            instrument_var_index = 1
            if dropout_scheme == '3.19':
                bandwidth_factor = 7.5
            elif dropout_scheme == '3.20':
                bandwidth_factor = 2.5
        else:
            missing_mechanism = 'mar'

    log_suffix = args.log_suffix
    prob_lbound = 1e-2  # 1e-2, 1e-3
    gamma_true = None  # 1.5
    if missing_mechanism and missing_mechanism.lower() == 'mnar':
        plugin_gamma_true = False
        initialize_with_gammaT = True
    if not ipw:
        estimator = 'cc'
    elif not estimate_missing_prob:
        estimator = 'ipw_propT'
    else:
        estimator = 'ipw_propF'

    eval_policy_mc_size = args.eval_policy_mc_size  # 10000
    eval_horizon = args.eval_horizon  # 500
    initial_scenario_list = ['C']  # ['A', 'C', 'E']
    scale_state = args.scale_state
    scale = "Identity" if scale_state else default_scaler
    low = 0 if scale_state else -norm.ppf(0.999)  # -np.inf
    high = 1 if scale_state else norm.ppf(0.999)  # np.inf
    eval_seed = 123
    product_tensor = True
    spline_degree = 3
    if adaptive_dof:
        dof = max(4, int(np.sqrt((n * T)**(3 / 7))))  # degree of freedom
    else:
        dof = 7
    ridge_factor = 1e-3  # 1e-9
    folder_suffix = '_scaled' if scale_state else '_unscaled'
    folder_suffix += f'_missing{dropout_rate}'
    folder_suffix += f'_ridge{ridge_factor}'
    if env_class.upper() == 'SAVE':
        folder_suffix += '_SAVE'
    grid_search = False
    basis_scale_factor = 100
    basis_type = args.basis_type  # 'spline'
    if basis_scale_factor != 1:
        folder_suffix += f'_scale{int(basis_scale_factor)}'
    if basis_type != 'spline':
        folder_suffix += f'_{basis_type}'
    folder_suffix += log_suffix
    if scale_state or scale == default_scaler:
        knots = np.linspace(start=-spline_degree / (dof - spline_degree),
                            stop=1 + spline_degree / (dof - spline_degree),
                            num=dof + spline_degree +
                            1)  # take care of the boundary
    else:
        knots = 'equivdist'  # 'equivdist', None
    export_dir = os.path.join(
        log_dir,
        f'{env_class}_value_coverage{folder_suffix}/T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}'
    )

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
    print(f'Logged to folder: {export_dir}')

    mnar_scheme_list = ['3.19', '3.19-mar', '3.20', '3.20-mar']
    scheme_map = {
        '0': 'no dropout',
        '3.19': 'scheme (a)',
        '3.19-mar': 'scheme (a*)',
        '3.20': 'scheme (b)',
        '3.20-mar': 'scheme (b*)'
    }
    method_map = {'cc': 'CC', 'ipw_propT': 'IPW', 'ipw_propF': 'IPW(est)'}

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
            S_next = (2 * action - 1) * np.matmul(
                obs, np.array([[0.75, 0], [0, -0.75]]))
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
            weight = np.array([0, 0.5, 2, 1, -1 / 2]).reshape(-1, 1)
            reward = np.matmul(obs_nobs_action, weight)
            bias = np.zeros_like(reward) + 1 / 4
            reward += bias
            noise = rng.normal(loc=0., scale=0., size=reward.shape)
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
                                logit = intercept + 2.4 + 0.8 * obs_history[
                                    -2][0] - 1.5 * reward_history[-1]
                            elif dropout_obs_count_thres == 2:
                                logit = intercept + 3.2 + 0.8 * obs_history[
                                    -2][0] - 1.5 * reward_history[-1]
                            elif dropout_obs_count_thres == 1:
                                logit = intercept + 4 + 0.8 * obs_history[-2][
                                    0] - 1.5 * reward_history[-1]
                        else:
                            raise NotImplementedError
                else:
                    logit = intercept + 6.5 + 1. * obs_history[-2][
                        0] - 4.5 * reward_history[-1]
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
            if action_history.shape[1] <= dropout_obs_count_thres:
                return np.zeros(shape=obs_history.shape[0]).reshape(-1, 1)
            intercept = 3.5  # 3
            if dropout_scheme == '0':
                prob = np.zeros(shape=obs_history.shape[0])
            elif dropout_scheme == '3.19':
                if not scale_state:
                    logit = intercept + 2.4 + 0.8 * obs_history[:, -2, 0] - 1.5 * reward_history[:,-1]
                else:
                    logit = intercept + 6.5 + 1. * obs_history[:, -2,0] - 4.5 * reward_history[:,-1]
                prob = 1 / (np.exp(logit) + 1)
            elif dropout_scheme == '3.19-mar':
                if not scale_state:
                    logit = intercept + 1.5 + 0.8 * obs_history[:, -2,0] - 1.5 * reward_history[:,-2]
                else:
                    logit = intercept + 6.5 + 1. * obs_history[:, -2,0] - 4.5 * reward_history[:,-2]
                prob = 1 / (np.exp(logit) + 1)
            elif dropout_scheme == '3.20':
                if not scale_state:
                    logit = intercept + 3 - 1. * np.power(
                        obs_history[:, -2, 0], 2) - 1.5 * reward_history[:, -1]
                else:
                    logit = intercept + 6.5 + 0.5 * np.power(
                        obs_history[:, -2, 0], 2) - 4.5 * reward_history[:, -1]
                prob = 1 / (np.exp(logit) + 1)
            elif dropout_scheme == '3.20-mar':
                if not scale_state:
                    logit = intercept + 3 - 1. * np.power(
                        obs_history[:, -2, 0], 2) - 1.5 * reward_history[:, -2]
                else:
                    logit = intercept + 6.5 + 0.5 * np.power(
                        obs_history[:, -2, 0], 2) - 4.5 * reward_history[:, -2]
                prob = 1 / (np.exp(logit) + 1)
            else:
                raise NotImplementedError
            return prob.reshape(-1, 1)

        def dropout_model_0(obs_history, action_history, reward_history):
            return 0

        def vec_dropout_model_0(obs_history, action_history, reward_history):
            return np.zeros(shape=(obs_history.shape[0], 1))
    # environment used in our paper
    else:
        def vec_state_trans_model(obs, action, rng=np.random):
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
            if len(obs.shape) == 1:
                obs = obs.reshape(1, -1)
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            if not action.shape or len(action.shape) == 1:
                action = action.reshape(-1, 1)

            obs_action = np.concatenate([obs, action, obs * action], axis=1)
            trans_mat = np.array([[-1, 0], [0, 1], [0, 0], [2, 0], [0, -2]])
            S_next = np.matmul(obs_action, trans_mat).astype('float')
            S_next += rng.normal(loc=0.0, scale=0.5, size=S_next.shape)
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
            weight = np.array([0, 0.5, 2, 1, -1 / 2]).reshape(-1, 1)
            reward = np.matmul(obs_nobs_action, weight)
            bias = np.zeros_like(reward) + 1 / 4
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
                                logit = intercept + 2.4 + 0.8 * obs_history[
                                    -2][0] - 1.5 * reward_history[-1]
                            elif dropout_obs_count_thres == 2:
                                logit = intercept + 3.2 + 0.8 * obs_history[
                                    -2][0] - 1.5 * reward_history[-1]
                            elif dropout_obs_count_thres == 1:
                                logit = intercept + 4 + 0.8 * obs_history[-2][
                                    0] - 1.5 * reward_history[-1]
                        else:
                            raise NotImplementedError
                else:
                    logit = intercept + 6.5 + 1. * obs_history[-2][
                        0] - 4.5 * reward_history[-1]
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
            # if action_history.shape[1] <= dropout_obs_count_thres:
            if action_history.shape[1] < dropout_obs_count_thres:
                return np.zeros(shape=obs_history.shape[0]).reshape(-1, 1)
            intercept = 3.5  # 3
            if dropout_scheme == '0':
                prob = np.zeros(shape=obs_history.shape[0])
            elif dropout_scheme == '3.19':
                if not scale_state:
                    if T == 25:
                        if dropout_rate == 0.6:
                            logit = intercept + 7.0 + 0.8 * obs_history[:, -2,0] - 1.5 * reward_history[:,-1]
                            logit = intercept + 5.0 + 0.8 * obs_history[:, -2,0] - 1.5 * reward_history[:,-1]
                        elif dropout_rate == 0.8:
                            logit = intercept + 3.6 + 0.8 * obs_history[:, -2,0] - 1.5 * reward_history[:,-1]
                        elif dropout_rate == 0.9:
                            if dropout_obs_count_thres == 5:
                                logit = intercept + 2.4 + 0.8 * obs_history[:,-2,0] - 1.5 * reward_history[:, -1]
                            elif dropout_obs_count_thres == 2:
                                logit = intercept + 3.2 + 0.8 * obs_history[:,-2,0] - 1.5 * reward_history[:,-1]
                            elif dropout_obs_count_thres == 1:
                                logit = intercept + 4 + 0.8 * obs_history[:,-2,0] - 1.5 * reward_history[:,-1]
                        else:
                            raise NotImplementedError
                    elif T == 10:
                        if dropout_rate == 0.6:
                            if dropout_obs_count_thres == 5:
                                logit = intercept + 2 + 0.8 * obs_history[:,-2,0] - 1.5 * reward_history[:,-1]
                            elif dropout_obs_count_thres == 3:
                                logit = intercept + 3 + 0.8 * obs_history[:,-2,0] - 1.5 * reward_history[:,-1]
                            elif dropout_obs_count_thres == 2:
                                logit = intercept + 4 + 0.8 * obs_history[:,-2,0] - 1.5 * reward_history[:,-1]
                        elif dropout_rate == 0.7:
                            logit = intercept + 0. + 0.8 * obs_history[:, -2,0] - 1.5 * reward_history[:,-1]
                        elif dropout_rate == 0.8:
                            logit = intercept - 0.8 + 0.8 * obs_history[:, -2,0] - 1.5 * reward_history[:,-1]
                        elif dropout_rate == 0.9:
                            if dropout_obs_count_thres == 6:
                                logit = intercept - 2. + 0.8 * obs_history[:,-2,0] - 1.5 * reward_history[:,-1]
                            elif dropout_obs_count_thres == 5:
                                logit = intercept - 1.2 + 0.8 * obs_history[:,-2,0] - 1.5 * reward_history[:,-1]
                            elif dropout_obs_count_thres == 4:
                                logit = intercept - 0.9 + 0.8 * obs_history[:,-2,0] - 1.5 * reward_history[:,-1]
                            elif dropout_obs_count_thres == 3:
                                logit = intercept - 0.6 + 0.8 * obs_history[:,-2,0] - 1.5 * reward_history[:,-1]
                            elif dropout_obs_count_thres == 2:
                                logit = intercept - 0.2 + 0.8 * obs_history[:,-2,0] - 1.5 * reward_history[:,-1]
                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError
                else:
                    logit = intercept + 6.5 + 1. * obs_history[:, -2,0] - 4.5 * reward_history[:,-1]
                prob = 1 / (np.exp(logit) + 1)
            elif dropout_scheme == '3.19-mar':
                if not scale_state:
                    if T == 25:
                        if dropout_rate == 0.6:
                            logit = intercept + 4.5 + 0.8 * obs_history[:, -2,0] - 1.5 * reward_history[:,-2]
                        elif dropout_rate == 0.7:
                            logit = intercept + 3.2 + 0.8 * obs_history[:, -2,0] - 1.5 * reward_history[:,-2]
                        elif dropout_rate == 0.8:
                            logit = intercept + 2.5 + 0.8 * obs_history[:, -2,0] - 1.5 * reward_history[:,-2]
                        elif dropout_rate == 0.9:
                            if dropout_obs_count_thres == 5:
                                logit = intercept + 1.5 + 0.8 * obs_history[:,-2,0] - 1.5 * reward_history[:,-2]
                            elif dropout_obs_count_thres == 2:
                                logit = intercept + 2.3 + 0.8 * obs_history[:,-2,0] - 1.5 * reward_history[:,-2]
                            elif dropout_obs_count_thres == 1:
                                logit = intercept + 2.8 + 0.8 * obs_history[:,-2,0] - 1.5 * reward_history[:,-2] 
                        else:
                            raise NotImplementedError
                    elif T == 10:
                        if dropout_rate == 0.6:
                            if dropout_obs_count_thres == 5:
                                logit = intercept + 0. + 0.8 * obs_history[:,-2,0] - 1.5 * reward_history[:,-2]
                            elif dropout_obs_count_thres == 2:
                                logit = intercept + 2.5 + 0.8 * obs_history[:,-2,0] - 1.5 * reward_history[:,-2]
                        elif dropout_rate == 0.7:
                            logit = intercept - 1. + 0.8 * obs_history[:, -2,0] - 1.5 * reward_history[:,-2]
                        elif dropout_rate == 0.8:
                            logit = intercept - 2. + 0.8 * obs_history[:, -2,0] - 1.5 * reward_history[:,-2]
                        elif dropout_rate == 0.9:
                            logit = intercept - 3. + 0.8 * obs_history[:, -2,0] - 1.5 * reward_history[:,-2]  
                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError
                else:
                    logit = intercept + 6.5 + 1. * obs_history[:, -2, 0] - 4.5 * reward_history[:,-2]
                prob = 1 / (np.exp(logit) + 1)
            elif dropout_scheme == '3.20':
                if not scale_state:
                    if T == 25:
                        if dropout_rate == 0.6:
                            logit = intercept + 8 - 0.5 * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-1]
                        elif dropout_rate == 0.7:
                            logit = intercept + 5.5 - 0.5 * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-1]
                        elif dropout_rate == 0.8:
                            logit = intercept + 4.2 - 0.5 * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:, -1]
                        elif dropout_rate == 0.9:
                            if dropout_obs_count_thres == 5:
                                logit = intercept + 3 - 0.5 * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-1]
                            elif dropout_obs_count_thres == 2:
                                logit = intercept + 3.7 - 0.5 * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-1]
                            elif dropout_obs_count_thres == 1:
                                logit = intercept + 4.5 - 0.5 * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-1]
                        else:
                            raise NotImplementedError
                    elif T == 10:
                        if dropout_rate == 0.6:
                            if dropout_obs_count_thres == 5:
                                logit = intercept + 1.8 - 1. * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-1]
                            elif dropout_obs_count_thres == 2:
                                logit = intercept + 6 - 1. * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-1]
                        elif dropout_rate == 0.7:
                            logit = intercept + 0.5 - 1. * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-1]
                        elif dropout_rate == 0.8:
                            logit = intercept - 0.5 - 1. * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-1]
                            if dropout_obs_count_thres == 6:
                                logit = intercept - 2. - 1. * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-1]
                            elif dropout_obs_count_thres == 5:
                                logit = intercept - 1. - 1. * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-1]
                            elif dropout_obs_count_thres == 4:
                                logit = intercept - 0.4 - 1. * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-1]
                            elif dropout_obs_count_thres == 3:
                                logit = intercept - 0. - 1. * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-1]
                            elif dropout_obs_count_thres == 2:
                                logit = intercept + 0.2 - 1. * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-1]
                            elif dropout_obs_count_thres == 1:
                                logit = intercept + 0.5 - 1. * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-1]
                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError
                else:
                    logit = intercept + 6.5 + 0.5 * np.power(
                        obs_history[:, -2, 0], 2) - 4.5 * reward_history[:, -1]
                prob = 1 / (np.exp(logit) + 1)
            elif dropout_scheme == '3.20-mar':
                if not scale_state:
                    if T == 25:
                        if dropout_rate == 0.6:
                            logit = intercept + 8 - 0.5 * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-2]
                        elif dropout_rate == 0.7:
                            logit = intercept + 5.5 - 0.5 * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-2]
                        elif dropout_rate == 0.8:
                            logit = intercept + 4.2 - 0.5 * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-2]
                        elif dropout_rate == 0.9:
                            if dropout_obs_count_thres == 5:
                                logit = intercept + 3 - 0.5 * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-2]
                            elif dropout_obs_count_thres == 2:
                                logit = intercept + 3.7 - 0.5 * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-2]
                            elif dropout_obs_count_thres == 1:
                                logit = intercept + 4.5 - 0.5 * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-2]
                        else:
                            raise NotImplementedError
                    elif T == 10:
                        if dropout_rate == 0.6:
                            if dropout_obs_count_thres == 5:
                                logit = intercept + 1.3 - 1. * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-2]
                            elif dropout_obs_count_thres == 2:
                                logit = intercept + 6 - 1. * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-2]
                        elif dropout_rate == 0.7:
                            logit = intercept + 0.5 - 1. * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-2]
                        elif dropout_rate == 0.8:
                            logit = intercept - 0.5 - 1. * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-2]
                        elif dropout_rate == 0.9:
                            logit = intercept - 2. - 1. * np.power(obs_history[:, -2, 0], 2) - 1.5 * reward_history[:,-2]
                        else:
                            raise NotImplementedError
                    else:
                        raise NotImplementedError
                else:
                    logit = intercept + 6.5 + 0.5 * np.power(
                        obs_history[:, -2, 0], 2) - 4.5 * reward_history[:, -2]
                prob = 1 / (np.exp(logit) + 1)
            else:
                raise NotImplementedError
            return prob.reshape(-1, 1)

        def dropout_model_0(obs_history, action_history, reward_history):
            return 0

        def vec_dropout_model_0(obs_history, action_history, reward_history):
            return np.zeros(shape=(obs_history.shape[0], 1))

    # specify environment
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
    else:
        raise NotImplementedError

    # specify policy
    if env_class.lower() in ['linear', 'save'] and scale_state:
        action_thres = 0.5

        def target_policy(S):
            if S[0] > action_thres and S[1] > action_thres:
                return [0., 1.]
            else:
                return [1., 0.]

        def vec_target_policy(S):
            if len(S.shape) == 1:
                S = S.reshape(1, -1)
            return np.where(
                ((S[:, 0] > action_thres) & (S[:, 1] > action_thres)).reshape(
                    -1, 1), np.repeat([[0., 1.]], repeats=S.shape[0], axis=0),
                np.repeat([[1., 0.]], repeats=S.shape[0], axis=0))
    else:
        if env_class.upper() == 'SAVE':
            def target_policy(S):
                if S[0] > 0 and S[1] > 0:
                    return [1, 0]
                else:
                    return [0, 1]

            def vec_target_policy(S):
                if len(S.shape) == 1:
                    S = S.reshape(1, -1)
                return np.where(
                    ((S[:, 0] > 0) & (S[:, 1] > 0)).reshape(-1, 1),
                    np.repeat([[1, 0]], repeats=S.shape[0], axis=0),
                    np.repeat([[0, 1]], repeats=S.shape[0], axis=0))
        # target policy used in our paper
        elif env_class.lower() == 'linear':
            def target_policy(S):
                if S[0] + S[1] > 0:
                    return [0, 1]
                else:
                    return [1, 0]

            def vec_target_policy(S):
                if len(S.shape) == 1:
                    S = S.reshape(1, -1)
                return np.where(
                    (S[:, 0] + S[:, 1] > 0).reshape(-1, 1),
                    np.repeat([[0, 1]], repeats=S.shape[0], axis=0),
                    np.repeat([[1, 0]], repeats=S.shape[0], axis=0))

        policy = vec_target_policy if vectorize_env else target_policy

    ## eval_S_inits
    np.random.seed(seed=eval_seed)
    if env_class.lower() in ['linear', 'save']:
        eval_S_inits_dict = {}
        for initial_scenario in initial_scenario_list:
            if initial_scenario.lower() == 'a':
                eval_S_inits = np.tile(np.array([[0.5, 0.5]]),
                                       reps=(eval_policy_mc_size, 1))
            elif initial_scenario.lower() == 'b':
                eval_S_inits = np.tile(np.array([[-0.5, -0.5]]),
                                       reps=(eval_policy_mc_size, 1))
            elif initial_scenario.lower() == 'c':
                eval_S_inits = np.random.normal(loc=0,
                                                scale=1,
                                                size=(eval_policy_mc_size,
                                                      env.dim))
            elif initial_scenario.lower() == 'd':
                eval_S_inits = np.clip(np.random.normal(
                    loc=0, scale=1, size=(eval_policy_mc_size, env.dim)),
                                       a_min=-2,
                                       a_max=2)
            elif initial_scenario.lower() == 'e':
                # truncated normal
                eval_S_inits = truncnorm.rvs(-2,
                                             2,
                                             size=(eval_policy_mc_size,
                                                   env.dim))
            eval_S_inits = np.clip(eval_S_inits, a_min=low, a_max=high)
            eval_S_inits_dict[initial_scenario] = eval_S_inits
        print(eval_S_inits_dict)
        eval_kargs = {}

    if total_N is not None:
        export_dir += '_fixTotal'
    if estimator.lower() != 'cc':
        export_dir += f'_Ubound{round(1/prob_lbound)}'
    if not args.bootstrap:
        filename_true_value = f'{env_class}_true_value_int_gamma{gamma}_size{eval_policy_mc_size}'
        filename_est_value = f'est_value_int_gamma{gamma}'
        filename_CI = f'CI_V_int_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}'  # .pkl
        result_filename = f'result_T_{T}_n_{n}_S_integration_L_{dof}_{estimator}.txt'
        figure_name = f"est_value_scatterplot_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}.png"
        eval_V_int_CI_multi(
            T=T,
            n=n,
            env=None,
            basis_type=basis_type,
            L=dof,
            d=spline_degree,
            eval_T=eval_horizon,
            discount=gamma,
            knots=knots,
            total_N=total_N,
            burn_in=burn_in,
            use_vector_env=True,
            ipw=ipw,
            weight_curr_step=weight_curr_step,
            estimate_missing_prob=estimate_missing_prob,
            prob_lbound=prob_lbound,  # 1e-3, None
            ridge_factor=ridge_factor,
            train_S_inits=None,
            train_S_inits_kwargs={},
            basis_scale_factor=basis_scale_factor,
            vf_mc_size=eval_policy_mc_size,
            mc_size=mc_size,
            alpha_list=alpha_list,
            target_policy=policy,
            dropout_model_type='linear',
            dropout_obs_count_thres=dropout_obs_count_thres,
            dropout_scale_obs=False,
            dropout_include_reward=True, # True, False
            missing_mechanism=missing_mechanism,
            instrument_var_index=instrument_var_index,
            mnar_y_transform=mnar_y_transform,
            gamma_init=None if missing_mechanism == 'mnar'
            and not initialize_with_gammaT else gamma_true,
            bandwidth_factor=bandwidth_factor,
            value_import_dir=os.path.join(
                log_dir, f'{env_class}_value_coverage{folder_suffix}'),
            export_dir=export_dir,
            filename_true_value=filename_true_value,
            filename_est_value=filename_est_value,
            filename_CI=filename_CI,
            figure_name=figure_name,
            result_filename=result_filename,
            scale=scale,
            product_tensor=product_tensor,
            eval_env=env,
            eval_S_inits_dict=eval_S_inits_dict,
            eval_seed=eval_seed,
            verbose_freq=1,
            **env_dropout_kwargs)

        # visualize V_int estimation
        with open(os.path.join(export_dir, filename_est_value), 'rb') as f:
            est_value_dict = pickle.load(f)
        initial_states = est_value_dict['initial_states']
        est_value_int = est_value_dict['est_V_int_list']
        est_value_int_std = est_value_dict['est_V_int_std_list']
        true_value_int = est_value_dict['true_V_int']
        est_V_mse = est_value_dict['est_V_mse_list']
        max_inverse_wt_list = est_value_dict['max_inverse_wt_list']
        mnar_gamma_est_list = est_value_dict['mnar_gamma_est_list']
        dropout_prob_mse_list = est_value_dict['dropout_prob_mse_list']
        for initial_scenario in initial_scenario_list:
            nrows, ncols = 1, 2
            fig, ax = plt.subplots(nrows=nrows,
                                   ncols=ncols,
                                   figsize=(4 * ncols, 4 * nrows))
            sns.histplot(data=est_value_int[initial_scenario],
                         ax=ax[0],
                         stat='probability')
            ax[0].axvline(true_value_int[initial_scenario], color='red')
            ax[0].set_title(
                f'V hat ({scheme_map[dropout_scheme]}, {estimator}, n={n})')
            sns.histplot(data=est_value_int_std[initial_scenario],
                         ax=ax[1],
                         stat='probability')
            ax[1].set_title(
                f'V hat std ({scheme_map[dropout_scheme]}, {estimator}, n={n})'
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    export_dir,
                    f"value_est_dist_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}_init_{initial_scenario}.png"
                ))
            plt.close()
            # scatterplot: max inverse weight vs. MSE
            if ipw:
                ax = sns.scatterplot(x=est_V_mse[initial_scenario],
                                     y=max_inverse_wt_list)
                ax.set_title(
                    f'scheme {scheme_map[dropout_scheme]}, {method_map[estimator]}, n={n}'
                )
                ax.set_xlabel('V mse')
                ax.set_ylabel('maximum inverse weight')
                plt.savefig(
                    os.path.join(
                        export_dir,
                        f"value_vs_maxInvWt_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}_init_{initial_scenario}.png"
                    ))
                plt.close()
    else:
        filename_true_value = f'{env_class}_true_value_int_gamma{gamma}_size{eval_policy_mc_size}'
        filename_est_value = f'est_value_int_gamma{gamma}'
        filename_CI = f'CI_V_int_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}'  # .pkl
        filename_bootstrap_CI = f'bootstrap_CI_V_int_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}'  # .pkl
        result_filename = f'result_T_{T}_n_{n}_S_integration_L_{dof}_{estimator}.txt'
        figure_name = f"est_value_scatterplot_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}.png"
        eval_V_int_CI_bootstrap_multi(
            T=T,
            n=n,
            env=None,
            basis_type=basis_type,
            L=dof,
            d=spline_degree,
            eval_T=eval_horizon,
            discount=gamma,
            knots=knots,
            total_N=total_N,
            burn_in=burn_in,
            use_vector_env=True,
            ipw=ipw,
            weight_curr_step=weight_curr_step,
            estimate_missing_prob=estimate_missing_prob,
            prob_lbound=prob_lbound,  # 1e-3, None
            ridge_factor=ridge_factor,
            train_S_inits=None,
            train_S_inits_kwargs={},
            basis_scale_factor=basis_scale_factor,
            vf_mc_size=eval_policy_mc_size,
            mc_size=mc_size,
            alpha_list=alpha_list,
            bootstrap_size=25,
            target_policy=policy,
            dropout_model_type='linear',
            dropout_obs_count_thres=dropout_obs_count_thres,
            dropout_scale_obs=False,
            dropout_include_reward=True, # True, False
            missing_mechanism=missing_mechanism,
            instrument_var_index=instrument_var_index,
            mnar_y_transform=mnar_y_transform,
            gamma_init=None if missing_mechanism == 'mnar'
            and not initialize_with_gammaT else gamma_true,
            bandwidth_factor=bandwidth_factor,
            value_import_dir=os.path.join(
                log_dir, f'{env_class}_value_coverage{folder_suffix}'),
            export_dir=export_dir,
            filename_true_value=filename_true_value,
            filename_est_value=filename_est_value,
            filename_CI=filename_CI,
            filename_bootstrap_CI=filename_bootstrap_CI,
            figure_name=figure_name,
            result_filename=result_filename,
            scale=scale,
            product_tensor=product_tensor,
            eval_env=env,
            eval_S_inits_dict=eval_S_inits_dict,
            eval_seed=eval_seed,
            verbose_freq=1,
            **env_dropout_kwargs)

        with open(os.path.join(export_dir, filename_est_value), 'rb') as f:
            est_value_dict = pickle.load(f)
        initial_states = est_value_dict['initial_states']
        est_value_int = est_value_dict['est_V_int_list']
        est_value_int_std = est_value_dict['est_V_int_std_list']
        bootstrap_value_int_std = est_value_dict['bootstrap_V_int_std_list']
        true_value_int = est_value_dict['true_V_int']
        est_V_mse = est_value_dict['est_V_mse_list']
        max_inverse_wt_list = est_value_dict['max_inverse_wt_list']
        mnar_gamma_est_list = est_value_dict['mnar_gamma_est_list']
        dropout_prob_mse_list = est_value_dict['dropout_prob_mse_list']
        # histograms to visualize V_int estimation
        for initial_scenario in initial_scenario_list:
            nrows, ncols = 1, 2
            fig, ax = plt.subplots(nrows=nrows,
                                   ncols=ncols,
                                   figsize=(4 * ncols, 4 * nrows))
            sns.histplot(data=est_value_int[initial_scenario],
                         ax=ax[0],
                         stat='probability')
            ax[0].axvline(true_value_int[initial_scenario], color='red')
            ax[0].set_title(
                f'V hat ({scheme_map[dropout_scheme]}, {estimator}, n={n})')
            sns.histplot(data=bootstrap_value_int_std[initial_scenario],
                         ax=ax[1],
                         stat='probability')
            ax[1].axvline(np.std(est_value_int[initial_scenario], ddof=1),
                          color='red')
            ax[1].set_title(
                f'V hat std ({scheme_map[dropout_scheme]}, {estimator}, n={n})'
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    export_dir,
                    f"value_est_dist_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}_init_{initial_scenario}.png"
                ))
            plt.close()
            # scatterplot: max inverse weight vs. MSE
            if ipw:
                ax = sns.scatterplot(x=est_V_mse[initial_scenario],
                                     y=max_inverse_wt_list)
                ax.set_title(
                    f'scheme {scheme_map[dropout_scheme]}, {method_map[estimator]}, n={n}'
                )
                ax.set_xlabel('V mse')
                ax.set_ylabel('maximum inverse weight')
                plt.savefig(
                    os.path.join(
                        export_dir,
                        f"value_vs_maxInvWt_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}_init_{initial_scenario}.png"
                    ))
                plt.close()

    if ipw and estimate_missing_prob and missing_mechanism == 'mnar':
        mnar_gamma_est_list = np.hstack(mnar_gamma_est_list)
        nrows, ncols = 1, 2
        fig, ax = plt.subplots(nrows=nrows,
                               ncols=ncols,
                               figsize=(4 * ncols, 4 * nrows))
        sns.histplot(data=mnar_gamma_est_list, stat='probability', ax=ax[0])
        ax[0].axvline(1.5, color='red')
        ax[0].set_title(f'gamma est')
        sns.histplot(data=dropout_prob_mse_list, stat='probability', ax=ax[1])
        ax[1].set_title(f'prob est mse')
        plt.savefig(
            os.path.join(
                export_dir,
                f"mnar_est_gamma_T_{T}_n_{n}_L_{dof}_gamma{gamma}_dropout{dropout_scheme}_{estimator}.png"
            ))
        plt.close()