"""
Examine the point-estimator of value estimator.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import argparse
import pathlib
import time
import gc
import torch
import torch.nn as nn
torch.manual_seed(0) # for better reproducibility

try:
    from ope_mnar.utils import InitialStateSampler, DiscretePolicy, MinMaxScaler
    from custom_env.linear2d import Linear2dEnv, Linear2dVectorEnv
    from ope_mnar.importance_sampling import MWL, NeuralDualDice
    from ope_mnar.direct_method import FQE, LSTDQ
    from ope_mnar.doubly_robust import DRL
except:
    import sys
    sys.path.append(os.path.expanduser('~/Projects/ope_mnar/ope_mnar'))
    sys.path.append(os.path.expanduser('~/Projects/ope_mnar'))
    from ope_mnar.utils import InitialStateSampler, DiscretePolicy, MinMaxScaler
    from custom_env.linear2d import Linear2dEnv, Linear2dVectorEnv
    from ope_mnar.importance_sampling import MWL, NeuralDualDice
    from ope_mnar.direct_method import FQE, LSTDQ
    from ope_mnar.doubly_robust import DRL

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='linear2d')
parser.add_argument('--method', type=str, default='dualdice', choices=["mwl", "fqe", "lstdq", "dualdice", "drl"])
parser.add_argument('--max_episode_length', type=int, default=25)  # 10, 25
parser.add_argument('--discount', type=float, default=0.8)
parser.add_argument('--num_trajs', type=int, default=500) # 250, 500
parser.add_argument('--burn_in', type=int, default=0)
parser.add_argument('--mc_size', type=int, default=250)
parser.add_argument('--eval_policy_mc_size', type=int, default=10000)
parser.add_argument('--eval_horizon', type=int, default=250)
parser.add_argument('--dropout_scheme', type=str, default='0', choices=["0", "3.19", "3.19-mar"])
parser.add_argument('--dropout_rate', type=float, default=0.9)
parser.add_argument(
    '--dropout_obs_count_thres',
    type=int,
    default=2,
    help="the number of observations that is not subject to dropout")
parser.add_argument('--ipw',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=False)
parser.add_argument('--estimate_missing_prob',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=False)
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
    ope_method = args.method.lower()
    T = args.max_episode_length
    n = args.num_trajs
    total_N = None
    default_scaler = "MinMax" # "NormCdf", "MinMax"
    gamma = args.discount
    mc_size = args.mc_size
    dropout_scheme = args.dropout_scheme
    dropout_rate = args.dropout_rate
    burn_in = args.burn_in
    vectorize_env = args.vectorize_env
    ipw = args.ipw
    weight_curr_step = args.weight_curr_step  # if False, use survival probability
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
    if env_class.lower() == 'linear2d':
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

    gamma_true = None  # 1.5, None
    if missing_mechanism and missing_mechanism.lower() == 'mnar':
        initialize_with_gammaT = False

    prop = 'propT' if not estimate_missing_prob else 'propF'
    eval_policy_mc_size = args.eval_policy_mc_size  # 10000
    eval_horizon = args.eval_horizon  # 500
    folder_suffix = ''
    folder_suffix += f'_missing{dropout_rate}'

    low = -np.inf
    high = np.inf
    eval_seed = 123
    dropout_model_type = 'linear'
    prob_lbound = 1e-2
    omega_func_class = 'expo_linear' # marginalized density ratio, {'nn', 'spline', 'expo_linear'}
    Q_func_class = 'rf' # {'nn', 'rf', 'spline'}
    
    if ope_method == 'lstdq':
        Q_func_class = 'spline' # override
    
    if omega_func_class in ['spline', 'expo_linear']:
        # spline related configuration
        basis_scale_factor = 100 # 1, 100
        basis_type = 'spline'
        spline_degree = 3
        if adaptive_dof:
            dof = max(4, int(np.sqrt((n * T)**(3/7)))) # degree of freedom
        else:
            dof = 7
        ridge_factor = 1e-3 # 1e-3
        folder_suffix += f'_ridge{ridge_factor}'
        if basis_scale_factor != 1:
            folder_suffix += f'_scale{int(basis_scale_factor)}'
        if basis_type != 'spline':
            folder_suffix += f'_{basis_type}'
        if default_scaler == "MinMax":
            knots = np.linspace(start=-spline_degree/(dof-spline_degree), stop=1+spline_degree/(dof-spline_degree), num=dof+spline_degree+1) # handle the boundary
        else:
            knots = 'equivdist' # 'equivdist', None

    print('Configuration:')
    print(f'T : {T}')
    print(f'n : {n}')
    print(f'total_N : {total_N}')
    print(f'gamma : {gamma}')
    print(f'dropout_scheme : {dropout_scheme}')
    print(f'ipw : {ipw}')
    print(f'estimate_missing_prob : {estimate_missing_prob}')
    print(f'eval_policy_mc_size : {eval_policy_mc_size}')
    print(f'eval_horizon : {eval_horizon}')
    print(
        f'Logged to folder: T_{T}_n_{n}_gamma{gamma}_dropout{dropout_scheme}')


    np.random.seed(seed=eval_seed)
    default_key = 'C'

    if env_class.lower() == 'linear2d':
        # specify env and env_dropout
        low = -norm.ppf(0.999)  # -np.inf
        high = norm.ppf(0.999)  # np.inf
        num_actions = 2

        if vectorize_env:
            env = Linear2dVectorEnv(
                num_envs=n,
                T=T,
                dropout_scheme='0',
                dropout_rate=0.,
                dropout_obs_count_thres=dropout_obs_count_thres,
                low=low,
                high=high)
            env_dropout = Linear2dVectorEnv(
                num_envs=n,
                T=T,
                dropout_scheme=dropout_scheme,
                dropout_rate=dropout_rate,
                dropout_obs_count_thres=dropout_obs_count_thres,
                low=low,
                high=high)
        else:            
            env = Linear2dEnv(
                T=T,
                dropout_scheme='0',
                dropout_rate=0.,
                dropout_obs_count_thres=dropout_obs_count_thres,
                low=low,
                high=high)
            env_dropout = Linear2dEnv(
                T=T,
                dropout_scheme=dropout_scheme,
                dropout_rate=dropout_rate,
                dropout_obs_count_thres=dropout_obs_count_thres,
                low=low,
                high=high)
    
        # eval_S_inits
        eval_S_inits = np.random.normal(loc=0,
                                        scale=1,
                                        size=(eval_policy_mc_size, env.dim))
        eval_S_inits = np.clip(eval_S_inits, a_min=low, a_max=high)
        eval_S_inits_dict = {default_key: eval_S_inits}

        # specify target policy
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
    else:
        raise NotImplementedError


    train_dir = os.path.join(
        log_dir,
        f'{env_class}_{ope_method}_est_Q_func{folder_suffix}/T_{T}_n_{n}_gamma{gamma}_dropout{dropout_scheme}'
    )
    value_dir = os.path.join(
        log_dir,
        f'{env_class}_{ope_method}_est_value{folder_suffix}/T_{T}_n_{n}_gamma{gamma}_dropout{dropout_scheme}'
    )
    true_value_dir = os.path.join(
        log_dir, f'{env_class}_{ope_method}_est_value{folder_suffix}')
    pathlib.Path(train_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(value_dir).mkdir(parents=True, exist_ok=True)
    filename_true_value = f'{env_class}_true_value_T_{eval_horizon}_gamma{gamma}_size{eval_policy_mc_size}'

    V_est_list = []
    true_V_list = []
    verbose = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for itr in range(mc_size):
        seed = itr
        # np.random.seed(seed)
        # if the observational space of the environemnt is bounded, the initial states will only be sampled from uniform distribution
        # if we still want a normal distribution, pass random initial states manually.
        if env_class.lower() == 'linear2d':
            train_S_inits = np.random.normal(loc=0, scale=1, size=(n, env.dim))
            train_S_inits = np.clip(train_S_inits, a_min=low, a_max=high)
        else:
            train_S_inits = None

        if ipw:
            suffix = f'ipw_{prop}_itr_{itr}'
        else:
            suffix = f'itr_{itr}'

        # filename_train = f'train_with_T_{T}_n_{n}_gamma{gamma}_dropout{dropout_scheme}_{suffix}'
        # filename_value_int = f'value_int_with_T_{T}_n_{n}_gamma{gamma}_dropout{dropout_scheme}_{suffix}'
        # filename_value = f'value_with_T_{T}_n_{n}_gamma{gamma}_dropout{dropout_scheme}_{suffix}'
        model_suffix = suffix

        # generate data
        if ope_method == 'mwl':
            agent = MWL(
                env=env_dropout,
                n=n,
                horizon=T + burn_in,
                discount=gamma,
                eval_env=env,
                device=device,
                seed=seed
            )
        elif ope_method == 'dualdice':
            agent = NeuralDualDice(
                env=env_dropout,
                n=n,
                horizon=T + burn_in,
                discount=gamma,
                eval_env=env,
                device=device,
                seed=seed
            )
        elif ope_method == 'fqe':
            agent = FQE(
                env=env_dropout,
                n=n,
                horizon=T + burn_in,
                discount=gamma,
                eval_env=env,
                device=device,
                seed=seed
            )
        elif ope_method == 'lstdq':
            agent = LSTDQ(
                env=env_dropout,
                n=n,
                horizon=T + burn_in,
                scale="MinMax",
                product_tensor=True,
                discount=gamma,
                eval_env=env,
                basis_scale_factor=basis_scale_factor
            ) # TODO: put scale, basis_scale_factor, product_tensor to self.estimate_Q()
        elif ope_method == 'drl':
            agent = DRL(
                env=env_dropout,
                n=n,
                horizon=T + burn_in,
                discount=gamma,
                eval_env=env,
                device=device,
                seed=seed
            )
        agent.dropout_obs_count_thres = max(dropout_obs_count_thres - 1, 0)  # -1 because this is the index
        agent.gen_masked_buffer(policy=agent.obs_policy,
                                S_inits=None,
                                total_N=total_N,
                                burn_in=burn_in,
                                seed=seed)
        print(f'dropout rate: {agent.dropout_rate}')
        print(f'missing rate: {agent.missing_rate}')
        print(f'n: {agent.n}')
        print(f'total_N: {agent.total_N}')

        # print(agent.masked_buffer[0])
        # print(agent.masked_buffer[1])

        if missing_mechanism is None:
            # no missingness, hence no need of adjustment
            ipw = False
            estimate_missing_prob = False

        if estimate_missing_prob:
            print(f'Fit dropout model : {missing_mechanism}')
            pathlib.Path(os.path.join(train_dir, 'models')).mkdir(parents=True,
                                                                exist_ok=True)
            print('Start fitting dropout model...')
            fit_dropout_start = time.time()
            agent.train_dropout_model(
                model_type=dropout_model_type,
                missing_mechanism=missing_mechanism,
                train_ratio=0.8,
                scale_obs=False,
                dropout_obs_count_thres=dropout_obs_count_thres,
                export_dir=os.path.join(train_dir, 'models'),
                pkl_filename=
                f"dropout_model_{dropout_model_type}_T{T}_n{n}_gamma{gamma}_{model_suffix}.pkl",
                seed=seed,
                include_reward=True,
                instrument_var_index=instrument_var_index,
                mnar_y_transform=mnar_y_transform,
                gamma_init=None if missing_mechanism=='mnar' and not initialize_with_gammaT else gamma_true,
                bandwidth_factor=bandwidth_factor,
                verbose=True)
            agent.estimate_missing_prob(missing_mechanism=missing_mechanism)
            fit_dropout_end = time.time()
            print(f'Finished! {fit_dropout_end-fit_dropout_start} secs elapsed.')

        # estimate value
        if ope_method == 'mwl': # minimax weight learning
            if omega_func_class == 'nn':
                agent.estimate_omega(
                    target_policy=DiscretePolicy(policy_func=policy, num_actions=num_actions),
                    initial_state_sampler=InitialStateSampler(initial_states=eval_S_inits_dict[default_key], seed=seed),
                    #   initial_state_sampler=InitialStateSampler(initial_states=agent._initial_obs, seed=seed),
                    func_class='nn',
                    hidden_sizes=[64, 64],
                    separate_action=True, # True, False
                    max_iter=500,
                    batch_size=32,
                    lr=0.0005,
                    ipw=ipw,
                    prob_lbound=prob_lbound,
                    print_freq=50,
                    patience=50, # 10
                    verbose=verbose
                )
            elif omega_func_class == 'spline':
                agent.estimate_omega(
                    target_policy=DiscretePolicy(policy_func=policy, num_actions=num_actions),
                    initial_state_sampler=InitialStateSampler(initial_states=eval_S_inits_dict[default_key], seed=seed),
                    #   initial_state_sampler=InitialStateSampler(initial_states=agent._initial_obs, seed=seed),
                    func_class='spline',
                    ipw=ipw,
                    prob_lbound=prob_lbound,
                    scaler = MinMaxScaler(min_val=env.low, max_val=env.high) if env is not None else MinMaxScaler(),
                    # spline fitting related arguments
                    ridge_factor=ridge_factor, 
                    L=max(3, dof), 
                    d=spline_degree, 
                    knots=knots,
                    product_tensor=True,
                    basis_scale_factor=basis_scale_factor,
                    verbose=verbose
                )
            elif omega_func_class == 'expo_linear':
                agent.estimate_omega(
                    target_policy=DiscretePolicy(policy_func=policy, num_actions=num_actions),
                    initial_state_sampler=InitialStateSampler(initial_states=eval_S_inits_dict[default_key], seed=seed),
                    #   initial_state_sampler=InitialStateSampler(initial_states=agent._initial_obs, seed=seed),
                    func_class='expo_linear',
                    ipw=ipw,
                    prob_lbound=prob_lbound,
                    scaler = MinMaxScaler(min_val=env.low, max_val=env.high) if env is not None else MinMaxScaler(),
                    L=dof, 
                    d=spline_degree, 
                    knots=knots,
                    product_tensor=True,
                    basis_scale_factor=1, # to ensure the input lies in a reasonable range, otherwise the training is not as stable
                    lr=0.05,
                    batch_size=512,
                    max_iter=2000,
                    print_freq=50,
                    patience=50,
                    verbose=verbose
                )

            value_est = agent.get_value()
            agent.validate_visitation_ratio(grid_size=10, visualize=True)
        elif ope_method == 'dualdice':
            zeta_pos = False
            nu_network_kwargs = {
                'hidden_sizes': [64, 64], 
                'hidden_nonlinearity': nn.ReLU(), 
                'hidden_w_init': nn.init.xavier_uniform_, 
                'output_w_inits': nn.init.xavier_uniform_
            }
            zeta_network_kwargs = {
                'hidden_sizes': [64, 64], 
                'hidden_nonlinearity': nn.ReLU(), 
                'hidden_w_init': nn.init.xavier_uniform_, 
                'output_w_inits': nn.init.xavier_uniform_, 
                # 'output_nonlinearities': nn.Identity()
            }

            agent.estimate_omega(
                target_policy=DiscretePolicy(policy_func=policy, num_actions=num_actions),
                initial_state_sampler=InitialStateSampler(initial_states=eval_S_inits_dict[default_key], seed=seed),
                nu_network_kwargs=nu_network_kwargs, 
                nu_learning_rate=0.0001,
                zeta_network_kwargs=zeta_network_kwargs,
                zeta_learning_rate=0.0001,
                zeta_pos=zeta_pos,
                solve_for_state_action_ratio=True,
                max_iter=5000,
                batch_size=1024,
                f_exponent=2,
                primal_form=False,
                print_freq=50,
                verbose=verbose
            )
            value_est = agent.get_value()
            agent.validate_visitation_ratio(grid_size=10, visualize=True)
        elif ope_method == 'fqe': # fitted Q-evalution
            if Q_func_class == 'nn':
                agent.estimate_Q(
                    target_policy=DiscretePolicy(policy_func=policy, num_actions=num_actions),
                    max_iter=200,
                    tol=0.001,
                    use_RF=False,
                    hidden_sizes=[64,64],
                    lr=0.001,
                    batch_size=128,
                    epoch=50, # 100
                    patience=10,
                    scaler="Standard",
                    print_freq=10,
                    verbose=verbose
                )
            elif Q_func_class == 'rf':
                agent.estimate_Q(
                    target_policy=DiscretePolicy(policy_func=policy, num_actions=num_actions),
                    max_iter=200,
                    tol=0.001,
                    use_RF=True,
                    scaler="Standard",
                    n_estimators=250,
                    max_depth=15, 
                    min_samples_leaf=10,
                    verbose=verbose
                )
            
            # value_est = np.mean(agent.get_state_values(S_inits=eval_S_inits_dict[default_key]))
            value_est = agent.get_value(S_inits=eval_S_inits_dict[default_key])
            agent.validate_Q(grid_size=10, visualize=True)
        elif ope_method == 'lstdq':
            assert Q_func_class == 'spline'
            agent.estimate_Q(target_policy=policy,
                            ipw=ipw,
                            estimate_missing_prob=estimate_missing_prob,
                            weight_curr_step=weight_curr_step,
                            prob_lbound=prob_lbound,
                            ridge_factor=ridge_factor,
                            L=max(3, dof),
                            d=spline_degree,
                            knots=knots,
                            grid_search=False,
                            verbose=True)
            value_est = agent.get_value(S_inits=eval_S_inits_dict[default_key])
            print(value_est)
            value_interval_est = agent.get_value_interval(S_inits=eval_S_inits_dict[default_key], alpha=0.05)
            print(value_interval_est)
            agent.validate_Q(grid_size=10, visualize=True)

        elif ope_method == 'drl':
            common_kwargs = {
                'env': env_dropout,
                'n': n,
                'horizon': T + burn_in,
                'discount': gamma,
                'eval_env': env,
                'device': device,
                'seed': seed
            }
            estimate_omega_kwargs = {
                'func_class': 'spline',
                'ipw': ipw,
                'prob_lbound': prob_lbound,
                'scaler': MinMaxScaler(min_val=env.low, max_val=env.high) if env is not None else MinMaxScaler(),
                'ridge_factor': ridge_factor, 
                'L': max(3, dof), 
                'd': spline_degree, 
                'knots': knots,
                'product_tensor': True,
                'basis_scale_factor': basis_scale_factor,
                'verbose': verbose
            }
            estimate_Q_kwargs = {
                'max_iter': 200,
                'tol': 0.001,
                'use_RF': True,
                'scaler': "Standard",
                'n_estimators': 250,
                'max_depth': 15, 
                'min_samples_leaf': 10,
                'verbose': verbose
            }
            omega_estimator = MWL(**common_kwargs)
            Q_estimator = FQE(**common_kwargs)
            agent.estimate_omega(
                omega_estimator=omega_estimator, 
                target_policy=DiscretePolicy(policy_func=policy, num_actions=num_actions),
                initial_state_sampler=InitialStateSampler(initial_states=eval_S_inits_dict[default_key], seed=seed),
                estimate_omega_kwargs=estimate_omega_kwargs
            )
            agent.estimate_Q(
                Q_estimator=Q_estimator, 
                target_policy=DiscretePolicy(policy_func=policy, num_actions=num_actions),
                estimate_Q_kwargs=estimate_Q_kwargs
            )
            value_est = agent.get_value()
            print(value_est)
            value_interval_est = agent.get_value_interval(alpha=0.05)
            print(value_interval_est)
            agent.validate_Q(grid_size=10, visualize=True)
        else:
            raise NotImplementedError
        
        
        V_est_list.append(value_est)

        # true_value = agent.evaluate_policy(policy=policy, eval_size=eval_policy_mc_size, eval_horizon=eval_horizon, S_inits=eval_S_inits_dict[default_key], seed=seed)
        # true_value = agent.evaluate_policy(policy=policy, eval_size=len(agent._initial_obs), eval_horizon=eval_horizon, S_inits=agent._initial_obs, seed=seed)
        # true_V_list.append(true_value)
    
    print(V_est_list)
    # print(true_V_list)
    
    if mc_size > 1:
        print('[est V_int] average:', np.mean(V_est_list), ', std:', np.std(V_est_list, ddof=1))
        # print('[true V_int] average:', np.mean(true_V_list), ', std:', np.std(true_V_list, ddof=1))
