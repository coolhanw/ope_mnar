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
import pickle
import torch
import torch.nn as nn
torch.manual_seed(0) # for better reproducibility

try:
    from ope_mnar.utils import InitialStateSampler, DiscretePolicy, MinMaxScaler
    from custom_env.linear2d import Linear2dEnv, Linear2dVectorEnv
    from ope_mnar.importance_sampling import MWL, NeuralDualDice
    from ope_mnar.direct_method import FQE, LSTDQ, MQL
    from ope_mnar.doubly_robust import DRL
    from ope_mnar.base import SimulationBase
except:
    import sys
    sys.path.append(os.path.expanduser('~/Projects/ope_mnar/ope_mnar'))
    sys.path.append(os.path.expanduser('~/Projects/ope_mnar'))
    from ope_mnar.utils import InitialStateSampler, DiscretePolicy, MinMaxScaler
    from custom_env.linear2d import Linear2dEnv, Linear2dVectorEnv
    from ope_mnar.importance_sampling import MWL, NeuralDualDice
    from ope_mnar.direct_method import FQE, LSTDQ, MQL
    from ope_mnar.doubly_robust import DRL
    from ope_mnar.base import SimulationBase

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='linear2d')
parser.add_argument('--method', type=str, default='lstdq', choices=["mwl", "mql", "fqe", "lstdq", "dualdice", "drl"])
parser.add_argument('--max_episode_length', type=int, default=25)  # 10, 25
parser.add_argument('--discount', type=float, default=0.8)
parser.add_argument('--num_trajs', type=int, default=500) # 250, 500
parser.add_argument('--burn_in', type=int, default=0)
parser.add_argument('--mc_size', type=int, default=1) # 250
parser.add_argument('--eval_policy_mc_size', type=int, default=50000) # 10000
parser.add_argument('--eval_horizon', type=int, default=250)
parser.add_argument('--dropout_scheme', type=str, default='mar.v0', choices=["0", "mnar.v0", "mar.v0"])
parser.add_argument('--dropout_rate', type=float, default=0.9)
parser.add_argument(
    '--dropout_obs_count_thres',
    type=int,
    default=2,
    help="the number of observations that is not subject to dropout")
parser.add_argument('--ipw',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
parser.add_argument('--estimate_missing_prob',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
parser.add_argument('--weight_curr_step',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
# parser.add_argument('--env_model_type', type=str, default='linear')
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
    gamma = args.discount
    mc_size = args.mc_size
    dropout_scheme = args.dropout_scheme
    dropout_rate = args.dropout_rate
    dropout_obs_count_thres = args.dropout_obs_count_thres
    burn_in = args.burn_in
    vectorize_env = args.vectorize_env
    ipw = args.ipw
    weight_curr_step = args.weight_curr_step # if False, use survival probability
    estimate_missing_prob = args.estimate_missing_prob
    eval_policy_mc_size = args.eval_policy_mc_size  # 10000
    eval_horizon = args.eval_horizon  # 500
    eval_seed = 123
    prob_lbound = 1e-2

    # model configuration
    omega_func_class = 'nn' # marginalized density ratio, {'nn', 'spline', 'expo_linear'}
    Q_func_class = 'nn' # {'nn', 'rf', 'spline'}
    default_scaler = "MinMax" # "NormCdf", "MinMax"
    if ope_method == 'lstdq':
        Q_func_class = 'spline' # override
    if omega_func_class in ['spline', 'expo_linear'] or Q_func_class == 'spline':
        adaptive_dof = False
        # spline related configuration
        basis_scale_factor = 100 # 1, 100
        spline_degree = 3
        if adaptive_dof:
            dof = max(4, int(np.sqrt((n * T)**(3/7)))) # degree of freedom
        else:
            dof = 7
        ridge_factor = 1e-3 # 1e-3
        if default_scaler == "MinMax":
            knots = np.linspace(start=-spline_degree/(dof-spline_degree), stop=1+spline_degree/(dof-spline_degree), num=dof+spline_degree+1) # handle the boundary
        else:
            knots = 'equivdist' # 'equivdist', None
    
    # dropout model configuration
    dropout_model_type = 'linear'
    instrument_var_index = None
    mnar_y_transform = None
    bandwidth_factor = None
    if env_class.lower() == 'linear2d':
        if dropout_scheme == '0':
            missing_mechanism = None
        elif dropout_scheme.startswith('mnar'):
            missing_mechanism = 'mnar'
            instrument_var_index = 1
            if dropout_scheme == 'mnar.v0':
                bandwidth_factor = 7.5
            elif dropout_scheme == 'mnar.v1':
                bandwidth_factor = 2.5
        else:
            missing_mechanism = 'mar'
    psi_true = None  # 1.5, None
    if missing_mechanism and missing_mechanism.lower() == 'mnar':
        initialize_with_psiT = False
    if missing_mechanism is None:
        # no missingness, hence no need of adjustment
        ipw = False
        estimate_missing_prob = False
    
    # filename suffix configuration
    if not ipw:
        weighting_method = 'cc'
    elif ipw and estimate_missing_prob:
        weighting_method = 'ipw_propF'
    elif ipw and not estimate_missing_prob:
        weighting_method = 'ipw_propT'
    folder_suffix = ''
    folder_suffix += f'_missing{dropout_rate}' # add here
    export_dir = os.path.join(
        log_dir, 
        f'{env_class}{folder_suffix}/T_{T}_n_{n}_gamma{gamma}_dropout_{dropout_scheme}_{weighting_method}'
    )
    pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)

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
    print(f'Logged to folder: {export_dir}')

    # environment, initial states and policy configuration
    np.random.seed(seed=eval_seed)
    if env_class.lower() == 'linear2d':
        # specify env and env_dropout
        low = -norm.ppf(0.999)  # -np.inf
        high = norm.ppf(0.999)  # np.inf
        num_actions = 2
        default_key = 'a'

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

    # get true value via MC approximation (use as baseline)
    true_value_path = os.path.join(log_dir, f'{env_class}{folder_suffix}/{env_class}_true_value_T_{eval_horizon}_gamma{gamma}_size{eval_policy_mc_size}')
    if not os.path.exists(true_value_path):
        print(f'compute true value via MC approximation...')
        agent = SimulationBase(env=env_dropout, n=n, horizon=T+burn_in, discount=gamma, eval_env=env)
        true_value_dict = {}
        for k in eval_S_inits_dict.keys():
            _, true_value_list, true_value_int = agent.evaluate_policy(
                policy=policy,
                seed=eval_seed,
                S_inits=eval_S_inits_dict[k],
                eval_size=eval_policy_mc_size,
                eval_horizon=eval_horizon,
            )      
            true_value_dict[k] = {
                'initial_states': eval_S_inits_dict[k], 
                'true_value_list': true_value_list,
                'true_value_int': true_value_int,
            }
            print(f'initial scenario {k}: true integrated value={true_value_int}')
        with open(true_value_path,'wb') as outfile:
            pickle.dump(true_value_dict, outfile)
    else:
        with open(true_value_path,'rb') as outfile:
            true_value_dict = pickle.load(outfile)    

    # main part for value estimation simulation
    verbose = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    value_est_summary = {}
    value_est_summary_path = os.path.join(export_dir, f'{ope_method}_value_est_summary')
    for initial_key in eval_S_inits_dict.keys():
        value_est_list = []
        true_value_int = true_value_dict[initial_key]['true_value_int']
        for itr in range(mc_size):
            seed = itr
            suffix = f'{weighting_method}_itr_{itr}'
            # np.random.seed(seed)
            # if the observational space of the environemnt is bounded, the initial states will only be sampled from uniform distribution
            # if we still want a normal distribution, pass random initial states manually.
            if env_class.lower() == 'linear2d':
                train_S_inits = np.random.normal(loc=0, scale=1, size=(n, env.dim))
                train_S_inits = np.clip(train_S_inits, a_min=low, a_max=high)
            else:
                train_S_inits = None

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
                    discount=gamma,
                    eval_env=env
                )
            elif ope_method == 'mql':
                agent = MQL(
                    env=env_dropout,
                    n=n,
                    horizon=T + burn_in,
                    discount=gamma,
                    eval_env=env,
                    device=device,
                    seed=seed
                )  
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

            # estimate dropout probability
            if estimate_missing_prob:
                model_suffix = suffix
                pathlib.Path(os.path.join(export_dir, 'models')).mkdir(parents=True,
                                                                    exist_ok=True)
                print(f'Fit dropout model ({missing_mechanism})')
                fit_dropout_start = time.time()
                agent.train_dropout_model(
                    model_type=dropout_model_type,
                    missing_mechanism=missing_mechanism,
                    train_ratio=0.8,
                    scale_obs=False,
                    dropout_obs_count_thres=dropout_obs_count_thres,
                    export_dir=os.path.join(export_dir, 'models'),
                    pkl_filename=
                    f"dropout_model_{dropout_model_type}_T{T}_n{n}_gamma{gamma}_{model_suffix}.pkl",
                    seed=seed,
                    include_reward=True,
                    instrument_var_index=instrument_var_index,
                    mnar_y_transform=mnar_y_transform,
                    psi_init=None if missing_mechanism=='mnar' and not initialize_with_psiT else psi_true,
                    bandwidth_factor=bandwidth_factor,
                    verbose=True)
                print(f'Estimate dropout propensities')
                agent.estimate_missing_prob(missing_mechanism=missing_mechanism)
                fit_dropout_end = time.time()
                print(f'Finished! {fit_dropout_end-fit_dropout_start} secs elapsed.')

            # estimate value
            if ope_method == 'mwl': # minimax weight learning
                if omega_func_class == 'nn':
                    agent.estimate_omega(
                        target_policy=DiscretePolicy(policy_func=policy, num_actions=num_actions),
                        initial_state_sampler=InitialStateSampler(initial_states=eval_S_inits_dict[initial_key], seed=seed),
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
                        initial_state_sampler=InitialStateSampler(initial_states=eval_S_inits_dict[initial_key], seed=seed),
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
                        initial_state_sampler=InitialStateSampler(initial_states=eval_S_inits_dict[initial_key], seed=seed),
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
                zeta_pos = True
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
                    initial_state_sampler=InitialStateSampler(initial_states=eval_S_inits_dict[initial_key], seed=seed),
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
                        max_iter=100,
                        tol=0.01,
                        use_RF=False,
                        hidden_sizes=[64,64],
                        lr=0.001,
                        batch_size=128,
                        epoch=50, # 100
                        patience=10,
                        scaler="Standard",
                        print_freq=1,
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
                
                # value_est = np.mean(agent.get_state_values(S_inits=eval_S_inits_dict[initial_key]))
                value_est = agent.get_value(S_inits=eval_S_inits_dict[initial_key])
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
                                scale="MinMax",
                                product_tensor=True,
                                basis_scale_factor=basis_scale_factor,
                                grid_search=False,
                                verbose=True)
                value_est = agent.get_value(S_inits=eval_S_inits_dict[initial_key])
                value_interval_est = agent.get_value_interval(S_inits=eval_S_inits_dict[initial_key], alpha=0.05)
                if mc_size == 1:
                    print('value est:', value_est)
                    print('value interval:', value_interval_est)
                    agent.validate_Q(grid_size=10, visualize=True) # sanity check
            elif ope_method == 'mql':
                agent.estimate_Q(
                        target_policy=DiscretePolicy(policy_func=policy, num_actions=num_actions),
                        max_iter=500,
                        hidden_sizes=[64,64],
                        lr=0.0005,
                        batch_size=64,
                        target_update_frequency=100,
                        patience=50,
                        scaler="Standard",
                        print_freq=50,
                        verbose=verbose
                    )
                value_est = agent.get_value(S_inits=eval_S_inits_dict[initial_key])
                print(value_est)
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
                    initial_state_sampler=InitialStateSampler(initial_states=eval_S_inits_dict[initial_key], seed=seed),
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
            
            value_est_list.append(value_est)
        
        if mc_size > 1:
            print(f'initial state scheme:', initial_key)
            print(f'[true V_int] {true_value_int}')
            print(
                '[est V_int] average:', round(np.mean(value_est_list),3), 
                'std:', round(np.std(value_est_list, ddof=1),3),
                'bias:', round(np.mean(value_est_list) - true_value_int,3),
                'RMSE:', round(np.mean((np.array(value_est_list) - true_value_int) ** 2),3)
            )

        value_est_summary[initial_key] = {
            'initial_states': eval_S_inits_dict[initial_key],
            'value_est_list': value_est_list,
            'true_value_int': true_value_int
        }

    with open(value_est_summary_path,'wb') as outfile:
        pickle.dump(value_est_summary, outfile)
