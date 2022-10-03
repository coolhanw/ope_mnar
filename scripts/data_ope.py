import sys
import os
import numpy as np
import pandas as pd
import copy
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr
import gc
import pickle
import time
import argparse
import math
import collections
import torch.nn.functional as F
import torch.nn as nn
import torch
import pathlib
import json

try:
    from batch_rl.dqn import QNetwork, DQN, DuelingDQN, DuelingQNetwork
    from batch_rl.policy import DiscreteQFArgmaxPolicy
    from batch_rl.bcq import BehaviorQNetwork, discrete_BCQ
    from batch_rl.rem import MulitNetworkQNetwork, REM, random_stochastic_matrix
    from batch_rl.utils import ReplayBuffer, ReplayBufferPER  
    from ope_mnar.utils import MinMaxScaler, VectorSepsisEnv
    from ope_mnar.agent import OfflineQLearn  
except:
    sys.path.append(os.path.expanduser('~/Projects/ope_mnar'))
    sys.path.append(os.path.expanduser('~/Projects/ope_mnar/ope_mnar'))
    from batch_rl.dqn import QNetwork, DQN, DuelingDQN, DuelingQNetwork
    from batch_rl.policy import DiscreteQFArgmaxPolicy
    from batch_rl.bcq import BehaviorQNetwork, discrete_BCQ
    from batch_rl.rem import MulitNetworkQNetwork, REM, random_stochastic_matrix
    from batch_rl.utils import ReplayBuffer, ReplayBufferPER  
    from ope_mnar.utils import MinMaxScaler, VectorSepsisEnv
    from ope_mnar.agent import OfflineQLearn      

parser = argparse.ArgumentParser()
parser.add_argument('--discount', type=float, default=0.8)
parser.add_argument('--ipw',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
parser.add_argument('--missing_mechanism', type=str, default='mnar')
parser.add_argument('--run_offline_RL',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
parser.add_argument('--RL_agent', type=str, default='dqn') # 'dqn','bcq','rem', 'dueling-dqn'
parser.add_argument('--prioritized_replay',
                    type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--ope',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
parser.add_argument('--eval_behavior_policy',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=False)
parser.add_argument('--exclude_icu_morta',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=True)
parser.add_argument('--use_complete_trajs',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=False)
parser.add_argument('--apply_custom_dropout',
                    type=lambda x: (str(x).lower() == 'true'),
                    default=False)
args = parser.parse_args()

if __name__ == '__main__':
    print(args)
    seed = 123
    np.random.seed(seed)
    discount = args.discount
    ipw = args.ipw
    RL_agent = args.RL_agent
    prioritized_replay = args.prioritized_replay
    estimate_missing_prob = ipw
    missing_mechanism = args.missing_mechanism  # 'mnar', 'mar'
    dropout_model_type = 'linear' if missing_mechanism == 'mar' else None  # 'linear', 'rf'
    # include_reward = False if missing_mechanism == 'mar' else True
    eval_behavior_policy = args.eval_behavior_policy
    run_offline_RL = args.run_offline_RL
    ope = args.ope
    print(f'discount: {discount}')
    print(f'ipw: {ipw}')
    print(f'missing_mechanism: {missing_mechanism}')
    print(f'run_offline_RL: {run_offline_RL}')
    print(f'RL_agent: {RL_agent}')
    print(f'prioritized_replay: {prioritized_replay}')
    print(f'ope: {ope}')
    print(f'eval_behavior_policy: {eval_behavior_policy}')
    if run_offline_RL:
        eval_behavior_policy = False  # evaluate learned policy instead

    if run_offline_RL:
        policy_str = RL_agent.lower()
        if prioritized_replay:
            policy_str += '_per'
    elif eval_behavior_policy:
        policy_str = 'behavior'
    else:
        policy_str = 'random'
    if not ipw:
        estimator_str = 'cc'
    elif missing_mechanism == 'mnar':
        estimator_str = 'ipwMNAR'
    elif missing_mechanism == 'mar':
        estimator_str = 'ipwMAR'

    data_dir = os.path.expanduser("~/Data/mimic-iii")
    export_dir = os.path.expanduser(f'~/Projects/ope_mnar/output/sepsis')
    pathlib.Path(os.path.join(export_dir)).mkdir(parents=True, exist_ok=True)

    shift_bloc = True
    use_complete_trajs = args.use_complete_trajs
    apply_custom_dropout = args.apply_custom_dropout
    exclude_icu_morta = args.exclude_icu_morta
    shift_bloc_str = '_shiftbloc' if shift_bloc else ''
    exclude_icu_morta_str = '_exclude_icu_morta' if exclude_icu_morta else ''
    full_trajs_str = '_full_trajs' if use_complete_trajs else ''
    if use_complete_trajs and apply_custom_dropout:
        synthetic_str = '_synthetic'
    elif use_complete_trajs:
        synthetic_str = '_synthetic_complete'
    else:
        synthetic_str = ''
    
    bandwidth_factor = 5 # 7.5, 2.5
    subsample_size = 500  # None, 250
    if subsample_size is None:
        data_path = os.path.join(
            data_dir, f"sepsis_processed{full_trajs_str}_state_action3_reward{exclude_icu_morta_str}{shift_bloc_str}_split2.csv")
    else:
        data_path = os.path.join(
            data_dir, f"sepsis_processed{full_trajs_str}_state_action3_reward{exclude_icu_morta_str}{shift_bloc_str}_split2_sample{subsample_size}.csv")
    holdout_data_path = os.path.join(
        data_dir, f"sepsis_processed{full_trajs_str}_state_action3_reward{exclude_icu_morta_str}{shift_bloc_str}_split1.csv")
    traj_df = pd.read_csv(data_path)
    traj_df['mechvent'] = traj_df['mechvent'].astype('int')
    holdout_df = pd.read_csv(holdout_data_path)
    holdout_df['mechvent'] = holdout_df['mechvent'].astype('int')
    with open(os.path.join(data_dir, 'state_features.txt')) as f:
        features_cols = f.read().split()

    demog = pd.read_csv(os.path.join(data_dir, "demog.csv"), sep='|')
    demog['icustay_id'] = (demog['icustay_id'] - 2e5).astype('int')
    demog = demog.sort_values(by='icustay_id').reset_index(drop=True)
    demog.rename(columns={'icustay_id': 'icustayid'}, inplace=True)
    demog['outtime'] = demog['outtime'].apply(pd.to_datetime, unit='s')
    traj_df = demog[['icustayid', 'outtime']].merge(traj_df, how='right', on='icustayid')
    traj_df['presumed_onset'] = traj_df['presumed_onset'].apply(pd.to_datetime, unit='s')

    ########################################################################
    ##                    specify rewards
    ########################################################################

    # specify reward function
    # # 1) use the reward defined in Raghu, A. (2017)
    # custom_reward_name = 'raghu'
    # c0 = -0.025
    # c1 = -0.125
    # c2 = -2
    # c3 = 0  # currently do not use +-15 on the terminal state

    # def custom_reward(rows):
    #     next_sofa = rows['SOFA'].iloc[1:].values
    #     curr_sofa = rows['SOFA'].iloc[:-1].values
    #     next_lactate = rows['Arterial_lactate'].iloc[1:].values
    #     curr_lactate = rows['Arterial_lactate'].iloc[:-1].values
    #     reward = ((next_sofa == curr_sofa) & (next_sofa > 0)) * c0 + (next_sofa - curr_sofa) * c1 \
    #         + np.tanh(next_lactate - curr_lactate) * c2
    #     reward_full = np.append(
    #         reward, c3 * (2 * rows['died_in_hosp'].iloc[-1]-1))
    #     rows['reward_'+custom_reward_name] = reward_full
    #     return rows

    # traj_df = traj_df.groupby('icustayid').apply(custom_reward)
    # holdout_df = holdout_df.groupby('icustayid').apply(custom_reward)

    # 2) use the reward defined in Raghu, A. (2017) with different weight
    custom_reward_name = 'raghu_v1'
    c0 = -0.5
    c1 = -0.25
    c2 = -1
    c3 = 1

    def custom_reward(rows):
        next_sofa = rows['SOFA'].iloc[1:].values
        curr_sofa = rows['SOFA'].iloc[:-1].values
        next_lactate = rows['Arterial_lactate'].iloc[1:].values
        curr_lactate = rows['Arterial_lactate'].iloc[:-1].values
        reward = ((next_sofa == curr_sofa) & (next_sofa > 0)) * c0 + (next_sofa - curr_sofa) * c1 \
            + np.tanh(next_lactate - curr_lactate) * c2 + c3
        reward_full = np.append(reward, 0)
        rows['reward_'+custom_reward_name] = reward_full
        return rows

    traj_df = traj_df.groupby('icustayid').apply(custom_reward)
    holdout_df = holdout_df.groupby('icustayid').apply(custom_reward)

    id_col = 'icustayid'
    action_col = 'action'
    reward_col = 'raghu_reward' if custom_reward_name == 'raghu' else 'reward_' + \
        custom_reward_name  # 'raghu_reward', 'SOFA'
    print(f'reward_col: {reward_col}')
    reward_transform = None if reward_col.lower(
    ) != 'sofa' else lambda x: 0.1 * (23 - x)
    print(f'reward_transform is None: {reward_transform is None}')

    n_traj = traj_df[id_col].nunique()
    num_actions = traj_df[action_col].nunique()
    traj_value = traj_df.groupby(id_col).agg({reward_col: lambda x: np.sum(x * discount**np.arange(start=0,stop=len(x)))}).values
    behavior_value = np.mean(traj_value)

    T = traj_df.groupby(id_col).size().max()  # 20
    n = traj_df[id_col].nunique()
    id_list = traj_df[id_col].unique()

    burn_in = 0
    if use_complete_trajs:
        dropout_obs_count_thres = 1
    else:
        dropout_obs_count_thres = traj_df.groupby(id_col).size().min()

    n_traj = traj_df['icustayid'].nunique()
    print(f'maximum horizon: {T}')
    print(f'number of trajectories: {n_traj}')
    print(f'drop out_obs_count_thres: {dropout_obs_count_thres}')

    ########################################################################
    ##                    some data process
    ########################################################################
    subsample_id_list = []
    # apply downsampling to accelerate training
    print(f'number of trajectories: {len(id_list)}')
    if len(id_list) > 2000:
        downsample_size = 500
        downsample_size = min(
            downsample_size, subsample_size) if subsample_size is not None else downsample_size
        # for i in range(math.ceil(len(id_list) / downsample_size)):
        for i in range(max(math.ceil(len(id_list) / downsample_size), 250)): # Monte Carlo iterations
            # after setting seed, subsample ids in each iteration will be the same for CC and IPW estimator
            # we can then calculate pairwise difference
            subsample_id = np.random.choice(
                id_list, size=downsample_size, replace=False) # replace=True
            subsample_id_list.append(tuple(subsample_id))
    else:
        subsample_id_list = [tuple(id_list)]

    ## custom dropout model
    if use_complete_trajs and apply_custom_dropout:
        def custom_dropout_model(rows):
            # # discharge - model 0:
            # FiO2 = rows['FiO2_1'].values
            # HR = rows['HR'].values
            # RR = rows['RR'].values
            # GCS = rows['GCS'].values
            # I1 = 1 * (FiO2 <= 0.6)
            # I2 = 1 * np.logical_and(HR >= 60, HR <= 100)
            # I3 = 1 * np.logical_and(RR >= 10, RR <= 30)
            # I4 = 1 * (GCS >= 14)
            # c1 = -0.8
            # c2 = -0.8
            # c3 = -0.6
            # c4 = -1.5
            # dropout_prob = 1 / (1 + np.exp(4.5 + c1 * I1[:-1] + c2 * I2[:-1] + c3 * I3[:-1] + c4 * I4[1:]))
            
            # discharge - model 1:
            SpO2 = rows['SpO2'].values
            HR = rows['HR'].values
            RR = rows['RR'].values
            SOFA = rows['SOFA'].values
            c0 = 0.1
            c1 = 0.01
            c2 = 0.002
            c3 = 0.004
            c4 = -0.12
            dropout_prob = 1 / (1 + np.exp(c0 + c1 * SpO2[:-1] + c2 * HR[:-1] + c3 * RR[:-1] + c4 * SOFA[1:]))
            rows['custom_dropout_prob'] = np.append(dropout_prob, None)

            # # discharge - model 3:
            # SpO2 = rows['SpO2'].values
            # HR = rows['HR'].values
            # RR = rows['RR'].values
            # Arterial_lactate = rows['Arterial_lactate'].values
            # c0 = 0.1
            # c1 = 0.01
            # c2 = 0.002
            # c3 = 0.004
            # c4 = 0.1
            # dropout_prob = 1 / (1 + np.exp(c0 + c1 * SpO2[:-1] + c2 * HR[:-1] + c3 * RR[:-1] + c4 * Arterial_lactate[1:]))

            # # discharge - model 4:
            # SpO2 = rows['SpO2'].values
            # HR = rows['HR'].values
            # RR = rows['RR'].values
            # GCS = rows['GCS'].values
            # c0 = 0.1
            # c1 = 0.01
            # c2 = 0.002
            # c3 = 0.004
            # c4 = -0.1
            # dropout_prob = 1 / (1 + np.exp(c0 + c1 * SpO2[:-1] + c2 * HR[:-1] + c3 * RR[:-1] + c4 * GCS[1:]))

            # # mortality - model 1:
            # SpO2 = rows['SpO2'].values
            # HR = rows['HR'].values
            # RR = rows['RR'].values
            # SOFA = rows['SOFA'].values
            # c0 = -25
            # c1 = 0.32
            # c2 = 0.01
            # c3 = 0.05
            # c4 = -0.2
            # dropout_prob = 1 / (1 + np.exp(c0 + c1 * SpO2[:-1] + c2 * HR[:-1] + c3 * RR[:-1] + c4 * SOFA[1:]))

            # # mortality - model 2:
            # SpO2 = rows['SpO2'].values
            # HR = rows['HR'].values
            # RR = rows['RR'].values
            # reward = rows[reward_col].values
            # c0 = -30
            # c1 = 0.32
            # c2 = 0.01
            # c3 = 0.05
            # c4 = 1
            # dropout_prob = 1 / (1 + np.exp(c0 + c1 * SpO2[:-1] + c2 * HR[:-1] + c3 * RR[:-1] + c4 * reward[:-1]))
            
            rows['custom_dropout_prob'] = np.append(dropout_prob, None)
            return rows

        traj_df = traj_df.groupby('icustayid').apply(func=custom_dropout_model)
        print('custom_dropout_prob:')
        print(traj_df['custom_dropout_prob'].describe())

    ## custom dropout indicator 
    if not args.use_complete_trajs:
        print('apply custom dropout definition')
        dropout_col_name = 'custom_dropout' # this col name is used as identifier for custom dropout, do not alter it

        # # early death in ICU as dropout
        # def mark_terminal_event(rows):
        #     early_morta = (rows['died_within_48h_of_out_time'].iloc[0] == 1) & (rows['delay_end_of_record_and_discharge_or_death'].iloc[0] < 24)
        #     rows[dropout_col_name] = 0
        #     if early_morta:
        #         rows[dropout_col_name].iloc[-1] = 1
        #     return rows

        # # early discharge: when the trajectory length is shorter than T(max_horizon)
        # def mark_terminal_event(rows):
        #     early_morta = (rows['died_within_48h_of_out_time'].iloc[0] == 1) & (rows['delay_end_of_record_and_discharge_or_death'].iloc[0] < 24)
        #     rows[dropout_col_name] = 0
        #     if len(rows) < T:
        #         rows[dropout_col_name].iloc[-1] = 1
        #     return rows     
        
        # early discharge: only when there is outtime present in the time window
        def mark_terminal_event(rows):
            early_morta = (rows['died_within_48h_of_out_time'].iloc[0] == 1) & (rows['delay_end_of_record_and_discharge_or_death'].iloc[0] < 24)
            rows[dropout_col_name] = 0
            if len(rows) < T and rows['outtime'].iloc[0] <= rows['presumed_onset'].iloc[0] + pd.Timedelta(52, unit='h'):
                rows[dropout_col_name].iloc[-1] = 1
            return rows        
        
        traj_df = traj_df.groupby('icustayid').apply(func=mark_terminal_event)
        print('number of dropout cases:', traj_df[dropout_col_name].sum())

    ########################################################################
    ##                    specify features
    ########################################################################
    selected_features = ['Arterial_pH', 'SpO2', 'Temp_C', 'Chloride', 'Hb', 'INR', 'age',
                         'PT', 'HR', 'Arterial_BE', 'Ionised_Ca', 'Calcium', 'Arterial_lactate'] + ['SOFA']  # 14 features
    static_features = [f for f in selected_features if f in set(
        ['gender', 'age', 'Weight_kg', 're_admission'])]
    dynamic_features = [
        f for f in selected_features if f not in static_features]
    selected_features = static_features + dynamic_features
    print(f'selected_features: {selected_features}')
    state_dim = len(selected_features)
    scaler_path = os.path.join(data_dir, f'feature{state_dim}_scaler.pkl')
    default_scaler = "MinMax"
    if os.path.exists(scaler_path):
        scale = scaler_path
    else:
        scale = default_scaler

    ########################################################################
    ##                 dropout model specification
    ########################################################################

    # discrete candidate instrumental variables based on quartiles
    traj_df['GCS_disc'] = 0
    traj_df.loc[traj_df['GCS'] <= 6, 'GCS_disc'] = 1
    traj_df.loc[(traj_df['GCS'] > 6) & (traj_df['GCS'] <= 9), 'GCS_disc'] = 2
    traj_df.loc[(traj_df['GCS'] > 9) & (traj_df['GCS'] <= 12), 'GCS_disc'] = 3
    traj_df.loc[traj_df['GCS'] > 12, 'GCS_disc'] = 4
    traj_df['Hb_disc'] = 0
    traj_df.loc[traj_df['Hb'] <= 9, 'Hb_disc'] = 1
    traj_df.loc[(traj_df['Hb'] > 9) & (traj_df['Hb'] <= 10), 'Hb_disc'] = 2
    traj_df.loc[(traj_df['Hb'] > 10) & (traj_df['Hb'] <= 11), 'Hb_disc'] = 3
    traj_df.loc[traj_df['Hb'] > 11, 'Hb_disc'] = 4    
    traj_df['PT_disc'] = 0
    traj_df.loc[traj_df['PT'] <= 13, 'PT_disc'] = 1
    traj_df.loc[(traj_df['PT'] > 13) & (traj_df['PT'] <= 14), 'PT_disc'] = 2
    traj_df.loc[(traj_df['PT'] > 14) & (traj_df['PT'] <= 16), 'PT_disc'] = 3
    traj_df.loc[traj_df['PT'] > 16, 'PT_disc'] = 4
    traj_df['Arterial_pH_disc'] = 0
    traj_df.loc[traj_df['Arterial_pH'] <= 7.35, 'Arterial_pH_disc'] = 1
    traj_df.loc[(traj_df['Arterial_pH'] > 7.35) & (traj_df['Arterial_pH'] <= 7.40), 'Arterial_pH_disc'] = 2
    traj_df.loc[(traj_df['Arterial_pH'] > 7.40) & (traj_df['Arterial_pH'] <= 7.44), 'Arterial_pH_disc'] = 3
    traj_df.loc[traj_df['Arterial_pH'] > 7.44, 'Arterial_pH_disc'] = 4
    # indicator of normal range
    traj_df['FiO2_1_normal'] = 1. * (traj_df['FiO2_1'] <= 0.6)
    traj_df['HR_normal'] = 1. * ((traj_df['HR'] >= 60) & (traj_df['HR'] <= 100))
    traj_df['RR_normal'] = 1. * ((traj_df['RR'] >= 10) & (traj_df['RR'] <= 30))
    traj_df['GCS_normal'] = 1. * (traj_df['GCS'] >= 14)
    
    mnar_instrument_var = 'Hb_disc' # 'GCS_disc', 'PT_disc', 'Hb_disc', 'Arterial_pH_disc'
    mnar_instrument_var_index = None
    mnar_nextobs_var = ['SOFA'] # ['SOFA'], ['Arterial_lactate'], ['GCS'], ['GCS_normal']
    mnar_noninstrument_var = ['SpO2','HR','RR'] # ['FiO2_1_normal','HR_normal','RR_normal'], ['SpO2','HR','RR']
    include_reward = False # use reward as outcome, this will override mnar_nextobs_var

    ########################################################################
    ##                    OPE configuration
    ########################################################################
    adaptive_dof = False  # True
    basis_type = 'spline'
    spline_degree = 3
    ridge_factor = 1e-3 # 1e-3
    basis_scale_factor = 1. # 100
    weight_curr_step = True
    prob_lbound = 1e-3
    if adaptive_dof:
        dof = max(4, int(np.sqrt((n * T)**(3/7))))  # degree of freedom
    else:
        dof = 4 # 7
    if scale == 'MinMax':
        knots = np.linspace(start=-spline_degree/(dof-spline_degree), stop=1+spline_degree/(
            dof-spline_degree), num=dof+spline_degree+1)  # take care of the boundary
    else:
        knots = 'equivdist'  # 'equivdist', 'quantile'

    env = VectorSepsisEnv(
        num_envs=n_traj,
        T=T,
        static_state_list=static_features,
        dynamic_state_list=dynamic_features,
        action_levels=num_actions,
        vec_state_trans_model=None,
        vec_reward_model=None,
        vec_dropout_model=None,
        low=-np.inf,  # -np.inf, None
        high=np.inf,  # np.inf, None
        dtype=np.float32
    )

    ########################################################################
    ##                    define target policy to be evaluated
    ########################################################################
    if run_offline_RL:
        print(f'RL agent: {RL_agent}')
        # forward RL
        holdout_id_list = holdout_df[id_col].unique()
        rl_suffix = '' # can be used to distinguish different versions
        rl_export_dir =  os.path.expanduser(f"~/Projects/ope_mnar/output/sepsis/batch_rl{rl_suffix}")
        pathlib.Path(os.path.join(rl_export_dir)).mkdir(
            parents=True, exist_ok=True)
        logger_filename = os.path.join(
                rl_export_dir, f'{RL_agent.lower()}_state{state_dim}_act{num_actions}_reward_{custom_reward_name}_Q')
        print(f'logger_filename: {logger_filename}')
        if not os.path.exists(logger_filename):
            # retrain the optimal policy
            device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu")
            if os.path.exists(scaler_path):
                with open(scaler_path,'rb') as f:
                    scaler = pickle.load(f)
            else:
                full_traj_df = pd.read_csv(os.path.join(data_dir, f"sepsis_processed_state_action3_reward{exclude_icu_morta_str}{shift_bloc_str}.csv"))
                scaler = MinMaxScaler()
                scaler.fit(full_traj_df[selected_features])
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
            holdout_df_scaled = holdout_df.copy(deep=True)
            holdout_df_scaled[selected_features] = scaler.transform(
                holdout_df[selected_features])
            print('Load data into buffer...')
            if prioritized_replay:
                buffer = ReplayBufferPER(
                    state_dim=state_dim, buffer_size=holdout_df_scaled.shape[0], device=device)
            else:
                buffer = ReplayBuffer(
                    state_dim=state_dim, buffer_size=holdout_df_scaled.shape[0], device=device)
            initial_states = []
            for i, id_ in enumerate(holdout_id_list):
                tmp = holdout_df_scaled.loc[holdout_df_scaled[id_col] == id_]
                traj_len = len(tmp)
                initial_states.append(
                    tmp[selected_features].iloc[0].values)
                if traj_len < T:
                    states = tmp[selected_features].iloc[:-1].values
                    next_states = tmp[selected_features].iloc[1:].values
                    actions = tmp[action_col].iloc[:-
                                                    1].values.reshape(-1, 1)
                    rewards = tmp[reward_col].iloc[:-
                                                    1].values.reshape(-1, 1)
                    dones = np.zeros(shape=rewards.shape)
                if traj_len == T:
                    states = tmp[selected_features].values
                    next_states = np.vstack([tmp[selected_features].iloc[1:].values, np.zeros(
                        shape=(1, state_dim))])  # use 0's to pad the last observation
                    actions = tmp[action_col].values.reshape(-1, 1)
                    rewards = tmp[reward_col].values.reshape(-1, 1)
                    dones = np.zeros(shape=rewards.shape)
                    dones[-1] = 1
                if prioritized_replay:
                    buffer.add_episode(
                        states, actions, next_states, rewards, dones, priorities=None)
                else:
                    buffer.add_episode(
                        states, actions, next_states, rewards, dones)
            initial_states = np.vstack(initial_states)
            print('Finished!')
        assert os.path.exists(scaler_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(pd.DataFrame({'Variable': selected_features, 'Min': scaler.data_min_, 'Max': scaler.data_max_}))
        holdout_df_scaled = holdout_df.copy(deep=True)
        holdout_df_scaled[selected_features] = scaler.transform(
            holdout_df[selected_features])
        S = holdout_df_scaled[selected_features]
        loss_fig_name = os.path.join(
            rl_export_dir, f'{RL_agent}_state{state_dim}_act{num_actions}_reward_{custom_reward_name}_loss.png')
        if RL_agent.lower() == 'dqn':
            if not prioritized_replay:
                dqn_filename = os.path.join(
                    rl_export_dir, f'dqn_state{state_dim}_act{num_actions}_reward_{custom_reward_name}')
            else:
                dqn_filename = os.path.join(
                    rl_export_dir, f'dqn_per_state{state_dim}_act{num_actions}_reward_{custom_reward_name}')
            hidden_sizes = [256, 256]
            if not os.path.exists(dqn_filename+'_Q'):
                # train DQN
                device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu")
                print('Train optimal policy...')
                RL_agent = 'dqn'
                dqn = DQN(num_actions=num_actions,
                            state_dim=state_dim,
                            device=device,
                            discount=discount,
                            hidden_sizes=hidden_sizes,
                            optimizer="Adam",
                            optimizer_parameters={'lr': 0.0005},
                            polyak_target_update=True,
                            target_update_frequency=1e3,
                            tau=0.005)
                dqn.train(replay_buffer=buffer, max_iters=int(
                    1e5), minibatch_size=256, verbose=True)
                print('Finished!')
                # evaluate learned policy
                V_int = dqn.evaluate(initial_states=initial_states)
                print(f'estimated value: {V_int}')
                # track Q loss, sometimes the algorithm will diverge
                fig = dqn.plot_loss(smooth_window=1000)

                fig.savefig(loss_fig_name)
                dqn.save(filename=dqn_filename)
            dqn_Q = QNetwork(
                state_dim=state_dim, num_actions=num_actions, hidden_sizes=hidden_sizes)
            dqn_Q.load_state_dict(torch.load(dqn_filename+'_Q'))

            def target_policy(S):
                S_tensor = torch.FloatTensor(scaler.transform(S))
                with torch.no_grad():
                    opt_action = dqn_Q(S_tensor).argmax(
                        dim=1, keepdim=True)
                return opt_action.detach().numpy()
        elif RL_agent.lower() == 'dueling-dqn':
            if not prioritized_replay:
                dqn_filename = os.path.join(
                    rl_export_dir, f'dueling-dqn_state{state_dim}_act{num_actions}_reward_{custom_reward_name}')
            else:
                dqn_filename = os.path.join(
                    rl_export_dir, f'dueling-dqn_per_state{state_dim}_act{num_actions}_reward_{custom_reward_name}')
            hidden_sizes = [256, 256]
            if not os.path.exists(dqn_filename+'_Q'):
                # train DQN
                print('Train optimal policy...')
                RL_agent = 'dqn'
                dqn = DuelingDQN(num_actions=num_actions,
                                    state_dim=state_dim,
                                    device=device,
                                    discount=discount,
                                    hidden_sizes=hidden_sizes,
                                    optimizer="Adam",
                                    optimizer_parameters={'lr': 0.0005},
                                    polyak_target_update=True,
                                    target_update_frequency=1e3,
                                    tau=0.005)
                dqn.train(replay_buffer=buffer, max_iters=int(
                    1e5), minibatch_size=256, verbose=True)
                print('Finished!')
                # evaluate learned policy
                V_int = dqn.evaluate(initial_states=initial_states)
                print(f'estimated value: {V_int}')
                # track Q loss, sometimes the algorithm will diverge
                fig = dqn.plot_loss(smooth_window=1000)
                fig.savefig(loss_fig_name)
                dqn.save(filename=dqn_filename)
            # retrieve Q-function
            dqn_Q = DuelingQNetwork(
                state_dim=state_dim, num_actions=num_actions, hidden_sizes=hidden_sizes)
            dqn_Q.load_state_dict(torch.load(dqn_filename+'_Q'))

            def target_policy(S):
                S_tensor = torch.FloatTensor(scaler.transform(S))
                with torch.no_grad():
                    opt_action = dqn_Q(S_tensor).argmax(
                        dim=1, keepdim=True)
                return opt_action.detach().numpy()
        elif RL_agent.lower() == 'bcq':
            bcq_filename = os.path.join(
                rl_export_dir, f'bcq_state{state_dim}_act{num_actions}_reward_{custom_reward_name}')
            hidden_sizes = [256, 256]
            if not os.path.exists(bcq_filename+'_Q'):
                # train BCQ
                print('Train optimal policy...')
                RL_agent = 'bcq'
                bcq = discrete_BCQ(num_actions=num_actions,
                                    state_dim=state_dim,
                                    device=device,
                                    hidden_sizes=hidden_sizes,
                                    discount=discount,
                                    optimizer="Adam",
                                    optimizer_parameters={'lr': 0.0005},
                                    polyak_target_update=True,
                                    target_update_frequency=1e3,
                                    tau=0.005)
                bcq.train(replay_buffer=buffer, max_iters=int(
                    1e5), minibatch_size=256, verbose=True)
                print('Finished!')
                # evaluate learned policy
                V_int = bcq.evaluate(initial_states=initial_states)
                print(f'estimated value: {V_int}')
                # track Q loss, sometimes the algorithm will diverge
                fig = bcq.plot_loss(smooth_window=1000)
                fig.savefig(loss_fig_name)
                bcq.save(filename=bcq_filename)
            # retrieve Q-function
            bcq_Q = BehaviorQNetwork(
                state_dim=state_dim, num_actions=num_actions, hidden_sizes=hidden_sizes)
            bcq_Q.load_state_dict(torch.load(bcq_filename+'_Q'))

            def target_policy(S):
                S_tensor = torch.FloatTensor(scaler.transform(S))
                with torch.no_grad():
                    opt_action = bcq_Q.Q_layers(
                        S_tensor).argmax(dim=1, keepdim=True)
                return opt_action.detach().numpy()        
        elif RL_agent.lower() == 'rem':
            rem_filename = os.path.join(
                rl_export_dir, f'rem_state{state_dim}_act{num_actions}_reward_{custom_reward_name}')

            num_networks = 4
            hidden_sizes = [256, 256]
            transform_strategy = 'stochastic'
            if not os.path.exists(rem_filename+'_Q'):
                # train REM
                print('Train optimal policy...')
                RL_agent = 'rem'
                rem = REM(num_actions=num_actions,
                            state_dim=state_dim,
                            num_networks=num_networks,
                            device=device,
                            transform_strategy=transform_strategy,
                            num_convex_combinations=1,
                            hidden_sizes=hidden_sizes,
                            discount=discount,
                            optimizer="Adam",
                            optimizer_parameters={'lr': 0.0005},
                            polyak_target_update=True,
                            target_update_frequency=1e3,
                            tau=0.005)
                rem.train(replay_buffer=buffer, max_iters=int(
                    1e5), minibatch_size=256, verbose=True)
                print('Finished!')
                # evaluate learned policy
                V_int = rem.evaluate(initial_states=initial_states)
                print(f'estimated value: {V_int}')
                # track Q loss, sometimes the algorithm will diverge
                fig = rem.plot_loss(smooth_window=1000)
                fig.savefig(loss_fig_name)
                rem.save(filename=rem_filename)
            # retrieve Q-function
            rem_Q = MulitNetworkQNetwork(state_dim=state_dim, num_actions=num_actions, num_networks=num_networks,
                                            hidden_sizes=hidden_sizes, transform_strategy=transform_strategy, transform_matrix=random_stochastic_matrix(dim=num_networks, num_cols=1))  # transform_matrix=random_stochastic_matrix(dim=num_networks,num_cols=1)
            rem_Q.load_state_dict(torch.load(rem_filename+'_Q'))
            print(rem_Q._kwargs['transform_matrix'])

            def target_policy(S):
                S_tensor = torch.FloatTensor(scaler.transform(S))
                with torch.no_grad():
                    opt_action = rem_Q(S_tensor).q_values.argmax(
                        dim=1, keepdim=True)
                return opt_action.detach().numpy()
        else:
            raise NotImplementedError
    elif eval_behavior_policy:
        agent = OfflineQLearn(
            env=env,
            horizon=T,
            scale=scale,  # 'MinMax'
            product_tensor=False,
            discount=discount,
            eval_env=None,
            basis_scale_factor=basis_scale_factor)
        agent.import_holdout_buffer(
            data=holdout_df,
            data_filename=None, # holdout_data_path
            static_state_cols=static_features,
            dynamic_state_cols=dynamic_features,
            action_col=action_col,
            reward_col=reward_col,
            id_col=id_col,
            dropout_col=None,
            reward_transform=reward_transform,
            burn_in=0)
        # build basis
        if basis_type == 'spline':
            agent.B_spline(L=max(3, dof), d=spline_degree,
                               knots=knots)
        # model behavior policy
        behavior_model_type = 'rf'
        behavior_kwargs = {'n_estimators': 250}
        print(f'Fit behavior policy: {behavior_model_type}')
        agent.estimate_behavior_policy(
            model_type=behavior_model_type,
            train_ratio=0.8,
            export_dir=export_dir,
            pkl_filename="behavior_policy_model.pkl",
            seed=seed,
            verbose=True,
            use_holdout_data=True,
            **behavior_kwargs)
        
        # feature_imp = pd.DataFrame(sorted(zip(
        #     agent.fitted_behavior_model.feature_importances_, agent.state_cols)), columns=['Value', 'Feature'])
        # sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(
        #     by="Value", ascending=False).iloc[:25])
        # plt.title(f'Behavior Policy: Most Important Features by RF')
        # plt.savefig(os.path.join(os.path.expanduser(
        #         '~/ope_mnar/model'), f"sepsis_behavior_model_feature_imp.png"))
        # plt.close()

        def target_policy(S):
            X = agent._predictor(S=S)
            return agent.fitted_behavior_model.predict_proba(X=X)
    else:
        print('Evaluate random policy...')
        # random policy
        def target_policy(S):
            return np.tile(np.array([1]*num_actions)/num_actions, reps=(S.shape[0], 1))


    ########################################################################
    ##                    main OPE loop
    ########################################################################
    result_summary = collections.defaultdict(list)
    for subsample_id in subsample_id_list:
        mimic_ope = OfflineQLearn(
            env=env,
            horizon=T,
            scale=scale,  # 'MinMax'
            product_tensor=False,
            discount=discount,
            eval_env=None,
            basis_scale_factor=basis_scale_factor)

        mimic_ope.import_buffer(
            data=traj_df,
            data_filename=None,  # data_path
            static_state_cols=static_features,
            dynamic_state_cols=dynamic_features,
            action_col=action_col,
            reward_col=reward_col,
            id_col=id_col,
            dropout_col='custom_dropout' if not args.use_complete_trajs else None,
            subsample_id=subsample_id,  # None
            reward_transform=reward_transform,
            burn_in=burn_in,
            mnar_nextobs_var=mnar_nextobs_var,
            mnar_noninstrument_var=mnar_noninstrument_var,
            mnar_instrument_var=mnar_instrument_var
        )
        print(f'number of trajectories: {mimic_ope.n}')
        print(f'dropout rate: {mimic_ope.dropout_rate}')
        print(f'missing rate: {mimic_ope.missing_rate}')

        if basis_type == 'spline':
            mimic_ope.B_spline(L=max(3, dof), d=spline_degree,
                               knots=knots)

        _ = gc.collect()

        if ipw:
            print(f'Fit dropout model')
            print(f'missing mechanism: {missing_mechanism}')
            filename_train = f'train_with_T_{T}_n_{n}_L_{dof}_gamma{discount}_{estimator_str}'
            fit_dropout_start = time.time()
            mimic_ope.train_dropout_model(
                model_type=dropout_model_type,
                missing_mechanism=missing_mechanism,
                train_ratio=0.8,
                scale_obs=True,
                dropout_obs_count_thres=dropout_obs_count_thres,
                export_dir=export_dir,
                pkl_filename=f"sepsis_mnar_dropout_model_T{T}_n{n_traj}.pkl",
                seed=seed,
                include_reward=include_reward,
                instrument_var_index=mnar_instrument_var_index,
                gamma_init=None,  # None
                bandwidth_factor=bandwidth_factor,
                verbose=True)
            mimic_ope.estimate_missing_prob(missing_mechanism=missing_mechanism)
            fit_dropout_end = time.time()
            print(
                f'Finished! {fit_dropout_end-fit_dropout_start} secs elapsed.')
            if missing_mechanism == 'mnar':
                gamma_hat = mimic_ope.fitted_dropout_model['model'].gamma_hat
            else:
                gamma_hat = '[null]'
        else:
            gamma_hat = '[null]'

        if ope:
            filename_train = f'train_with_T_{T}_n_{n}_L_{dof}_gamma{discount}_{estimator_str}'
            ope_export_dir = export_dir
            # estimate beta
            print("start updating Q-function for target policy...")
            mimic_ope._beta_hat(policy=target_policy,
                                block=False,
                                ipw=ipw,
                                estimate_missing_prob=estimate_missing_prob,
                                weight_curr_step=weight_curr_step,
                                prob_lbound=prob_lbound,
                                ridge_factor=ridge_factor,
                                grid_search=False,
                                verbose=True,
                                subsample_index=None)
            mimic_ope._store_para(mimic_ope.est_beta)

            print("end updating...")
            env.close()

            # estimate value
            est_V_int = mimic_ope.get_target_value(
                target_policy=target_policy, S_inits=None)
            print(f'estimated value integral: {est_V_int}')

            # make inference
            V_int_summary = mimic_ope.inference_int(policy=target_policy,
                                                    alpha=0.05,
                                                    block=False,
                                                    ipw=ipw,
                                                    estimate_missing_prob=estimate_missing_prob,
                                                    weight_curr_step=True,
                                                    prob_lbound=prob_lbound,
                                                    ridge_factor=ridge_factor,
                                                    MC_size=None,
                                                    S_inits=None,
                                                    verbose=True)
            V_int_lower = V_int_summary['lower_bound']
            V_int_upper = V_int_summary['upper_bound']
            V_int_est = V_int_summary['value']
            V_int_std = V_int_summary['std']
            print(f'Integrated Value: {V_int_est.squeeze()}')
            print(f'Integrated Value std: {V_int_std}')
            print(f'Integrated Value lower bound: {V_int_lower.squeeze()}')
            print(f'Integrated Value upper bound: {V_int_upper.squeeze()}')
            print(f'value of behavior policy: {behavior_value}')
            result_summary['V_int_est'].append(V_int_est)
            result_summary['V_int_std'].append(V_int_std)
            result_summary['V_int_lower'].append(V_int_lower)
            result_summary['V_int_upper'].append(V_int_upper)
            if ipw and missing_mechanism == 'mnar':
                result_summary['gamma_hat'].append(list(mimic_ope.mnar_gamma))

            if ope_export_dir:
                filename_train = os.path.join(ope_export_dir, filename_train)
            with open(filename_train, 'wb') as outfile_train:
                pickle.dump(
                    {
                        'scaler':
                        mimic_ope.scaler,
                        'knot':
                        mimic_ope.knot,
                        'bspline':
                        mimic_ope.bspline,
                        'basis_scale_factor':
                        mimic_ope.basis_scale_factor,
                        'para':
                        mimic_ope.para,
                        'para_dim':
                        mimic_ope.para_dim,
                        'init_obs':
                        mimic_ope._initial_obs,
                        'Sigma_hat':
                        mimic_ope.Sigma_hat,
                        'vector':
                        mimic_ope.vector,
                        'inv_Sigma_hat':
                        mimic_ope.inv_Sigma_hat,
                        'max_inverse_wt':
                        mimic_ope.max_inverse_wt
                    }, outfile_train)
            result_filename = os.path.join(
                export_dir, f'ope_{policy_str}_{estimator_str}{synthetic_str}.json')
            with open(result_filename, "w") as f:
                json.dump(result_summary, f)

        del mimic_ope
        _ = gc.collect()
    print(result_summary)
    print('average: ')
    print({k: np.mean(v) for k, v in result_summary.items()})
