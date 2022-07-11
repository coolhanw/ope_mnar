import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import pickle
import pathlib

from batch_rl.dqn import DQN,DuelingDQN
from batch_rl.bcq import discrete_BCQ
from batch_rl.rem import REM
from batch_rl.utils import ReplayBuffer, ReplayBufferPER


parser = argparse.ArgumentParser()
parser.add_argument('--discount', type=float, default=0.8)
parser.add_argument('--RL_agent', type=str, default='dueling-dqn') # 'dqn', 'bcq', 'rem', 'dueling-dqn'
parser.add_argument('--prioritized_replay', type=lambda x: (str(x).lower() == 'true'), default=False)   
parser.add_argument('--max_iters', type=int, default=int(2e4))  # int(2e5)
parser.add_argument('--minibatch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

if __name__ == '__main__':
    discount = args.discount
    RL_agent = args.RL_agent
    prioritized_replay = args.prioritized_replay
    max_iters = args.max_iters
    minibatch_size = args.minibatch_size
    lr = args.lr
    subsample_size = None # None, 500
    data_dir = os.path.expanduser('~/ope_mnar/data/20210501')
    export_dir = os.path.expanduser('~/ope_mnar/output/sepsis/batch_rl')
    shift_bloc = True
    shift_bloc_str = '_shiftbloc' if shift_bloc else ''
    use_complete_trajs = False
    if use_complete_trajs:
        full_trajs_str = '_full_trajs'
    else:
        full_trajs_str = ''
    if subsample_size is None:
        data_path = os.path.join(data_dir, f'sepsis_processed{full_trajs_str}_state_action3_reward{shift_bloc_str}_split1.csv')
    else:
        data_path = os.path.join(data_dir, f'sepsis_processed{full_trajs_str}_state_action3_reward{shift_bloc_str}_split1_sample{subsample_size}.csv')

    data = pd.read_csv(data_path)
    full_data = pd.read_csv(os.path.join(data_dir, f'sepsis_processed_state_action3_reward{shift_bloc_str}.csv'))
    pathlib.Path(export_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(data_dir, 'state_features.txt')) as f:
        state_features = f.read().split()

    ########################################################################
    ##                    Specify state features
    ########################################################################
    ## all features
    with open(os.path.join(data_dir, 'state_features.txt')) as f:
        features_cols = f.read().split()

    ## selected features
    features_cols = ['Arterial_pH','SpO2','Temp_C','Chloride','Hb','INR','age','PT','HR','Arterial_BE','Ionised_Ca','Calcium','Arterial_lactate'] + ['SOFA'] # 14 features
    static_features = [f for f in features_cols if f in set(['gender', 'age', 'Weight_kg', 're_admission'])]
    dynamic_features = [f for f in features_cols if f not in static_features]
    features_cols = static_features + dynamic_features
    state_dim = len(features_cols)

    ########################################################################
    ##                    Specify actions
    ########################################################################
    action_col = 'action' # already discretized
    num_actions = 3

    ########################################################################
    ##                    Process data then add to bufffer
    ########################################################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_scaled = data.copy(deep=True)

    ## scale features
    scaler_path = os.path.join(data_dir, f'feature{state_dim}_scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path,'rb') as f:
            scaler = pickle.load(f)
    else:
        scaler = MinMaxScaler()
        scaler.fit(full_data[features_cols])
        with open(scaler_path,'wb') as f:
            pickle.dump(scaler, f)

    print(f'data_min_: {scaler.data_min_}')
    print(f'data_max_: {scaler.data_max_}')
    data_scaled[features_cols] = scaler.transform(data[features_cols]) 
    data = data_scaled.copy(deep=True)

    id_col = 'icustayid'
    custom_reward_name = 'raghu'
    ## custom reward function
    custom_reward_name = 'raghu_v1'
    c0 = -0.5
    c1 = -0.25
    c2 = -1

    def custom_reward(rows):
        next_sofa = rows['SOFA'].iloc[1:].values
        curr_sofa = rows['SOFA'].iloc[:-1].values
        next_lactate = rows['Arterial_lactate'].iloc[1:].values
        curr_lactate = rows['Arterial_lactate'].iloc[:-1].values
        reward = ((next_sofa == curr_sofa) & (next_sofa > 0)) * c0 + (next_sofa - curr_sofa) * c1 \
            + np.tanh(next_lactate - curr_lactate) * c2
        reward_full = np.append(reward, 0)
        rows['reward_' + custom_reward_name] = reward_full
        return rows

    data = data.groupby('icustayid').apply(custom_reward)

    if custom_reward_name == 'raghu':   
        reward_col = 'raghu_reward' # without the dominant term at terminal step
    else:
        reward_col = 'reward_' + custom_reward_name

    id_list = data[id_col].unique()
    T = data.groupby(id_col).size().max()

    # add data to replay buffer
    print('Load data into buffer...')
    print(f'Number of trajectories: {len(id_list)}')
    if prioritized_replay:
        buffer = ReplayBufferPER(state_dim=state_dim, buffer_size=data.shape[0], device=device)
    else:
        buffer = ReplayBuffer(state_dim=state_dim, buffer_size=data.shape[0], device=device)
    initial_states = []
    for i, id_ in enumerate(id_list):
        tmp = data.loc[data[id_col] == id_]
        traj_len = len(tmp)
        initial_states.append(tmp[features_cols].iloc[0].values)
        if traj_len < T:
            states = tmp[features_cols].iloc[:-1].values
            next_states = tmp[features_cols].iloc[1:].values
            actions = tmp[action_col].iloc[:-1].values.reshape(-1,1)
            rewards = tmp[reward_col].iloc[:-1].values.reshape(-1,1)
            dones = np.zeros(shape=rewards.shape)
        if traj_len == T:
            states = tmp[features_cols].values
            next_states = np.vstack([tmp[features_cols].iloc[1:].values, np.zeros(shape=(1, state_dim))]) # use 0's to pad the last observation
            actions = tmp[action_col].values.reshape(-1,1)
            rewards = tmp[reward_col].values.reshape(-1,1)
            dones = np.zeros(shape=rewards.shape)
            dones[-1] = 1
        if prioritized_replay:
            buffer.add_episode(states, actions, next_states, rewards, dones, priorities=None)
        else:
            buffer.add_episode(states, actions, next_states, rewards, dones)

    initial_states = np.vstack(initial_states)
    print('Finished!')
        
    ########################################################################
    ##                    Train Agent
    ########################################################################
    print(f'agent: {RL_agent.lower()}')
    if RL_agent.lower() == 'dqn':
        ## train optimal policy
        dqn = DQN(num_actions=num_actions,
            state_dim=state_dim,
            device=device,
            discount=discount,
            optimizer="Adam",
            optimizer_parameters={'lr': lr},
            polyak_target_update=True,
            target_update_frequency=1e3,
            tau=0.005)
        dqn.train(replay_buffer=buffer, max_iters=max_iters, minibatch_size=minibatch_size, verbose=True)
        # evaluate learned policy
        V_int = dqn.evaluate(initial_states=initial_states)
        print(f'estimated value: {V_int}')
        # track Q loss
        fig = dqn.plot_loss(smooth_window=1000)
        if prioritized_replay:
            fig.savefig(os.path.join(export_dir, f'{RL_agent}_per_state{state_dim}_act{num_actions}_reward_{custom_reward_name}_loss.png'))
            dqn.save(filename=os.path.join(export_dir, f'{RL_agent}_per_state{state_dim}_act{num_actions}_reward_{custom_reward_name}'))
        else:
            fig.savefig(os.path.join(export_dir, f'{RL_agent}_state{state_dim}_act{num_actions}_reward_{custom_reward_name}_loss.png'))
            dqn.save(filename=os.path.join(export_dir, f'{RL_agent}_state{state_dim}_act{num_actions}_reward_{custom_reward_name}'))
    elif RL_agent.lower() == 'dueling-dqn':
        # train optimal policy
        dqn = DuelingDQN(num_actions=num_actions,
            state_dim=state_dim,
            device=device,
            discount=discount,
            optimizer="Adam",
            optimizer_parameters={'lr': lr},
            polyak_target_update=True,
            target_update_frequency=1e3,
            tau=0.005)
        dqn.train(replay_buffer=buffer, max_iters=max_iters, minibatch_size=minibatch_size, verbose=True)
        # evaluate learned policy
        V_int = dqn.evaluate(initial_states=initial_states)
        print(f'estimated value: {V_int}')
        # track Q loss
        fig = dqn.plot_loss(smooth_window=1000)
        if prioritized_replay:
            fig.savefig(os.path.join(export_dir, f'{RL_agent}_per_state{state_dim}_act{num_actions}_reward_{custom_reward_name}_loss.png'))
            dqn.save(filename=os.path.join(export_dir, f'{RL_agent}_per_state{state_dim}_act{num_actions}_reward_{custom_reward_name}'))
        else:
            fig.savefig(os.path.join(export_dir, f'{RL_agent}_state{state_dim}_act{num_actions}_reward_{custom_reward_name}_loss.png'))
            dqn.save(filename=os.path.join(export_dir, f'{RL_agent}_state{state_dim}_act{num_actions}_reward_{custom_reward_name}'))
    elif RL_agent.lower() == 'bcq':
        # train optimal policy
        bcq = discrete_BCQ(num_actions=num_actions,
            state_dim=state_dim,
            device=device,
            BCQ_threshold=0.3,
            discount=discount,
            optimizer="Adam",
            optimizer_parameters={'lr': lr},
            polyak_target_update=True,
            target_update_frequency=1e3,
            tau=0.005)
        bcq.train(replay_buffer=buffer, max_iters=max_iters, minibatch_size=minibatch_size, verbose=True)
        # evaluate learned policy
        V_int = bcq.evaluate(initial_states=initial_states)
        print(f'estimated value: {V_int}')
        # track Q loss
        fig = bcq.plot_loss(smooth_window=1000)
        fig.savefig(os.path.join(export_dir, f'{RL_agent}_state{state_dim}_act{num_actions}_reward_{custom_reward_name}_loss.png'))
        bcq.save(filename=os.path.join(export_dir, f'{RL_agent}_state{state_dim}_act{num_actions}_reward_{custom_reward_name}'))
    elif RL_agent.lower() == 'rem':
        # train optimal policy
        rem = REM(num_actions=num_actions,
            state_dim=state_dim,
            num_networks=4, 
            device=device,
            transform_strategy='stochastic',
            num_convex_combinations=1,
            discount=discount,
            optimizer="Adam",
            optimizer_parameters={'lr': lr},
            polyak_target_update=True,
            target_update_frequency=1e3,
            tau=0.005)
        rem.train(replay_buffer=buffer, max_iters=max_iters, minibatch_size=minibatch_size, verbose=True)
        # evaluate learned policy
        V_int = rem.evaluate(initial_states=initial_states)
        print(f'estimated value: {V_int}')
        # track Q loss
        fig = rem.plot_loss(smooth_window=1000)
        fig.savefig(os.path.join(export_dir, f'{RL_agent}_state{state_dim}_act{num_actions}_reward_{custom_reward_name}_loss.png'))
        rem.save(filename=os.path.join(export_dir, f'{RL_agent}_state{state_dim}_act{num_actions}_reward_{custom_reward_name}'))
    else:
        raise NotImplementedError

