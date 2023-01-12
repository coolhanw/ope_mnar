import numpy as np
import scipy
import time
from termcolor import colored
import torch
import collections
# RL environment
from gym.spaces import Tuple
from gym.vector.utils.spaces import batch_space
# visualization
import seaborn as sns
import matplotlib.pyplot as plt

from base import SimulationBase
from density import StateActionVisitationRatio, StateActionVisitationRatioSpline, StateActionVisitationRatioExpoLinear
from utils import SimpleReplayBuffer


class MWL(SimulationBase):
    """
    Uehara, Masatoshi, Jiawei Huang, and Nan Jiang. "Minimax weight and q-function learning for off-policy evaluation." 
    International Conference on Machine Learning. PMLR, 2020.
    """

    def __init__(self, env, n, horizon, eval_env=None, discount=0.9, device=None):

        super().__init__(
            env=env, 
            n=n, 
            horizon=horizon, 
            discount=discount, 
            eval_env=eval_env
        )

        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.gamma = discount

        # self.alphas = [0.05, 0.1]
        # self.z_stats = [scipy.stats.norm.ppf((1 - alpha / 2)) for alpha in self.alphas]

        # self.omegas_values = []
        # self.omegas = []
        # self.IS_it = []

    def estimate_w(self,
              target_policy,
              initial_state_sampler,
              func_class='nn',
              hidden_sizes=64,
              action_dim=1,
              separate_action=False,
              max_iter=100,
              batch_size=32,
              lr=0.0002,
              ipw=False,
              prob_lbound=1e-3,
              scaler=None,
              print_freq=20,
              patience=5,
            #   rep_loss=3,
              seed=0,
              verbose=0,
              **kwargs):

        self.replay_buffer = SimpleReplayBuffer(trajs=self.masked_buffer, seed=seed)
        self.target_policy = target_policy
        self.func_class = func_class
        self.ipw = ipw
        self.prob_lbound = prob_lbound
        self.separate_action = separate_action
        self.scaler = scaler
        self.verbose = verbose
        
        if self.separate_action:
            action_dim = 0
        
        # curr_time = time.time()
        if self.func_class == 'nn':
            omega_func = StateActionVisitationRatio(
                replay_buffer=self.replay_buffer,
                initial_state_sampler=initial_state_sampler,
                discount=self.gamma,
                # action_range=self.action_range,
                hidden_sizes=hidden_sizes,
                num_actions=self.num_actions,
                action_dim=action_dim, 
                separate_action=self.separate_action,
                lr=lr,
                device=self.device
            )

            omega_func.fit(
                target_policy=target_policy,
                batch_size=batch_size,
                max_iter=max_iter,
                ipw=ipw,
                prob_lbound=prob_lbound,
                print_freq=print_freq,
                patience=patience,
                # rep_loss=rep_loss
                )
            self.omega_func = omega_func
            # self.omega_model = omega_func.model
        elif self.func_class == 'spline':
            omega_func = StateActionVisitationRatioSpline(
                replay_buffer=self.replay_buffer, 
                initial_state_sampler=initial_state_sampler,  
                discount=self.gamma,
                scaler=self.scaler,
                num_actions=self.num_actions
            )

            omega_func.fit(
                target_policy=target_policy,
                ipw=False, 
                prob_lbound=1e-3, 
                **kwargs
                # ridge_factor=0., 
                # L=10, 
                # d=3, 
                # knots=None
            )

            self.omega_func = omega_func
        elif self.func_class == 'expo_linear':
            omega_func = StateActionVisitationRatioExpoLinear(
                replay_buffer=self.replay_buffer, 
                initial_state_sampler=initial_state_sampler,  
                discount=self.gamma,
                scaler=self.scaler,
                num_actions=self.num_actions
            )

            omega_func.fit(
                target_policy=target_policy,
                ipw=False, 
                prob_lbound=1e-3, 
                batch_size=batch_size,
                max_iter=max_iter,
                lr=lr,
                print_freq=print_freq,
                patience=patience,
                **kwargs
            )

            self.omega_func = omega_func
        else:
            raise NotImplementedError

        # state, action, reward, next_state = [
        #     np.array([item[i] for traj in self.trajs for item in traj]) for i in range(4)
        # ]
        # state = self.replay_buffer.states
        # action = self.replay_buffer.actions
        # reward = self.replay_buffer.rewards
        # next_state = self.replay_buffer.next_states

        # state, action = torch.Tensor(state), torch.Tensor(action[:, np.newaxis])
        # omega = omega_func.model.batch_prediction(inputs=torch.cat([state, action], dim=-1))  #  (NT,)
        # omega = np.squeeze(omega)
        
        # if self.verbose:
        #     print(
        #         colored(
        #             "<------------- omega estimation DONE! Time cost = {:.1f} minutes ------------->"
        #             .format((time.time() - curr_time) / 60), 'green'))

    def get_value(self):
        # state, action, reward, next_state = [np.array([item[i] for traj in self.trajs for item in traj]) for i in range(4)]
        state = self.replay_buffer.states
        action = self.replay_buffer.actions
        reward = self.replay_buffer.rewards # (NT,)
        if self.ipw:
            dropout_prob = self.replay_buffer.dropout_prob # (NT,)
        else:
            dropout_prob = np.zeros_like(action)
        inverse_wts = 1 / np.clip(a=1 - dropout_prob, a_min=self.prob_lbound, a_max=1).astype(float) # (NT,)
        # next_state = self.replay_buffer.next_states

        omega = self.omega_func.batch_prediction(inputs=np.hstack([state, action[:, np.newaxis]]), batch_size=len(state)).squeeze()  #  (NT,)
        
        ## sanity check for spline
        # xi_mat = self.omega_func._Xi(S=state, A=action)
        # mat2 = np.sum(xi_mat * (inverse_wts * reward).reshape(-1,1), axis=0) / self.omega_func.total_T_ipw
        # V_int_IS = 1 / (1 - self.gamma) * np.matmul(mat2.T, self.omega_func.est_alpha)
        
        # # clip and normalize
        # omega = np.clip(omega, a_min=0, a_max=None) # avoid negative values
        omega = omega / np.mean(omega)
        
        print('omega: min = ', np.min(omega), 'max = ', np.max(omega))

        V_int_IS = 1 / (1 - self.gamma) * np.mean(omega * reward * inverse_wts)
        
        if self.verbose:
            print(colored("IS = {:.2f}".format(V_int_IS), 'red'))
        
        return V_int_IS

    def validate_visitation_ratio(self, grid_size=10, visualize=False):
        self.grid = []
        self.idx2states = collections.defaultdict(list)
        states = self.replay_buffer.states
        actions = self.replay_buffer.actions
        # omegas = self.omega_model(torch.Tensor(np.hstack([states, actions[:, np.newaxis]]))).detach().numpy().squeeze()
        omegas = self.omega_func.batch_prediction(torch.Tensor(np.hstack([states, actions[:, np.newaxis]])), batch_size=len(states)).squeeze()
        discretized_states = np.zeros_like(states)
        for i in range(self.state_dim):
            disc_bins = np.linspace(start=self.low[i] - 0.1, stop=self.high[i] + 0.1, num=grid_size + 1)
            # disc_bins = np.quantile(a=states[:,i], q=np.linspace(0, 1, grid_size + 1))
            # disc_bins[0] -= 0.1
            # disc_bins[-1] += 0.1
            self.grid.append(disc_bins)
            discretized_states[:,i] = np.digitize(states[:,i], bins=disc_bins) - 1
        discretized_states = list(map(tuple, discretized_states.astype('int')))
        for ds, s, a, o in zip(discretized_states, states, actions, omegas):
            # self.idx2states[ds].append(np.append(np.append(s, a), o))
            self.idx2states[ds].append(np.concatenate([s, [a], [o]]))

        # generate trajectories under the target policy
        self.idx2states_target = collections.defaultdict(list)

        init_states = self._initial_obs
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
            evaluation=True)
        states_target, actions_target, rewards_target = trajectories[:3]
        time_idxs_target = np.tile(np.arange(self.eval_env.T), reps=len(states_target))
        states_target = np.vstack(states_target)
        actions_target = actions_target.flatten()
        print('empirical value:', np.mean(rewards_target) / (1 - self.gamma))
        # recover eval_env
        self.eval_env.num_envs = old_num_envs
        self.eval_env.observation_space = batch_space(
            self.eval_env.single_observation_space,
            n=self.eval_env.num_envs)
        self.eval_env.action_space = Tuple(
            (self.eval_env.single_action_space, ) * self.eval_env.num_envs)

        discretized_states_target = np.zeros_like(states_target)
        for i in range(self.state_dim):
            disc_bins = self.grid[i]
            discretized_states_target[:,i] = np.digitize(states_target[:,i], bins=disc_bins) - 1
        discretized_states_target = list(map(tuple, discretized_states_target.astype('int')))
        for ds, s, a, t in zip(discretized_states_target, states_target, actions_target, time_idxs_target):
            self.idx2states_target[ds].append(np.concatenate([s, [a], [t]]))

        # only for  2D state, binary action
        freq_mat = np.zeros(shape=(self.num_actions, grid_size, grid_size))
        visit_ratio_mat = np.zeros(shape=(self.num_actions, grid_size, grid_size))
        for k, v in self.idx2states.items():
            v = np.array(v)
            freq_mat[0][k[0]][k[1]] = sum(v[:,self.state_dim] == 0) / len(states)
            freq_mat[1][k[0]][k[1]] = sum(v[:,self.state_dim] == 1) / len(states)
            visit_ratio_mat[0][k[0]][k[1]] = np.mean(v[v[:,self.state_dim] == 0,self.state_dim+1])
            visit_ratio_mat[1][k[0]][k[1]] = np.mean(v[v[:,self.state_dim] == 1,self.state_dim+1])

        freq_target_mat = np.zeros(shape=(self.num_actions, grid_size, grid_size))
        visit_ratio_ref_mat = np.zeros(shape=(self.num_actions, grid_size, grid_size))
        for k, v in self.idx2states_target.items():
            v = np.array(v)
            freq_target_mat[0][k[0]][k[1]] = (1-self.gamma) * sum(self.gamma ** v[v[:,self.state_dim] == 0, self.state_dim+1]) / eval_size
            freq_target_mat[1][k[0]][k[1]] = (1-self.gamma) * sum(self.gamma ** v[v[:,self.state_dim] == 1, self.state_dim+1]) / eval_size
            # freq_target_mat[1][k[0]][k[1]] = sum(v[:,self.state_dim] == 0) / len(states_target)
            # freq_target_mat[1][k[0]][k[1]] = sum(v[:,self.state_dim] == 1) / len(states_target)
            visit_ratio_ref_mat[0][k[0]][k[1]] = freq_target_mat[0][k[0]][k[1]] / max(freq_mat[0][k[0]][k[1]], 0.0001)
            visit_ratio_ref_mat[1][k[0]][k[1]] = freq_target_mat[1][k[0]][k[1]] / max(freq_mat[1][k[0]][k[1]], 0.0001)
    
        if visualize:

            fig, ax = plt.subplots(2, self.num_actions, figsize=(5*self.num_actions,8))
            for a in range(self.num_actions):
                sns.heatmap(
                    freq_mat[a], 
                    cmap="YlGnBu",
                    linewidth=1,
                    ax=ax[0,a]
                )
                ax[0,a].invert_yaxis()
                ax[0,a].set_title(f'discretized state visitation of pi_b (action={a})')
            for a in range(self.num_actions):
                sns.heatmap(
                    freq_target_mat[a], 
                    cmap="YlGnBu",
                    linewidth=1,
                    ax=ax[1,a]
                )
                ax[1,a].invert_yaxis()
                ax[1,a].set_title(f'discretized state visitation of pi (action={a})')
            plt.savefig('visitation_heatplot.png')

            fig, ax = plt.subplots(2, self.num_actions, figsize=(5*self.num_actions,8))
            for a in range(self.num_actions):
                sns.heatmap(
                    visit_ratio_mat[a], 
                    cmap="YlGnBu",
                    linewidth=1,
                    ax=ax[0,a]
                )
                ax[0,a].invert_yaxis()
                ax[0,a].set_title(f'est visitation ratio (action={a})')
            for a in range(self.num_actions):
                sns.heatmap(
                    visit_ratio_ref_mat[a], 
                    cmap="YlGnBu",
                    linewidth=1,
                    ax=ax[1,a]
                )
                ax[1,a].invert_yaxis()
                ax[1,a].set_title(f'empirical visitation ratio (action={a})')
            plt.savefig('est_visitation_ratio_heatplot.png')
            plt.close()
        

        

        