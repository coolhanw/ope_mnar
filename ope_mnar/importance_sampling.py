"""Note: currently the code is only compatible with continues state space and discrete action space"""

import numpy as np
import scipy
import time
from termcolor import colored
import torch
from torch import nn
import collections
# RL environment
from gym.spaces import Tuple
from gym.vector.utils.spaces import batch_space
# visualization
import seaborn as sns
import matplotlib.pyplot as plt

from base import SimulationBase
from density import StateActionVisitationRatio, StateActionVisitationRatioSpline, StateActionVisitationRatioExpoLinear
from utils import SimpleReplayBuffer, DiscretePolicy, MLPModule
from batch_rl.dqn import QNetwork

__all__ = ['MWL', 'NeuralDualDice']


class MWL(SimulationBase):
    """
    Uehara, Masatoshi, Jiawei Huang, and Nan Jiang. "Minimax weight and q-function learning for off-policy evaluation." 
    International Conference on Machine Learning. PMLR, 2020.
    """

    def __init__(self, env, n, horizon, eval_env=None, discount=0.9, device=None, seed=0):

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
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def estimate_omega(self,
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
              verbose=False,
              **kwargs):

        self.replay_buffer = SimpleReplayBuffer(trajs=self.masked_buffer, seed=self.seed)

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

    def omega(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        inputs = np.hstack([states, actions[:, np.newaxis]])
        omega_values = self.omega_func.batch_prediction(inputs=inputs, batch_size=len(states)).squeeze()
        return omega_values

    def get_value(self):
        # state, action, reward, next_state = [np.array([item[i] for traj in self.trajs for item in traj]) for i in range(4)]
        states = self.replay_buffer.states
        actions = self.replay_buffer.actions
        rewards = self.replay_buffer.rewards # (NT,)
        if self.ipw:
            dropout_prob = self.replay_buffer.dropout_prob # (NT,)
        else:
            dropout_prob = np.zeros_like(actions)
        inverse_wts = 1 / np.clip(a=1 - dropout_prob, a_min=self.prob_lbound, a_max=1).astype(float) # (NT,)
        # next_state = self.replay_buffer.next_states

        # est_omega = self.omega_func.batch_prediction(inputs=np.hstack([states, actions[:, np.newaxis]]), batch_size=len(states)).squeeze()  #  (NT,)
        est_omega = self.omega(states, actions) #  (NT,)
        
        ## sanity check for spline
        # xi_mat = self.omega_func._Xi(S=states, A=actions)
        # mat2 = np.sum(xi_mat * (inverse_wts * rewards).reshape(-1,1), axis=0) / self.omega_func.total_T_ipw
        # V_int_IS = 1 / (1 - self.gamma) * np.matmul(mat2.T, self.omega_func.est_alpha)
        
        # est_omega = np.clip(est_omega, a_min=0, a_max=None) # clip, avoid negative values
        est_omega = est_omega / np.mean(est_omega) # normalize
        
        print('omega: min = ', np.min(est_omega), 'max = ', np.max(est_omega))

        V_int_IS = 1 / (1 - self.gamma) * np.mean(est_omega * rewards * inverse_wts)
        
        if self.verbose:
            print(colored("IS = {:.2f}".format(V_int_IS), 'red'))
        
        return V_int_IS

    # def validate_visitation_ratio(self, grid_size=10, visualize=False):
    #     self.grid = []
    #     self.idx2states = collections.defaultdict(list)
    #     states = self.replay_buffer.states
    #     actions = self.replay_buffer.actions
    #     # omega_values = self.omega_model(torch.Tensor(np.hstack([states, actions[:, np.newaxis]]))).detach().numpy().squeeze()
    #     # omega_values = self.omega_func.batch_prediction(torch.Tensor(np.hstack([states, actions[:, np.newaxis]])), batch_size=len(states)).squeeze()
    #     omega_values = self.omega(states, actions)
    #     discretized_states = np.zeros_like(states)
    #     for i in range(self.state_dim):
    #         disc_bins = np.linspace(start=self.low[i] - 0.1, stop=self.high[i] + 0.1, num=grid_size + 1)
    #         # disc_bins = np.quantile(a=states[:,i], q=np.linspace(0, 1, grid_size + 1))
    #         # disc_bins[0] -= 0.1
    #         # disc_bins[-1] += 0.1
    #         self.grid.append(disc_bins)
    #         discretized_states[:,i] = np.digitize(states[:,i], bins=disc_bins) - 1
    #     discretized_states = list(map(tuple, discretized_states.astype('int')))
    #     for ds, s, a, o in zip(discretized_states, states, actions, omega_values):
    #         # self.idx2states[ds].append(np.append(np.append(s, a), o))
    #         self.idx2states[ds].append(np.concatenate([s, [a], [o]]))

    #     # generate trajectories under the target policy
    #     self.idx2states_target = collections.defaultdict(list)

    #     init_states = self._initial_obs
    #     old_num_envs = self.eval_env.num_envs
    #     eval_size = len(init_states)

    #     self.eval_env.num_envs = eval_size
    #     self.eval_env.observation_space = batch_space(
    #         self.eval_env.single_observation_space,
    #         n=self.eval_env.num_envs)
    #     self.eval_env.action_space = Tuple(
    #         (self.eval_env.single_action_space, ) * self.eval_env.num_envs)

    #     trajectories = self.gen_batch_trajs(
    #         policy=self.target_policy.policy_func,
    #         seed=None,
    #         S_inits=init_states,
    #         evaluation=True)
    #     states_target, actions_target, rewards_target = trajectories[:3]
    #     time_idxs_target = np.tile(np.arange(self.eval_env.T), reps=len(states_target))
    #     states_target = np.vstack(states_target)
    #     actions_target = actions_target.flatten()
    #     print('empirical value:', np.mean(rewards_target) / (1 - self.gamma))
    #     # recover eval_env
    #     self.eval_env.num_envs = old_num_envs
    #     self.eval_env.observation_space = batch_space(
    #         self.eval_env.single_observation_space,
    #         n=self.eval_env.num_envs)
    #     self.eval_env.action_space = Tuple(
    #         (self.eval_env.single_action_space, ) * self.eval_env.num_envs)

    #     discretized_states_target = np.zeros_like(states_target)
    #     for i in range(self.state_dim):
    #         disc_bins = self.grid[i]
    #         discretized_states_target[:,i] = np.digitize(states_target[:,i], bins=disc_bins) - 1
    #     discretized_states_target = list(map(tuple, discretized_states_target.astype('int')))
    #     for ds, s, a, t in zip(discretized_states_target, states_target, actions_target, time_idxs_target):
    #         self.idx2states_target[ds].append(np.concatenate([s, [a], [t]]))

    #     # only for  2D state, binary action
    #     freq_mat = np.zeros(shape=(self.num_actions, grid_size, grid_size))
    #     visit_ratio_mat = np.zeros(shape=(self.num_actions, grid_size, grid_size))
    #     for k, v in self.idx2states.items():
    #         v = np.array(v)
    #         freq_mat[0][k[0]][k[1]] = sum(v[:,self.state_dim] == 0) / len(states)
    #         freq_mat[1][k[0]][k[1]] = sum(v[:,self.state_dim] == 1) / len(states)
    #         visit_ratio_mat[0][k[0]][k[1]] = np.mean(v[v[:,self.state_dim] == 0,self.state_dim+1])
    #         visit_ratio_mat[1][k[0]][k[1]] = np.mean(v[v[:,self.state_dim] == 1,self.state_dim+1])

    #     freq_target_mat = np.zeros(shape=(self.num_actions, grid_size, grid_size))
    #     visit_ratio_ref_mat = np.zeros(shape=(self.num_actions, grid_size, grid_size))
    #     for k, v in self.idx2states_target.items():
    #         v = np.array(v)
    #         freq_target_mat[0][k[0]][k[1]] = (1-self.gamma) * sum(self.gamma ** v[v[:,self.state_dim] == 0, self.state_dim+1]) / eval_size
    #         freq_target_mat[1][k[0]][k[1]] = (1-self.gamma) * sum(self.gamma ** v[v[:,self.state_dim] == 1, self.state_dim+1]) / eval_size
    #         # freq_target_mat[1][k[0]][k[1]] = sum(v[:,self.state_dim] == 0) / len(states_target)
    #         # freq_target_mat[1][k[0]][k[1]] = sum(v[:,self.state_dim] == 1) / len(states_target)
    #         visit_ratio_ref_mat[0][k[0]][k[1]] = freq_target_mat[0][k[0]][k[1]] / max(freq_mat[0][k[0]][k[1]], 0.0001)
    #         visit_ratio_ref_mat[1][k[0]][k[1]] = freq_target_mat[1][k[0]][k[1]] / max(freq_mat[1][k[0]][k[1]], 0.0001)
    
    #     if visualize:

    #         fig, ax = plt.subplots(2, self.num_actions, figsize=(5*self.num_actions,8))
    #         for a in range(self.num_actions):
    #             sns.heatmap(
    #                 freq_mat[a], 
    #                 cmap="YlGnBu",
    #                 linewidth=1,
    #                 ax=ax[0,a]
    #             )
    #             ax[0,a].invert_yaxis()
    #             ax[0,a].set_title(f'discretized state visitation of pi_b (action={a})')
    #         for a in range(self.num_actions):
    #             sns.heatmap(
    #                 freq_target_mat[a], 
    #                 cmap="YlGnBu",
    #                 linewidth=1,
    #                 ax=ax[1,a]
    #             )
    #             ax[1,a].invert_yaxis()
    #             ax[1,a].set_title(f'discretized state visitation of pi (action={a})')
    #         plt.savefig('./output/visitation_heatplot.png')

    #         fig, ax = plt.subplots(2, self.num_actions, figsize=(5*self.num_actions,8))
    #         for a in range(self.num_actions):
    #             sns.heatmap(
    #                 visit_ratio_mat[a], 
    #                 cmap="YlGnBu",
    #                 linewidth=1,
    #                 ax=ax[0,a]
    #             )
    #             ax[0,a].invert_yaxis()
    #             ax[0,a].set_title(f'est visitation ratio (action={a})')
    #         for a in range(self.num_actions):
    #             sns.heatmap(
    #                 visit_ratio_ref_mat[a], 
    #                 cmap="YlGnBu",
    #                 linewidth=1,
    #                 ax=ax[1,a]
    #             )
    #             ax[1,a].invert_yaxis()
    #             ax[1,a].set_title(f'empirical visitation ratio (action={a})')
    #         plt.savefig(f'./output/mwl_{self.func_class}_est_visitation_ratio_heatplot.png')
    #         plt.close()


class NeuralDualDice(SimulationBase):
    """
    Nachum, Ofir, et al. "Dualdice: Behavior-agnostic estimation of discounted stationary distribution corrections." 
    Advances in Neural Information Processing Systems 32 (2019).

    Official implementation: https://github.com/google-research/dice_rl/blob/master/estimators/neural_dual_dice.py
    """

    def __init__(self, env, n, horizon, eval_env=None, discount=0.9, device=None, seed=0):

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

        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def estimate_omega(
        self,
        target_policy,
        initial_state_sampler,
        nu_network_kwargs, 
        nu_learning_rate,
        zeta_network_kwargs,
        zeta_learning_rate,
        zeta_pos: bool = False,
        solve_for_state_action_ratio: bool = True,
        max_iter: int = 10000,
        batch_size: int = 1024,
        f_exponent: float = 2,
        primal_form: bool = False,
        scaler=None,
        print_freq: int = 50,
        verbose: bool = False,
    ):
        """
        Estimate the state-action visitation ratio.

        Parameters
        ----------
        target_policy: DiscretePolicy
            The target policy to evaluate
        initial_state_sampler: InitialStateSampler
            Sampler for initial states and actions.
        nu_network_kwargs: dict
            Parameters of the nu-value network. Default {'hidden_sizes': [64, 64], 'hidden_nonlinearity': nn.ReLU(), 
            'hidden_w_init': nn.init.xavier_uniform_, 'output_w_inits': nn.init.xavier_uniform_}
        nu_learning_rate: float, 0.0001
            Learning rate for nu. 
        zeta_network_kwargs: dict
            Parameters of the zeta-value network. Default {'hidden_sizes': [64, 64], 'hidden_nonlinearity': nn.ReLU(), 
            'hidden_w_init': nn.init.xavier_uniform_, 'output_w_inits': nn.init.xavier_uniform_, 
            'output_nonlinearities': nn.Identity()}
        zeta_learning_rate: float, 0.0001
            Learning rate for zeta.
        zeta_pos: bool
            Whether to enforce positivity constraint.
        solve_for_state_action_ratio: bool, True
            Whether to solve for state-action density ratio. 
        f_exponent: float
            Exponent p to use for f(x) = |x|^p / p.
        primal_form: bool, False
            Whether to use primal form of DualDICE, which optimizes for
            nu independent of zeta. This form is biased in stochastic environments.
            Defaults to False, which uses the saddle-point formulation of DualDICE.
        print_freq: int
            Frequency to print loss values
        verbose: bool
            Whether to output intermediate results
        """
        # TODO: apply scaler before input to network

        self.replay_buffer = SimpleReplayBuffer(trajs=self.masked_buffer, seed=self.seed)

        self.verbose = verbose
        self.target_policy = target_policy
        self.initial_state_sampler = initial_state_sampler
        self._solve_for_state_action_ratio = solve_for_state_action_ratio
        if not solve_for_state_action_ratio:
            raise NotImplementedError
        self._categorical_action = True # currently only handles discrete action space
        self._primal_form = primal_form
        
        self._nu_network = QNetwork(input_dim=self.state_dim, output_dim=self.num_actions, **nu_network_kwargs)
        self._nu_optimizer = torch.optim.Adam(params=self._nu_network.parameters(), lr=nu_learning_rate)
        self._zeta_network = QNetwork(input_dim=self.state_dim, output_dim=self.num_actions, **zeta_network_kwargs)
        self._zeta_optimizer = torch.optim.Adam(params=self._zeta_network.parameters(), lr=zeta_learning_rate)

        if f_exponent <= 1:
            raise ValueError('Exponent for f must be greater than 1.')
        fstar_exponent = f_exponent / (f_exponent - 1)
        self._f_fn = lambda x: torch.abs(x) ** f_exponent / f_exponent
        self._fstar_fn = lambda x: torch.abs(x) ** fstar_exponent / fstar_exponent

        self._train(max_iter=max_iter, batch_size=batch_size, print_freq=print_freq)


    def _get_average_value(self, network, states, policy):
        """
        Parameters
        ----------
        network: nn.Module
        states: torch.FloatTensor
        policy: DiscretePolicy
        """
        action_weights = policy.get_action_prob(states)  # input should be on the original scale
        action_weights = torch.Tensor(action_weights).to(self.device)
        return  torch.sum(network(states) * action_weights, dim=1, keepdim=True)

    def _train_loss(self, states, actions, next_states, initial_states):
        """
        Parameters
        ----------
        states: torch.FloatTensor, dim=(n,state_dim)
        actions: torch.LongTensor, dim=(n,1)
        next_states: torch.FloatTensor, dim=(n,state_dim)
        initial_states: torch.FloatTensor, dim=(n,state_dim)
        """
        nu_values = self._nu_network(states).gather(dim=1, index=actions)
        initial_nu_values = self._get_average_value(self._nu_network, initial_states, self.target_policy)
        next_nu_values = self._get_average_value(self._nu_network, next_states, self.target_policy)
        zeta_values = self._zeta_network(states).gather(dim=1, index=actions)

        bellman_residuals = nu_values - self.gamma * next_nu_values
        zeta_loss = self._fstar_fn(zeta_values) - bellman_residuals.detach() * zeta_values
        if self._primal_form:
            nu_loss = self._f_fn(bellman_residuals) - (1 - self.gamma) * initial_nu_values
        else:
            nu_loss = bellman_residuals * zeta_values.detach() - (1 - self.gamma) * initial_nu_values
        return torch.mean(nu_loss), torch.mean(zeta_loss)

    def _train(self, max_iter: int, batch_size: int, print_freq: int = 50):
        self._nu_network.train()
        self._zeta_network.train()

        running_nu_losses = []
        running_zeta_losses = []

        for i in range(max_iter):
            transitions = self.replay_buffer.sample(batch_size)
            states, actions, rewards, next_states = transitions[:4]
            initial_states = self.initial_state_sampler.sample(batch_size)
            if actions.ndim == 1:
                actions = actions.reshape(-1,1)

            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            initial_states = torch.FloatTensor(initial_states).to(self.device)

            nu_loss, zeta_loss = self._train_loss(states, actions, next_states, initial_states)

            # optimize nu and zeta network
            self._nu_optimizer.zero_grad()
            nu_loss.backward()
            self._nu_optimizer.step()

            self._zeta_optimizer.zero_grad()
            zeta_loss.backward()
            self._zeta_optimizer.step()

            running_nu_losses.append(nu_loss.item())
            running_zeta_losses.append(zeta_loss.item())

            if i % print_freq == 0:
                moving_window = 100
                if i >= moving_window:
                    mean_nu_loss = np.mean(running_nu_losses[(i-moving_window+1):i+1])
                    mean_zeta_loss = np.mean(running_zeta_losses[(i-moving_window+1):i+1])
                else:
                    mean_nu_loss = np.mean(running_nu_losses[:i+1])
                    mean_zeta_loss = np.mean(running_zeta_losses[:i+1])
                print("omega(s,a) training {}/{} DONE! nu_loss = {:.5f}, zeta_loss = {:.5f}".format(
                    i, max_iter, mean_nu_loss, mean_zeta_loss))

    def omega(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        self._zeta_network.eval()
        if actions.ndim == 1:
            actions = actions.reshape(-1,1)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        
        with torch.no_grad():
            omega_values = self._zeta_network(states).gather(dim=1, index=actions)
        
        return omega_values.squeeze().numpy()

    def get_value(self) -> float:
        states = self.replay_buffer.states
        actions = self.replay_buffer.actions
        rewards = self.replay_buffer.rewards

        est_omega = self.omega(states, actions)
        est_omega = est_omega / np.mean(est_omega) # normalize
        print('omega: min = ', np.min(est_omega), 'max = ', np.max(est_omega))

        V_int_IS = 1 / (1 - self.gamma) * np.mean(est_omega * rewards)
        if self.verbose:
            print(colored("IS = {:.2f}".format(V_int_IS), 'red'))
        
        return V_int_IS
