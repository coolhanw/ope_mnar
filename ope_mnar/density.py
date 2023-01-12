import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import BSpline
from functools import reduce, partial
from itertools import product
import time


class StateActionVisitationModel(nn.Module):

    def __init__(self,
                 state_dim,
                 hidden_sizes,
                 action_dim=1,
                 output_dim=1):
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_dim = state_dim + action_dim

        self.seed = 0

        # build network structure
        layers = []

        # input layer
        input_layer = nn.Linear(in_features=self.input_dim,
                                out_features=hidden_sizes[0])
        nn.init.xavier_normal_(input_layer.weight)
        nn.init.zeros_(input_layer.bias)
        layers.append(input_layer)
        layers.append(nn.LeakyReLU(negative_slope=0.2))  # nn.LeakyReLU(0.01)

        # hidden layers
        for prev_size, size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            linear_layer = nn.Linear(in_features=prev_size, out_features=size)
            nn.init.xavier_normal_(linear_layer.weight)
            nn.init.zeros_(linear_layer.bias)
            layers.append(linear_layer)
            layers.append(nn.ReLU())  # nn.LeakyReLU(0.01)

        # output layer
        output_layer = nn.Linear(in_features=hidden_sizes[-1],
                                 out_features=output_dim)
        nn.init.xavier_normal_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        """
        inputs are concatenations of S, A, S_t, A_t = [r,r,t] (?)
        """
        out = self.layers(inputs)
        out = torch.log(1.0001 + torch.exp(out))
        return out

    # def batch_prediction(self, inputs, batch_size=int(1024 * 8 * 16)):
    #     with torch.no_grad():
    #         if inputs.shape[0] > batch_size:
    #             n_batch = inputs.shape[0] // batch_size + 1
    #             input_batches = np.array_split(inputs, n_batch)
    #             return np.vstack(
    #                 [self.forward(dat).numpy() for dat in input_batches])
    #         else:
    #             return self.forward(inputs).numpy()


class StateActionVisitationExpoLinear(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.input_dim = input_dim
        self.seed = 0

        layers = []
        layer = nn.Linear(in_features=self.input_dim, out_features=1)
        nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)
        layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, inputs):
        out = self.layers(inputs)
        # return torch.exp(out)
        return torch.log(1.0001 + torch.exp(out))


class StateActionVisitationRatio():

    def __init__(
            self,
            replay_buffer,
            initial_state_sampler,
            discount,
            #  action_range,
            hidden_sizes,
            action_dim=1,
            num_actions=None,
            separate_action=0,
            lr=0.001,
            w_clipping_val=0.5,
            w_clipping_norm=1.0,
            device=None):

        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.state_dim = replay_buffer.state_dim
        self.action_dim = action_dim
        self.num_actions = num_actions if num_actions is not None else 1
        self.replay_buffer = replay_buffer
        self.initial_state_sampler = initial_state_sampler
        self.separate_action = separate_action
        self.A_factor_over_S = 1  # 10
        # self.action_range = action_range
        # self.gpu_number = gpu_number
        self.gamma = discount

        if self.separate_action:
            self.action_dim = 0

        self.model = StateActionVisitationModel(
            state_dim=replay_buffer.state_dim,
            hidden_sizes=hidden_sizes,
            action_dim=self.action_dim,
            output_dim=1 if not self.separate_action else self.num_actions).to(self.device)

        # apply between loss.backward() and optimizer.step()
        self.w_clipping_norm = w_clipping_norm  # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.w_clipping_norm)
        self.w_clipping_val = w_clipping_val  # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=self.w_clipping_val)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=100,
                                                         gamma=0.99)

        self.losses = []

    def _compute_medians(self, n=32, rep=20):
        # do it iteratively to save memory

        if self.separate_action:
            median = torch.zeros((self.state_dim, ), dtype=torch.float64)
        else:
            median = torch.zeros((self.state_dim + self.action_dim, ),
                                 dtype=torch.float64)

        for _ in range(rep):
            transitions = self.replay_buffer.sample(n)
            state, action = transitions[0], transitions[1]
            state = torch.Tensor(state)
            state_pairwise_dist = torch.repeat_interleave(
                input=state, repeats=n, dim=0) - torch.tile(input=state,
                                                            dims=(n, 1))
            median_state_dist = torch.mean(torch.abs(state_pairwise_dist),
                                           dim=0)
            if self.separate_action:
                median += median_state_dist + 1e-6
            else:
                action = torch.Tensor(action[:, np.newaxis])
                action_pairwise_dist = torch.repeat_interleave(
                    action, repeats=n, dim=0) - torch.tile(input=action,
                                                           dims=(n, 1))
                median_action_dist = torch.mean(
                    torch.abs(action_pairwise_dist), dim=0)
                median += torch.cat(
                    (median_state_dist, median_action_dist), dim=0) + 1e-6

        median = median / rep

        # return median * (self.state_dim + self.action_dim)
        return median

        # if self.separate_action:
        #     median = torch.ones((self.state_dim,), dtype=torch.float64)
        # else:
        #     median = torch.ones((self.state_dim + self.action_dim,), dtype=torch.float64)
        # return median

    def _cal_dist(self, X1=None, X2=None, median=None):
        """
        Laplacian Kernel: K(x, y) = exp(-gamma ||x-y||_1)
        """
        X1, X2 = torch.Tensor(X1), torch.Tensor(X2)
        if not self.separate_action:
            dist = torch.exp(-torch.sum(torch.abs(X1 - X2) / median, dim=-1))
            dist = dist * (torch.sum(
                torch.abs(X1[:, :self.state_dim] - X2[:, :self.state_dim]),
                dim=-1) != 0).double()
        else:
            dist = torch.exp(-torch.sum(torch.abs(X1[:, :self.state_dim] -
                                                  X2[:, :self.state_dim]) /
                                        median,
                                        dim=-1))
            dist = dist * (X1[:, self.state_dim]
                           == X2[:, self.state_dim]).double()
            dist = dist * (torch.sum(
                torch.abs(X1[:, :self.state_dim] - X2[:, :self.state_dim]),
                dim=-1) != 0).double()
        return dist

    def repeat(self, X, rep):
        return torch.repeat_interleave(input=torch.Tensor(X),
                                       repeats=rep,
                                       dim=0)

    def tile(self, X, rep):
        return torch.tile(input=torch.Tensor(X), dims=(rep, 1))

    def _compute_loss(self, state, action, next_state, state2, action2,
                      next_state2):
        """
        This function is deprecated.

        X = [state, action]
        """
        # action *= self.A_factor_over_S
        # action2 *= self.A_factor_over_S

        state, state2 = torch.Tensor(state), torch.Tensor(state2)
        next_state, next_state2 = torch.Tensor(next_state), torch.Tensor(
            next_state2)
        action, action2 = torch.Tensor(action[:, np.newaxis]), torch.Tensor(
            action2[:, np.newaxis])

        batch_size = self.batch_size
        state_repeat = self.repeat(state, batch_size)  # n^2 x ...
        state_tile = self.tile(state, batch_size)  # n^2 x ...
        next_state_repeat = self.repeat(next_state, batch_size)  # n^2 x ...
        next_state_tile = self.tile(next_state, batch_size)  # n^2 x ...

        state_action = torch.cat([state, action], dim=-1)
        state_action_repeat = self.repeat(state_action,
                                          batch_size)  # n^2 x ...
        state_action_tile = self.tile(state_action, batch_size)  # n^2 x ...
        state2_action2 = torch.cat([state2, action2], dim=-1)
        state2_action2_tile = self.tile(state2_action2,
                                        batch_size)  # n^2 x ...

        pi_state = torch.Tensor(
            self.target_policy.get_action(state)
            [:, np.newaxis])  # * self.A_factor_over_S
        state_pi = torch.cat([state, pi_state], dim=-1)
        state_pi_repeat = self.repeat(state_pi, batch_size)
        state_pi_tile = self.tile(state_pi, batch_size)

        pi_state2 = torch.Tensor(
            self.target_policy.get_action(state2)
            [:, np.newaxis])  # * self.A_factor_over_S
        state2_pi = torch.cat([state2, pi_state2], dim=-1)
        state2_pi_tile = self.tile(state2_pi, batch_size)

        pi_next_state = torch.Tensor(
            self.target_policy.get_action(next_state)
            [:, np.newaxis])  # * self.A_factor_over_S
        next_state_pi = torch.cat([next_state, pi_next_state], dim=-1)
        next_state_pi_repeat = self.repeat(next_state_pi, batch_size)
        next_state_pi_tile = self.tile(next_state_pi, batch_size)

        pi_next_state2 = torch.Tensor(
            self.target_policy.get_action(next_state2)
            [:, np.newaxis])  # * self.A_factor_over_S
        next_state2_pi = torch.cat([next_state2, pi_next_state2], dim=-1)
        next_state2_pi_tile = self.repeat(next_state2_pi, batch_size)

        # omega
        omega = self.model.forward(state_action.to(self.device))  # n
        omega = omega / self.mean_omega  #torch.mean(omega)
        # omega = torch.squeeze(omega)
        omega_tile = torch.squeeze(torch.tile(omega, dims=(batch_size, 1)))

        omega2 = self.model.forward(state2_action2.to(self.device))  # n
        omega2 = omega2 / self.mean_omega  #torch.mean(omega)
        # omega2 = torch.squeeze(omega2)
        omega2_tile = torch.squeeze(torch.tile(omega2, dims=(batch_size, 1)))

        # part 1
        K_1 = torch.mean(
            self._cal_dist(
                X1=next_state_pi_repeat, X2=state2_pi_tile, median=self.median)
            * self.repeat(omega, batch_size))
        K_1 *= (2 * self.gamma * (1 - self.gamma))  #* self.gamma
        K_2 = torch.mean(
            self._cal_dist(
                X1=state_action_repeat, X2=state2_pi_tile, median=self.median)
            * self.repeat(omega, batch_size))
        K_2 *= (2 * (1 - self.gamma))
        part1 = K_1 - K_2

        # part 2
        state, state2 = torch.Tensor(state), torch.Tensor(state2)
        next_state, next_state2 = torch.Tensor(next_state), torch.Tensor(
            next_state2)
        action, action2 = torch.Tensor(action[:, np.newaxis]), torch.Tensor(
            action2[:, np.newaxis])

        batch_size = self.batch_size
        state_repeat = self.repeat(state, batch_size)  # n^2 x ...
        state_tile = self.tile(state, batch_size)  # n^2 x ...
        next_state_repeat = self.repeat(next_state, batch_size)  # n^2 x ...
        next_state_tile = self.tile(next_state, batch_size)  # n^2 x ...

        state_action = torch.cat([state, action], dim=-1)
        state_action_repeat = self.repeat(state_action,
                                          batch_size)  # n^2 x ...
        state_action_tile = self.tile(state_action, batch_size)  # n^2 x ...
        state2_action2 = torch.cat([state2, action2], dim=-1)
        state2_action2_tile = self.tile(state2_action2,
                                        batch_size)  # n^2 x ...

        pi_state = torch.Tensor(
            self.target_policy.get_action(state)
            [:, np.newaxis])  # * self.A_factor_over_S
        state_pi = torch.cat([state, pi_state], dim=-1)
        state_pi_repeat = self.repeat(state_pi, batch_size)
        state_pi_tile = self.tile(state_pi, batch_size)

        pi_state2 = torch.Tensor(
            self.target_policy.get_action(state2)
            [:, np.newaxis])  # * self.A_factor_over_S
        state2_pi = torch.cat([state2, pi_state2], dim=-1)
        state2_pi_tile = self.tile(state2_pi, batch_size)

        pi_next_state = torch.Tensor(
            self.target_policy.get_action(next_state)
            [:, np.newaxis])  # * self.A_factor_over_S
        next_state_pi = torch.cat([next_state, pi_next_state], dim=-1)
        next_state_pi_repeat = self.repeat(next_state_pi, batch_size)
        next_state_pi_tile = self.tile(next_state_pi, batch_size)

        pi_next_state2 = torch.Tensor(
            self.target_policy.get_action(next_state2)
            [:, np.newaxis])  # * self.A_factor_over_S
        next_state2_pi = torch.cat([next_state2, pi_next_state2], dim=-1)
        next_state2_pi_tile = self.repeat(next_state2_pi, batch_size)

        # omega
        omega = self.model.forward(state_action.to(self.device))  # n
        omega = omega / self.mean_omega  #torch.mean(omega)
        # omega = torch.squeeze(omega)
        omega_tile = torch.squeeze(torch.tile(omega, dims=(batch_size, 1)))

        omega2 = self.model.forward(state2_action2.to(self.device))  # n
        omega2 = omega2 / self.mean_omega  #torch.mean(omega)
        # omega2 = torch.squeeze(omega2)
        omega2_tile = torch.squeeze(torch.tile(omega2, dims=(batch_size, 1)))
        omega_repeat_tile = self.repeat(omega, batch_size) * self.tile(
            omega2, batch_size)  # n^2
        part2_1 = torch.mean(
            self._cal_dist(X1=next_state_pi_repeat,
                           X2=next_state2_pi_tile,
                           median=self.median) * omega_repeat_tile)
        part2_1 *= self.gamma**2
        part2_2 = torch.mean(
            self._cal_dist(X1=next_state_pi_repeat,
                           X2=state2_action2_tile,
                           median=self.median) * omega_repeat_tile)
        part2_2 *= (2 * self.gamma)
        part2_3 = torch.mean(
            self._cal_dist(X1=state_action_repeat,
                           X2=state2_action2_tile,
                           median=self.median) * omega_repeat_tile)
        part2 = part2_1 - part2_2 + part2_3

        # part 3
        part3 = torch.mean(
            self._cal_dist(X1=state_pi_repeat,
                           X2=state2_pi_tile,
                           median=self.median))
        part3 *= (1 - self.gamma)**2

        # final loss
        loss = part1 + part2 + part3

        return loss * 10000
        # return loss

    def fit(
        self,
        target_policy,
        batch_size=32,
        max_iter=100,
        ipw=False,
        prob_lbound=1e-3,
        print_freq=20,
        patience=5,
        # rep_loss=3
    ):
        # TODO: scale state and action first

        self.target_policy = target_policy
        self.batch_size = batch_size
        self.median = self._compute_medians()
        print('median', self.median)

        wait_count = 0
        min_loss = 1e10

        self.model.train()

        for i in range(max_iter):

            transitions = self.replay_buffer.sample(
                min(batch_size * 100, self.replay_buffer.N))
            state, action, next_state = transitions[0], transitions[
                1], transitions[3]
            state, action = torch.Tensor(state), torch.Tensor(
                action[:, np.newaxis])
            action = action * self.A_factor_over_S
            state_action = torch.cat([state, action], dim=-1)
            if not self.separate_action:
                omega = self.model.forward(state_action.to(self.device))
            else:
                omega = torch.gather(input=self.model.forward(state.to(self.device)),
                                     dim=1,
                                     index=(action /
                                            self.A_factor_over_S).long())
            self.mean_omega = torch.mean(omega)
            if i % print_freq == 0:
                print('mean(omega):',
                      torch.mean(omega).detach().numpy(), ', min(omega):',
                      torch.min(omega).detach().numpy(), ', max(omega):',
                      torch.max(omega).detach().numpy())

            # loss = 0
            # for j in range(rep_loss):
            #     transitions = self.replay_buffer.sample(batch_size)
            #     state, action, next_state = transitions[0], transitions[1], transitions[3]
            #     transitions_tilde = self.replay_buffer.sample(batch_size)
            #     state2, action2, next_state2 = transitions_tilde[0], transitions_tilde[1], transitions_tilde[3]
            #     loss += self._compute_loss(state, action, next_state, state2, action2, next_state2)
            # loss /= rep_loss

            ## part 1
            transitions = self.replay_buffer.sample(batch_size)
            state, action, next_state, dropout_prob = transitions[
                0], transitions[1], transitions[3], transitions[4]
            state, action, next_state = torch.Tensor(state), torch.Tensor(
                action[:, np.newaxis]), torch.Tensor(next_state)
            action = action * self.A_factor_over_S
            dropout_prob = torch.Tensor(dropout_prob[:, np.newaxis])
            if not ipw:
                dropout_prob = torch.zeros(state.size()[0], 1)
            state_action = torch.cat([state, action], dim=-1)
            state_action_repeat = self.repeat(state_action, batch_size)
            pi_next_state = torch.Tensor(
                self.target_policy.get_action(next_state)
                [:, np.newaxis]) * self.A_factor_over_S
            next_state_pi = torch.cat([next_state, pi_next_state], dim=-1)
            next_state_pi_repeat = self.repeat(next_state_pi, batch_size)

            # state2 = self.replay_buffer.sample_init_states(batch_size)
            state2 = self.initial_state_sampler.sample(batch_size)
            state2 = torch.Tensor(state2)
            pi_state2 = torch.Tensor(
                self.target_policy.get_action(state2)
                [:, np.newaxis]) * self.A_factor_over_S
            state2_pi = torch.cat([state2, pi_state2], dim=-1)
            state2_pi_tile = self.tile(state2_pi, batch_size)

            if not self.separate_action:
                omega = self.model.forward(state_action.to(self.device))  # n
            else:
                omega = torch.gather(input=self.model.forward(state.to(self.device)),
                                     dim=1,
                                     index=(action /
                                            self.A_factor_over_S).long())
            omega = omega / self.mean_omega
            # IPW adjustment for missing data
            omega = omega / torch.clamp(
                input=1 - dropout_prob, min=prob_lbound, max=1)

            K_1 = torch.mean(
                self._cal_dist(X1=next_state_pi_repeat,
                               X2=state2_pi_tile,
                               median=self.median) *
                self.repeat(omega, batch_size))
            K_1 *= (2 * self.gamma * (1 - self.gamma))  #* self.gamma
            K_2 = torch.mean(
                self._cal_dist(X1=state_action_repeat,
                               X2=state2_pi_tile,
                               median=self.median) *
                self.repeat(omega, batch_size)  # omega_repeat_tile
            )
            K_2 *= (2 * (1 - self.gamma))
            part1 = K_1 - K_2

            ## part 2

            # transitions = self.replay_buffer.sample(batch_size)
            # state, action, next_state = transitions[0], transitions[
            #     1], transitions[3]
            # state, action, next_state = torch.Tensor(state), torch.Tensor(action[:, np.newaxis]), torch.Tensor(next_state)
            # state_action = torch.cat([state, action], dim=-1)
            # state_action_repeat = self.repeat(state_action,batch_size)  # n^2 x ...
            # pi_next_state = torch.Tensor(
            #     self.target_policy.get_action(next_state)
            #     [:, np.newaxis])  # * self.A_factor_over_S
            # next_state_pi = torch.cat([next_state, pi_next_state], dim=-1)
            # next_state_pi_repeat = self.repeat(next_state_pi, batch_size)

            transitions_tilde = self.replay_buffer.sample(batch_size)
            state2, action2, next_state2, dropout_prob2 = transitions_tilde[
                0], transitions_tilde[1], transitions_tilde[
                    3], transitions_tilde[4]
            state2, action2, next_state2 = torch.Tensor(state2), torch.Tensor(
                action2[:, np.newaxis]), torch.Tensor(next_state2)
            action2 = action2 * self.A_factor_over_S
            dropout_prob2 = torch.Tensor(dropout_prob2[:, np.newaxis])
            if not ipw:
                dropout_prob2 = torch.zeros(state2.size()[0], 1)
            state2_action2 = torch.cat([state2, action2], dim=-1)
            state2_action2_tile = self.tile(state2_action2,
                                            batch_size)  # n^2 x ...
            pi_state2 = torch.Tensor(
                self.target_policy.get_action(state2)
                [:, np.newaxis]) * self.A_factor_over_S
            state2_pi = torch.cat([state2, pi_state2], dim=-1)
            state2_pi_tile = self.tile(state2_pi, batch_size)

            pi_next_state2 = torch.Tensor(
                self.target_policy.get_action(next_state2)
                [:, np.newaxis]) * self.A_factor_over_S
            next_state2_pi = torch.cat([next_state2, pi_next_state2], dim=-1)
            next_state2_pi_tile = self.repeat(next_state2_pi, batch_size)

            if not self.separate_action:
                omega2 = self.model.forward(state2_action2.to(self.device))  # n
            else:
                omega2 = torch.gather(input=self.model.forward(state2.to(self.device)),
                                      dim=1,
                                      index=(action2 /
                                             self.A_factor_over_S).long())
            omega2 = omega2 / self.mean_omega  #torch.mean(omega)
            # omega2_tile = torch.squeeze(
            #     torch.tile(omega2, dims=(batch_size, 1)))

            # IPW adjustment for missing data
            omega2 = omega2 / torch.clamp(
                input=1 - dropout_prob2, min=prob_lbound, max=1)

            omega_repeat_tile = self.repeat(omega, batch_size) * self.tile(
                omega2, batch_size)  # n^2

            part2_1 = torch.mean(
                self._cal_dist(X1=next_state_pi_repeat,
                               X2=next_state2_pi_tile,
                               median=self.median) * omega_repeat_tile)
            part2_1 *= self.gamma**2
            part2_2 = torch.mean(
                self._cal_dist(X1=next_state_pi_repeat,
                               X2=state2_action2_tile,
                               median=self.median) * omega_repeat_tile)
            part2_2 *= (2 * self.gamma)
            part2_3 = torch.mean(
                self._cal_dist(X1=state_action_repeat,
                               X2=state2_action2_tile,
                               median=self.median) * omega_repeat_tile)
            part2 = part2_1 - part2_2 + part2_3

            ## part 3, does not contrinbute to the final gradient
            # state = self.replay_buffer.sample_init_states(batch_size)
            # state2 = self.replay_buffer.sample_init_states(batch_size)
            state = self.initial_state_sampler.sample(batch_size)
            state2 = self.initial_state_sampler.sample(batch_size)
            state, state2 = torch.Tensor(state), torch.Tensor(state2)

            pi_state = torch.Tensor(
                self.target_policy.get_action(state)
                [:, np.newaxis]) * self.A_factor_over_S
            state_pi = torch.cat([state, pi_state], dim=-1)
            state_pi_repeat = self.repeat(state_pi, batch_size)

            pi_state2 = torch.Tensor(
                self.target_policy.get_action(state2)
                [:, np.newaxis]) * self.A_factor_over_S
            state2_pi = torch.cat([state2, pi_state2], dim=-1)
            state2_pi_tile = self.tile(state2_pi, batch_size)

            part3 = torch.mean(
                self._cal_dist(X1=state_pi_repeat,
                               X2=state2_pi_tile,
                               median=self.median))
            part3 *= (1 - self.gamma)**2

            ## final loss
            loss = part1 + part2 + part3

            self.optimizer.zero_grad()

            loss.backward()
            # # clip gradient
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(),
            #                                max_norm=self.w_clipping_norm)
            # torch.nn.utils.clip_grad_value_(self.model.parameters(),
            #                                 clip_value=self.w_clipping_val)

            self.optimizer.step()
            self.scheduler.step()

            self.losses.append(loss.detach().numpy())

            mean_loss = self.losses[0]
            if i % 5 == 0 and i >= 10:
                if i >= 50:
                    mean_loss = np.mean(self.losses[(i - 50):i])
                else:
                    mean_loss = np.mean(self.losses[(i - 10):i])
                if mean_loss / min_loss - 1 > -0.01:
                    wait_count += 1
                if mean_loss < min_loss:
                    min_loss = mean_loss
                    wait_count = 0
                if mean_loss < -0.01:  # mean_loss < 0 or mean_loss < self.losses[0] / 10
                    break
            if i % print_freq == 0:  # and i >= 10:
                print("omega(s,a) training {}/{} DONE! loss = {:.5f}".format(
                    i, max_iter, mean_loss))
            if patience is not None and wait_count >= patience:
                break

    def batch_prediction(self, inputs, batch_size=int(1024 * 8 * 16)):
        inputs = torch.Tensor(inputs)
        self.model.eval()

        if not self.separate_action:
            inputs[:,
                   self.state_dim] = inputs[:, self.
                                            state_dim] * self.A_factor_over_S
            with torch.no_grad():
                if inputs.shape[0] > batch_size:
                    n_batch = inputs.shape[0] // batch_size + 1
                    input_batches = np.array_split(inputs, n_batch)
                    return np.vstack([
                        self.model.forward(inputs.to(self.device)).cpu().numpy()
                        for inputs in input_batches
                    ])
                else:
                    return self.model.forward(inputs.to(self.device)).cpu().numpy()
        else:
            with torch.no_grad():
                states = inputs[:, :self.state_dim]
                action = inputs[:, [self.state_dim]]  # 0-indexed, 1-d action
                if inputs.shape[0] > batch_size:
                    n_batch = inputs.shape[0] // batch_size + 1
                    input_batches = np.array_split(inputs, n_batch)
                    return np.vstack([
                        torch.gather(input=self.model.forward(
                            inputs[:, :self.state_dim].to(self.device)),
                                     dim=1,
                                     index=action.long()).cpu().numpy()
                        for inputs in input_batches
                    ])
                else:
                    return torch.gather(input=self.model.forward(
                        inputs[:, :self.state_dim].to(self.device)),
                                        dim=1,
                                        index=action.long()).cpu().numpy()


class StateActionVisitationRatioSpline():

    def __init__(self,
                 replay_buffer,
                 initial_state_sampler,
                 discount,
                 scaler,
                 num_actions=None):
        self.replay_buffer = replay_buffer
        self.initial_state_sampler = initial_state_sampler
        self.gamma = discount
        self.scaler = scaler
        self.state_dim = replay_buffer.state_dim
        self.num_actions = num_actions

    def B_spline(self,
                 L=10,
                 d=3,
                 knots=None,
                 product_tensor=True,
                 basis_scale_factor=1.):
        """
        Construct B-Spline basis.

        Args:
            L (int): number of basis function (degree of freedom)
            d (int): B-spline degree
            knots (str or np.ndarray): location of knots
        """
        self.product_tensor = product_tensor
        self.basis_scale_factor = basis_scale_factor

        obs_concat = self.replay_buffer.states
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
        action = policy(S)
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

    def fit(self,
            target_policy,
            ipw=False,
            prob_lbound=1e-3,
            ridge_factor=0.,
            L=10,
            d=3,
            knots=None,
            product_tensor=True,
            basis_scale_factor=1.):
        self.B_spline(L=L,
                      d=d,
                      knots=knots,
                      product_tensor=product_tensor,
                      basis_scale_factor=basis_scale_factor)

        self.target_policy = target_policy
        states = self.replay_buffer.states
        actions = self.replay_buffer.actions
        next_states = self.replay_buffer.next_states
        if ipw:
            dropout_prob = self.replay_buffer.dropout_prob
        else:
            dropout_prob = np.zeros_like(actions)
        inverse_wts = 1 / np.clip(
            a=1 - dropout_prob, a_min=prob_lbound, a_max=1).astype(float)

        if ipw:
            self.total_T_ipw = self.n * (self.max_T - self.burn_in - 1)
        else:
            self.total_T_ipw = len(inverse_wts)

        Xi_mat = self._Xi(S=states, A=actions)
        U_mat = self._U(S=next_states, policy=self.target_policy.policy_func)

        # inverse_wts_mat = np.diag(v=inverse_wts.squeeze()).astype(float)
        # mat1 = reduce(np.matmul,[(Xi_mat - self.gamma * U_mat).T, inverse_wts_mat, Xi_mat])
        mat1 = np.matmul(
            (Xi_mat - self.gamma * U_mat).T,
            inverse_wts[:, np.newaxis] * Xi_mat)  # much faster computation

        self.Sigma_hat = np.diag(
            [ridge_factor] * mat1.shape[0]) + mat1 / self.total_T_ipw
        self.Sigma_hat = self.Sigma_hat.astype(float)
        self.inv_Sigma_hat = np.linalg.pinv(self.Sigma_hat)

        self.vector = np.mean(self._U(
            S=self.initial_state_sampler.initial_states,
            policy=self.target_policy.policy_func),
                              axis=0).T

        self.est_alpha = (1 - self.gamma) * np.matmul(self.inv_Sigma_hat,
                                                      self.vector)

        self._store_para(self.est_alpha)

    def _store_para(self, est_alpha):
        """Store the estimated beta in self.para
        
        Args:
            est_alpha (np.ndarray): vector of estimated alpha
        """
        for i in range(self.num_actions):
            self.para[i] = self.est_alpha[i * self.para_dim:(i + 1) *
                                          self.para_dim].reshape(-1)

    def batch_prediction(self, inputs, batch_size=float('inf')):
        # batch_size = len(inputs)
        states = inputs[:, :self.state_dim]
        actions = inputs[:, self.state_dim]
        xi_mat = self._Xi(S=states, A=actions)
        omega_hat = np.matmul(xi_mat, self.est_alpha.reshape(-1, 1))
        return omega_hat


class StateActionVisitationRatioExpoLinear():

    def __init__(self,
                 replay_buffer,
                 initial_state_sampler,
                 discount,
                 scaler,
                 num_actions=None,
                 device=None):

        self.replay_buffer = replay_buffer
        self.initial_state_sampler = initial_state_sampler
        self.gamma = discount
        self.scaler = scaler
        self.state_dim = replay_buffer.state_dim
        self.num_actions = num_actions

        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def B_spline(self,
                 L=10,
                 d=3,
                 knots=None,
                 product_tensor=True,
                 basis_scale_factor=1.):
        """
        Construct B-Spline basis.

        Args:
            L (int): number of basis function (degree of freedom)
            d (int): B-spline degree
            knots (str or np.ndarray): location of knots
        """
        self.product_tensor = product_tensor
        self.basis_scale_factor = basis_scale_factor

        obs_concat = self.replay_buffer.states
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
        action = policy(S)
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

    def fit(
        self,
        target_policy,
        ipw=False,
        prob_lbound=1e-3,
        L=10,
        d=3,
        knots=None,
        product_tensor=True,
        basis_scale_factor=1.,
        lr=0.001,
        batch_size=256,
        max_iter=100,
        print_freq=20,
        patience=5
    ):

        self.B_spline(L=L,
                      d=d,
                      knots=knots,
                      product_tensor=product_tensor,
                      basis_scale_factor=basis_scale_factor)
        self.target_policy = target_policy

        input_dim = self.para_dim * self.num_actions
        self.omega_model = StateActionVisitationExpoLinear(input_dim=input_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.omega_model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=100,
                                                         gamma=0.99)

        wait_count = 0
        min_loss = 10
        self.losses = []

        self.omega_model.train()

        for i in range(max_iter):
            transitions = self.replay_buffer.sample(batch_size)
            state, action, next_state, dropout_prob = transitions[
                0], transitions[1], transitions[3], transitions[4]
            state2 = self.initial_state_sampler.sample(batch_size)

            inputs = torch.Tensor(self._Xi(S=state, A=action))
            omega = self.omega_model(inputs)
            omega = omega / torch.mean(omega)
            pseudo_resid = torch.Tensor(self.gamma * self._U(S=next_state, policy=self.target_policy.policy_func) - self._Xi(S=state, A=action))

            l1 = torch.mean(omega * pseudo_resid, dim=0)
            l2 = (1 - self.gamma) * torch.mean(torch.Tensor(self._U(S=state2, policy=self.target_policy.policy_func)), dim=0)
            loss = torch.dot(l1 + l2, l1 + l2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            self.losses.append(loss.detach().numpy())

            mean_loss = self.losses[0]
            if i % 5 == 0 and i >= 10:
                if i >= 50:
                    mean_loss = np.mean(self.losses[(i - 50):i])
                else:
                    mean_loss = np.mean(self.losses[(i - 10):i])
                if mean_loss / min_loss - 1 > -0.01:
                    wait_count += 1
                if mean_loss < min_loss:
                    min_loss = mean_loss
                    wait_count = 0
                if mean_loss < -0.01:  # mean_loss < 0 or mean_loss < self.losses[0] / 10
                    break
            if i % print_freq == 0:  # and i >= 10:
                print("omega(s,a) training {}/{} DONE! loss = {:.5f}".format(
                    i, max_iter, mean_loss))
            if patience is not None and wait_count >= patience:
                print('wait_count reaches patience, stop training')
                break

    def batch_prediction(self, inputs, batch_size=float('inf')):
        self.omega_model.eval()
        # batch_size = len(inputs)
        states = inputs[:, :self.state_dim]
        actions = inputs[:, self.state_dim]
        xi_mat = self._Xi(S=states, A=actions)
        omega_hat = self.omega_model(torch.Tensor(xi_mat)).detach().numpy()
        return omega_hat