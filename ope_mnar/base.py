import numpy as np
import copy
# RL environment
from gym.spaces import Tuple
from gym.vector.utils.spaces import batch_space

class SimulationBase(object):

    def __init__(self, env=None, n=500, horizon=None, discount=0.8, eval_env=None):
        """
        Args:
            env (gym.Env): dynamic environment
            n (int): the number of subjects (trajectories)
            horizon (int): the maximum length of trajectories
            discount (float): discount factor
            eval_env (gym.Env): dynamic environment to evaluate the policy, if not specified, use env
        """
        assert env is not None or eval_env is not None, "please provide env or eval_env"
        self.env = env
        self.vectorized_env = env.vectorized if env is not None else True
        if eval_env is None and env is not None:
            self.eval_env = copy.deepcopy(env)
        else:
            self.eval_env = eval_env
        self.n = n
        self.max_T = self.env.T if horizon is None else horizon # maximum horizon
        self.gamma = discount
        self.obs_policy = lambda S: self.env.action_space.sample() if self.env is not None else None  # uniform sample

        if self.env is not None:
            if self.vectorized_env:
                self.env.num_envs = n
                self.env.observation_space = batch_space(
                    self.env.single_observation_space, n=self.env.num_envs)
                self.env.action_space = Tuple(
                    (self.env.single_action_space, ) * self.env.num_envs)
                self.num_actions = self.env.single_action_space.n # the number of candidate discrete actions
                self.state_dim = self.env.single_observation_space.shape[0]
                # store the last observation which is particularly designed for append block to make 
                # sure that the append block's first state can match the last state in current buffer
                self.last_obs = self.env.observation_space.sample()
            else:
                self.num_actions = self.env.action_space.n # the number of candidate discrete actions
                self.state_dim = self.env.observation_space.shape[0]
                # store the last observation which is particularly designed for append block to make 
                # sure that the append block's first state can match the last state in current buffer
                self.last_obs = np.vstack([
                    self.env.observation_space.sample() for _ in range(self.n)
                ])

        self.low = np.array([np.inf] * self.state_dim) # initial values, will be updated when generating trajectories
        self.high = np.array([-np.inf] * self.state_dim) # initial values, will be updated when generating trajectories
        self.masked_buffer = {} # masked data buffer, only observed data are included
        self.full_buffer = {}
        self.misc_buffer = {}  # hold any other information
        self.concat_trajs = None

    def sample_initial_states(self, size, from_data=False, seed=None):
        assert self.env is not None
        np.random.seed(seed)
        if not from_data:
            if self.vectorized_env:
                old_size = self.env.num_envs
                # reset size
                self.env.num_envs = size
                self.env.observation_space = batch_space(
                    self.env.single_observation_space, n=self.env.num_envs)
                self.env.action_space = Tuple(
                    (self.env.single_action_space, ) * self.env.num_envs)
                S_inits = self.env.observation_space.sample()
                # recover size
                self.env.num_envs = old_size
                self.env.observation_space = batch_space(
                    self.env.single_observation_space, n=self.env.num_envs)
                self.env.action_space = Tuple(
                    (self.env.single_action_space, ) * self.env.num_envs)
            else:
                S_inits = np.vstack([self.env.reset() for _ in range(size)])
            return S_inits
        else:
            assert hasattr(self, '_initial_obs'), "please generate data or import trajectories first"
            selected_id = np.random.choice(a=self.n, size=size)
            return self._initial_obs[selected_id]

    def sample_states(self, size, seed=None):
        assert hasattr(self, '_obs'), "please generate data or import trajectories first"
        np.random.seed(seed)
        selected_id = np.random.choice(a=len(self._obs), size=size)
        return self._obs[selected_id]

    def gen_single_traj(self,
                 policy=None,
                 S_init=None,
                 S_init_kwargs={},
                 A_init=None,
                 burn_in=0,
                 evaluation=False,
                 seed=None):
        """
        Generate a single trajectory. 

        Args:
            policy (callable): the policy to generate the trajectory
            S_init (np.ndarray): initial states
            S_init_kwargs (dict): other kwargs input to env.reset()
            A_init (np.ndarray): initial action, dimension should equal to the number of candidate actions
            burn_in (int): length of burn-in period
            evaluation (bool): if True, use eval_env to generate trajectory
            seed (int): random seed for env

        Returns:
            observations_traj (np.ndarray)
            action_traj (np.ndarray)
            reward_traj (np.ndarray)
            T (int): length of the trajectory
            survival_prob_traj (np.ndarray)
            dropout_next_traj (np.ndarray)
            dropout_prob_traj (np.ndarray)

        TODO: organize the output into a dictionary
        """
        if policy is None:
            policy = self.obs_policy if self.obs_policy is not None else lambda S: self.env.action_space.sample(
            )
        if evaluation:
            env = self.eval_env
        else:
            env = self.env
        # initialize the state
        if seed is not None:
            env.seed(seed)
        if S_init is not None or not S_init_kwargs:
            S = env.reset(S_init, **S_init_kwargs)
        else:
            S = env.reset()
        S_traj = [S]
        A_traj = []
        reward_traj = []
        survival_prob_traj = []
        dropout_next_traj, dropout_prob_traj = [], []
        step = 1
        while step <= env.T:
            self.low = np.minimum(self.low, np.array(S))
            self.high = np.maximum(self.high, np.array(S))
            if len(S_traj) == 1 and A_init is not None:
                A = A_init
            else:
                A = policy(S)
            if hasattr(A, '__iter__'):
                A = np.array(A).squeeze()
                assert len(
                    A
                ) == self.num_actions, f"len(A):{len(A)}, num_actions: {self.num_actions}, output of policy should match the number of actions"
                A = np.random.choice(range(self.num_actions), p=A).item()
            S_next, reward, done, env_infos = env.step(A)
            S_traj.append(S_next)
            A_traj.append(A)
            reward_traj.append(reward)
            survival_prob = env_infos.get('next_survival_prob', 1)
            survival_prob_traj.append(survival_prob)
            dropout_next = env_infos.get('dropout', None)
            dropout_next_traj.append(dropout_next)
            dropout_prob = env_infos.get('dropout_prob', None)
            dropout_prob_traj.append(dropout_prob)
            S = S_next
            if dropout_next:
                break
            step += 1
        T = len(reward_traj)
        # convert to numpy.ndarray
        S_traj = np.array(S_traj)[:T]
        A_traj = np.array(A_traj)
        reward_traj = np.array(reward_traj)
        survival_prob_traj = np.array(survival_prob_traj).reshape(-1)
        dropout_next_traj = np.array(dropout_next_traj).reshape(-1)
        dropout_prob_traj = np.array(dropout_prob_traj).reshape(-1)
        if burn_in is None:
            return [
                S_traj, 
                A_traj, 
                reward_traj, 
                T, 
                survival_prob_traj,
                dropout_next_traj, 
                dropout_prob_traj
            ]
        else:
            if T > burn_in:
                return [
                    S_traj[burn_in:], 
                    A_traj[burn_in:], 
                    reward_traj[burn_in:],
                    T - burn_in, 
                    survival_prob_traj[burn_in:],
                    dropout_next_traj[burn_in:], 
                    dropout_prob_traj[burn_in:]
                ]
            else:
                return []

    def gen_batch_trajs(self,
                        policy=None,
                        S_inits=None,
                        A_inits=None,
                        burn_in=0,
                        evaluation=False,
                        seed=None):
        """
        Generate a batch of trajectories. 

        Args: 
            policy (callable): the policy to generate trajectories
            S_inits (np.ndarray): initial states
            A_inits (np.ndarray): initial actions
            burn_in (int): the lenght of burn-in period
            evaluation (bool): if True, use eval_env
            seed (int): seed (int): random seed for env

        Returns:
            observations_traj (np.ndarray)
            action_traj (np.ndarray)
            reward_traj (np.ndarray)
            T (int)
            survival_prob_traj (np.ndarray)
            dropout_next_traj (np.ndarray)
            dropout_prob_traj (np.ndarray)

        TODO: organize the output into a dictionary
        """
        if A_inits is not None and len(A_inits.shape) == 1:
            A_inits = A_inits.reshape(-1, 1)
        if policy is None:
            policy = self.obs_policy if self.obs_policy is not None else lambda S: self.env.action_space.sample()
        if evaluation:
            env = self.eval_env
        else:
            env = self.env
        if seed is not None:
            env.seed(seed)
        # initialize the states
        if S_inits is not None:
            S = env.reset(S_inits)
        else:
            S = env.reset()
        survival_prob_traj = []
        dropout_next_traj, dropout_prob_traj = [], []
        step = 1
        while step <= env.T:
            self.low = np.minimum(self.low, np.min(S, axis=0))
            self.high = np.maximum(self.high, np.max(S, axis=0))
            if step == 1 and A_inits is not None:
                A = A_inits  # (num_envs,1)
            else:
                A = policy(S)  # (num_envs,num_actions)
                if isinstance(A, tuple):
                    A = np.expand_dims(np.array(A), axis=1)
            if A.shape[1] > 1:
                assert A.shape[
                    1] == self.num_actions, "output of policy should match the number of actions"
                # sample an action based on the probability
                A = (A.cumsum(axis=1) > np.random.rand(
                    A.shape[0])[:, None]).argmax(axis=1)
            S, _, _, env_infos = env.step(actions=A)
            survival_prob = env_infos.get('next_survival_prob',
                                          np.ones(shape=(len(S), 1)))
            survival_prob_traj.append(survival_prob)
            dropout_next = env_infos.get(
                'dropout', np.zeros(shape=(len(S), 1), dtype=np.int8))
            dropout_next_traj.append(dropout_next)
            dropout_prob = env_infos.get('dropout_prob',
                                         np.zeros(shape=(len(S), 1)))
            dropout_prob_traj.append(dropout_prob)
            step += 1
        S_traj = env.states_history  # (num_envs, T, dim)
        A_traj = env.actions_history  # (num_envs, T)
        reward_traj = env.rewards_history  # (num_envs, T)
        survival_prob_traj = np.concatenate(survival_prob_traj,
                                            axis=1)  # (num_envs, T)
        dropout_next_traj = np.concatenate(dropout_next_traj,
                                           axis=1)  # (num_envs, T)
        dropout_prob_traj = np.concatenate(dropout_prob_traj,
                                           axis=1)  # (num_envs, T)

        S_traj_mask = env.states_history_mask  # (num_envs, T)
        # T denotes the actual length of the observed trajectories
        T = np.argmin(S_traj_mask, axis=1)
        T[T == 0] = self.max_T
        # output state, action, reward trajectory and T
        if burn_in is None:
            return [
                S_traj, A_traj, reward_traj, T, survival_prob_traj,
                dropout_next_traj, dropout_prob_traj, S_traj_mask
            ]
        else:
            if any(T > burn_in):
                observed_index = T > burn_in
                return [
                    S_traj[observed_index, burn_in:, :], 
                    A_traj[observed_index, burn_in:],
                    reward_traj[observed_index,burn_in:], 
                    T[observed_index] - burn_in,
                    survival_prob_traj[observed_index, burn_in:],
                    dropout_next_traj[observed_index, burn_in:],
                    dropout_prob_traj[observed_index, burn_in:], 
                    S_traj_mask[observed_index, burn_in:]
                ]
            else:
                return []

    def gen_masked_buffer(self,
                          policy=None,
                          n=None,
                          total_N=None,
                          S_inits=None,
                          S_inits_kwargs={},
                          burn_in=0,
                          seed=None):
        """
        Generate masked (observed) buffer data.

        Args:
            policy (callable): the policy to generate trajectories
            n (int): number of trajectories
            total_N (int): total number of observed tuples
            S_init (np.ndarray): initial states
            S_init_kwargs (dict): additional kwargs passed to env.reset()
            burn_in (int): length of burn-in period
            seed (int): random seed for env
        """
        self.burn_in = burn_in
        if n is None:
            n = self.n
        if S_inits is not None:
            assert len(
                S_inits) == n, "the number of initial states should match n"
        if not self.vectorized_env:
            count = 0
            incomplete_cnt = 0
            if total_N is None:
                for i in range(n):
                    trajs = self.gen_single_traj(
                        policy=policy,
                        burn_in=burn_in,
                        S_init=S_inits[i] if S_inits is not None else None,
                        S_init_kwargs={
                            k: v[i]
                            for k, v in S_inits_kwargs.items()
                        } if S_inits_kwargs else {},
                        A_init=None,
                        evaluation=False)
                    if not trajs:
                        continue
                    self.masked_buffer[(i)] = trajs
                    if self.masked_buffer[(i)][3] < self.max_T:
                        incomplete_cnt += 1
                    count += self.masked_buffer[(i)][3]
            else:
                i = 0
                incomplete_cnt = 0
                while count < total_N:
                    trajs = self.gen_single_traj(
                        policy=policy,
                        burn_in=burn_in,
                        S_init=S_inits[i] if S_inits is not None else None,
                        S_init_kwargs={
                            k: v[i]
                            for k, v in S_inits_kwargs.items()
                        } if S_inits_kwargs else {},
                        A_init=None,
                        evaluation=False)
                    if not trajs:
                        continue
                    self.masked_buffer[(i)] = trajs
                    if self.masked_buffer[(i)][3] < self.max_T:
                        incomplete_cnt += 1
                    count += self.masked_buffer[(i)][3]
                    i += 1
                self.n = i
            self.total_N = count
        else:
            count = 0
            incomplete_cnt = 0
            if total_N is None:
                self.concat_trajs = self.gen_batch_trajs(policy=policy,
                                                         burn_in=burn_in,
                                                         S_inits=S_inits,
                                                         A_inits=None,
                                                         seed=seed,
                                                         evaluation=False)
                S_traj, A_traj, reward_traj, T, survival_prob_traj, dropout_next_traj, dropout_prob_traj, S_traj_mask = self.concat_trajs
                for i in range(len(S_traj)):
                    observed_state_index = np.where(S_traj_mask[i] == 1)[0]
                    observed_index = observed_state_index[
                        observed_state_index < self.max_T - burn_in].tolist()
                    observed_state_index = observed_state_index.tolist()
                    self.masked_buffer[(i)] = [
                        S_traj[i][observed_state_index],
                        # S_traj[i][observed_index],
                        A_traj[i][observed_index],
                        reward_traj[i][observed_index],
                        T[i],
                        survival_prob_traj[i][observed_index],
                        dropout_next_traj[i][observed_index],
                        dropout_prob_traj[i][observed_index]
                    ]
                    self.full_buffer[(i)] = [
                        S_traj[i], A_traj[i], reward_traj[i], T[i],
                        survival_prob_traj[i], dropout_next_traj[i],
                        dropout_prob_traj[i]
                    ]
                    if T[i] < self.max_T:
                        incomplete_cnt += 1
                    count += T[i]
            else:
                i = 0
                incomplete_cnt = 0
                while count < total_N:
                    self.concat_trajs = self.gen_batch_trajs(policy=policy,
                                                             burn_in=burn_in,
                                                             S_inits=None,
                                                             A_inits=None,
                                                             seed=seed,
                                                             evaluation=False)
                    S_traj, A_traj, reward_traj, T, survival_prob_traj, dropout_next_traj, dropout_prob_traj, S_traj_mask = self.concat_trajs
                    if count + np.sum(T) <= total_N:
                        sub_n = len(S_traj)
                    else:
                        sub_n = np.where(
                            T.cumsum() > total_N - count)[0].min() + 1
                    # for j in range(i, i + sub_n):
                    for j in range(sub_n):
                        # print(f'S_traj_mask.shape:{S_traj_mask.shape}')
                        observed_state_index = np.where(S_traj_mask[j] == 1)[0]
                        observed_index = observed_state_index[
                            observed_state_index < self.max_T -
                            burn_in].tolist()
                        observed_state_index = observed_state_index.tolist()
                        self.masked_buffer[(j + i)] = [
                            S_traj[j][observed_state_index],
                            # S_traj[j][observed_index],
                            A_traj[j][observed_index],
                            reward_traj[j][observed_index],
                            T[j],
                            survival_prob_traj[j][observed_index],
                            dropout_next_traj[j][observed_index],
                            dropout_prob_traj[j][observed_index]
                        ]
                        self.full_buffer[(j + i)] = [
                            S_traj[j], A_traj[j], reward_traj[j], T[j],
                            survival_prob_traj[j], dropout_next_traj[j],
                            dropout_prob_traj[j]
                        ]
                        if T[j] < self.max_T:
                            incomplete_cnt += 1
                        count += T[j]
                    i += sub_n
                self.n = i
            self.total_N = count
        self.dropout_rate = incomplete_cnt / self.n
        self.missing_rate = 1 - self.total_N / (self.n * self.max_T)
        initial_obs = []
        all_obs = []
        for k in self.masked_buffer.keys():
            initial_obs.append(self.masked_buffer[k][0][0])
            all_obs.append(self.masked_buffer[k][0])
        
        self._initial_obs = np.vstack(initial_obs)
        self._obs = np.vstack(all_obs)

    def export_buffer(self):
        """Convert unscaled trajectories in self.masked_buffer to a dataframe.
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
            if len(action_list) < nrows:
                tmp['action'] = np.append(action_list, [None])
                tmp['reward'] = np.append(self.masked_buffer[k][2], [None])
                tmp['surv_prob'] = np.append(self.masked_buffer[k][4], [None])
                tmp['dropout_prob'] = np.append(self.masked_buffer[k][6], [None])
            else:
                tmp['action'] = action_list
                tmp['reward'] = self.masked_buffer[k][2]
                tmp['surv_prob'] = self.masked_buffer[k][4]
                tmp['dropout_prob'] = self.masked_buffer[k][6]
            df = df.append(tmp)
        df = df.reset_index(drop=True)
        return df

    def import_buffer(self,
                      data=None,
                      data_filename=None,
                      static_state_cols=None,
                      dynamic_state_cols=None,
                      action_col='action',
                      reward_col='reward',
                      id_col='id',
                      dropout_col=None,
                      subsample_id=None,
                      reward_transform=None,
                      burn_in=0,
                      mnar_nextobs_var=None,
                      mnar_noninstrument_var=None,
                      mnar_instrument_var=None,
                      **kwargs):
        """Import unscaled trajectories from a dataframe to masked_buffer.

        Args:
            data (pd.DataFrame): dataframe to be imported
            data_filename (str): path to the table
            static_state_cols (list): column names of static features
            dynamic_state_cols (list): column names of dynamic features
            action_col (str): name of action column
            reward_col (str): name of reward column
            id_col (str): name of reward column
            dropout_col (str): name of the binary column that indicates dropout event
            subsample_id (list): ids of subsample to be imported
            reward_transform (callable): transformation funtion of reward
            burn_in (int): length of burn-in period
            mnar_nextobs_var (str): column name of the next observation for MNAR inference
            mnar_noninstrument_var (list): column names of noninstrument variable for MNAR inference
            mnar_instrument_var (str): column name of instrument variable for MNAR inference
            kwargs (dict): kwargs passed to pd.read_csv()        
        """
        assert data is not None or data_filename is not None, "please provide data or data_filename."
        if data is None:
            print(f'Import from {data_filename} to buffer')
            data = pd.read_csv(data_filename, **kwargs)

        state_cols = static_state_cols + dynamic_state_cols
        self.state_cols = state_cols
        self._df = data
        self.num_actions = data[action_col].nunique()
        self.state_dim = len(state_cols)
        self.last_obs = None
        self.burn_in = burn_in
        if subsample_id is None:
            id_list = data[id_col].unique().tolist()
        else:
            id_list = subsample_id
        num_trajs = len(id_list)
        self.masked_buffer = {} # observed data
        self.misc_buffer = {} # miscellaneous info
        if state_cols is None:
            state_cols = [c for c in data.columns if c.startswith('X')]
        incomplete_cnt = 0
        self.total_N = 0
        self._initial_obs = []
        if 'custom_dropout_prob' not in data.columns and dropout_col in data.columns:
            print(f'use column {dropout_col} as dropout indicator')
        for i, id_ in enumerate(id_list):
            tmp = data.loc[data[id_col] == id_]
            S_traj = tmp[state_cols].values
            self._initial_obs.append(S_traj[0])
            A_traj = tmp[action_col].values
            reward_traj = tmp[reward_col].values
            if reward_transform is not None:
                reward_traj = reward_transform(reward_traj)
            T = len(A_traj)
            if 'custom_dropout_prob' not in tmp.columns:
                if 'surv_prob' in tmp.columns:
                    survival_prob_traj = tmp['surv_prob'].values.astype(float)
                    dropout_prob_traj = 1 - survival_prob_traj[
                        1:] / survival_prob_traj[:-1]
                    dropout_prob_traj = np.append(np.array([0]),
                                                  dropout_prob_traj)
                else:
                    survival_prob_traj = None
                    dropout_prob_traj = None
                if dropout_col in tmp.columns:
                    dropout_next_traj = tmp[dropout_col].values
                elif len(reward_traj) < self.max_T:
                    dropout_next_traj = np.zeros_like(reward_traj)
                    dropout_next_traj[-1] = 1
                else:
                    dropout_next_traj = np.zeros_like(reward_traj)
            else:
                dropout_prob_traj = tmp['custom_dropout_prob'].values.astype(float)
                if dropout_prob_traj[-1] is None:
                    dropout_prob_traj[-1] = -1
                survival_prob_traj = np.append(
                    1, (1 - dropout_prob_traj).cumprod())[:-1]

                # dropout_index = np.where(
                #     np.random.uniform(
                #         low=0, high=1, size=dropout_prob_traj.shape) <
                #     dropout_prob_traj)[0]
                # dropout_next_traj = np.zeros_like(reward_traj)
                # if len(dropout_index) > 0:
                #     dropout_index = dropout_index.min()
                #     dropout_next_traj[dropout_index] = 1
                # else:
                #     dropout_index = self.max_T
                # if dropout_prob_traj[-1] < 0:
                #     dropout_prob_traj[-1] = np.nan

                assert dropout_col in tmp.columns
                dropout_next_traj = tmp[dropout_col].values
                dropout_index = np.where(dropout_next_traj == 1)[0]
                if len(dropout_index) > 0:
                    dropout_index = dropout_index.min()
                else:
                    dropout_index = self.max_T
                T = min(dropout_index + 1, self.max_T)
                S_traj = S_traj[:T]
                A_traj = A_traj[:T]
                reward_traj = reward_traj[:T]
                survival_prob_traj = survival_prob_traj[:T]
                dropout_next_traj = dropout_next_traj[:T]
                dropout_prob_traj = dropout_prob_traj[:T]

            self.total_N += T
            # if T < self.max_T:
            #     incomplete_cnt += 1
            incomplete_cnt += sum(dropout_next_traj)

            if T > burn_in:
                self.masked_buffer[(i)] = [
                    S_traj[burn_in:], 
                    A_traj[burn_in:], 
                    reward_traj[burn_in:],
                    T,
                    survival_prob_traj[burn_in:] if survival_prob_traj is not None else None,
                    dropout_next_traj[burn_in:], 
                    dropout_prob_traj[burn_in:] if dropout_prob_traj is not None else None
                ]
            if mnar_nextobs_var is not None or mnar_noninstrument_var is not None is not None or mnar_instrument_var is not None:
                self.misc_buffer[(i)] = {}
            if mnar_nextobs_var is not None:
                mnar_nextobs_arr = tmp[mnar_nextobs_var].values
                if len(mnar_nextobs_arr.shape) == 1:
                    mnar_nextobs_arr = mnar_nextobs_arr.reshape(-1, 1)
                self.misc_buffer[(i)]['mnar_nextobs_arr'] = np.vstack([
                    mnar_nextobs_arr[burn_in + 1:],
                    np.zeros(shape=(1, mnar_nextobs_arr.shape[1])) * np.nan
                ]) # shift one step
            if mnar_noninstrument_var is not None:
                mnar_noninstrument_arr = tmp[mnar_noninstrument_var].values
                self.misc_buffer[(
                    i
                )]['mnar_noninstrument_arr'] = mnar_noninstrument_arr[burn_in:]
            if mnar_instrument_var is not None:
                mnar_instrument_arr = tmp[mnar_instrument_var].values
                self.misc_buffer[(
                    i)]['mnar_instrument_arr'] = mnar_instrument_arr[burn_in:]

        self.n = len(self.masked_buffer)
        self.dropout_rate = incomplete_cnt / self.n
        self.missing_rate = 1 - self.total_N / (self.n * self.max_T)
        self._initial_obs = np.array(self._initial_obs)
        print('Import finished!')

    def import_holdout_buffer(self,
                              data=None,
                              data_filename=None,
                              static_state_cols=None,
                              dynamic_state_cols=None,
                              action_col='action',
                              reward_col='reward',
                              id_col='id',
                              dropout_col=None,
                              reward_transform=None,
                              burn_in=0,
                              **kwargs):
        """Import unscaled trajectories from a dataframe to self.holdout_buffer. 
        This buffer is used to estimate behavior policy or learn optimal policy.

        Args:
            data (pd.DataFrame): dataframe to be imported
            data_filename (str): path to the table
            static_state_cols (list): column names of static features
            dynamic_state_cols (list): column names of dynamic features
            action_col (str): name of action column
            reward_col (str): name of reward column
            id_col (str): name of reward column
            dropout_col (str): name of the binary column that indicates dropout event
            subsample_id (list): ids of subsample to be imported
            reward_transform (callable): transformation funtion of reward
            burn_in (int): length of burn-in period
            kwargs (dict): kwargs passed to pd.read_csv()  
        """
        if data is None:
            print(f'Import from {data_filename} to buffer')
            data = pd.read_csv(data_filename, **kwargs)

        state_cols = static_state_cols + dynamic_state_cols
        if hasattr(self, 'state_cols'):
            assert self.state_cols == state_cols
        if hasattr(self, 'num_actions'):
            assert self.num_actions == data[action_col].nunique()
        if hasattr(self, 'state_dim'):
            assert self.state_dim == len(state_cols)
        if hasattr(self, 'burn_in'):
            assert self.burn_in == burn_in

        id_list = data[id_col].unique().tolist()
        num_trajs = len(id_list)
        self.holdout_buffer = {}
        if state_cols is None:
            state_cols = [c for c in data.columns if c.startswith('X')]
        for i, id_ in enumerate(id_list):
            tmp = data.loc[data[id_col] == id_]
            S_traj = tmp[state_cols].values
            A_traj = tmp[action_col].values
            reward_traj = tmp[reward_col].values
            if reward_transform is not None:
                reward_traj = reward_transform(reward_traj)
            T = len(A_traj)
            if 'surv_prob' in tmp.columns:
                survival_prob_traj = tmp['surv_prob'].values
                dropout_prob_traj = 1 - survival_prob_traj[
                    1:] / survival_prob_traj[:-1]
                dropout_prob_traj = np.append(np.array([0]), dropout_prob_traj)
            else:
                survival_prob_traj = None
                dropout_prob_traj = None

            dropout_next_traj = np.zeros_like(reward_traj)
            if dropout_col in tmp.columns:
                dropout_next_traj = tmp[dropout_col].values
            elif len(dropout_next_traj) < self.max_T:
                dropout_next_traj[-1] = 1

            if T > burn_in:
                self.holdout_buffer[(i)] = [
                    S_traj[burn_in:], A_traj[burn_in:], reward_traj[burn_in:],
                    T, survival_prob_traj[burn_in:]
                    if survival_prob_traj is not None else None,
                    dropout_next_traj[burn_in:], dropout_prob_traj[burn_in:]
                    if dropout_prob_traj is not None else None
                ]
        print('Import finished!')

    def evaluate_pointwise_Q(
        self,
        policy,
        eval_size=20,
        eval_horizon=None,
        pointwise_eval_size=20,
        seed=None,
        S_inits=None,
        S_inits_kwargs={}
        ):
        """Evaluate pointwise Q and value

        Args:
            policy (callable): target policy to be evaluated
            eval_size (int): number of initial states to evaluate the policy
            eval_horizon (int): horizon of Monte Carlo approxiamation
            pointwise_eval_size (int): Monte Carlo size to estimate point-wise value
            seed (int): random seed passed to gen_batch_trajs()
            S_inits (np.ndarray): initial states to evaluate the policy
            S_inits_kwargs (dict): additional kwargs passed to env.reset()

        Returns:
            S_inits (np.ndarray): array of initial states to evaluate the policy
            true_value_avg (list): corresponding point-wise V(s)
            true_Q_dict (dict): corresponding point-wise Q(s,a), key is the action name
        """
        if S_inits is not None:
            if len(S_inits) == 1:
                S_inits = np.expand_dims(S_inits, axis=0)
            eval_size = len(S_inits)
        elif self.vectorized_env:
            S_inits = np.vstack([
                self.eval_env.single_observation_space.sample()
                for _ in range(eval_size)
            ])

        if self.vectorized_env:
            total_eval_size = pointwise_eval_size * eval_size
            if S_inits is not None:
                S_inits = np.repeat(S_inits,
                                    repeats=pointwise_eval_size,
                                    axis=0)
            old_num_envs = self.eval_env.num_envs
            old_horizon = self.eval_env.T
            # reset num_envs
            if eval_horizon is not None:
                self.eval_env.T = eval_horizon
            self.eval_env.num_envs = total_eval_size
            self.eval_env.observation_space = batch_space(
                self.eval_env.single_observation_space,
                n=self.eval_env.num_envs)
            self.eval_env.action_space = Tuple(
                (self.eval_env.single_action_space, ) * self.eval_env.num_envs)

            print('eval V: generate trajectories')
            trajectories = self.gen_batch_trajs(
                policy=policy,
                seed=seed,
                S_inits=S_inits if S_inits is not None else None,
                A_inits=None,
                burn_in=0,
                evaluation=True)
            rewards_history = self.eval_env.rewards_history
            print('eval V: calculate value')
            true_value = np.matmul(
                rewards_history,
                self.gamma**np.arange(start=0,
                                      stop=rewards_history.shape[1]).reshape(
                                          -1, 1))  # (total_eval_size, 1)
            true_value = true_value.reshape(eval_size, pointwise_eval_size)
            true_value_avg = true_value.mean(axis=1).tolist()
            self.eval_env.reset()
            del rewards_history
            _ = gc.collect()
            # get true Q for each action
            true_Q_dict = {}
            for a in range(self.eval_env.single_action_space.n):
                print('eval Q: generate trajectories')
                A_inits = np.repeat(a, repeats=len(S_inits))
                batch = self.gen_batch_trajs(
                    policy=policy,
                    seed=seed,
                    S_inits=S_inits if S_inits is not None else None,
                    A_inits=A_inits,
                    burn_in=0,
                    evaluation=True)
                rewards_history = self.eval_env.rewards_history * \
                    self.eval_env.states_history_mask[:,
                                                      :self.eval_env.rewards_history.shape[1]]
                print('eval Q: calculate value')
                true_Q = np.matmul(
                    rewards_history,
                    self.gamma**np.arange(
                        start=0, stop=rewards_history.shape[1]).reshape(-1, 1))
                true_Q = true_Q.reshape(eval_size, pointwise_eval_size)
                true_Q_avg = true_Q.mean(axis=1).tolist()
                true_Q_dict[a] = true_Q_avg

            # recover num_envs
            self.eval_env.T = old_horizon
            self.eval_env.num_envs = old_num_envs
            self.eval_env.observation_space = batch_space(
                self.eval_env.single_observation_space,
                n=self.eval_env.num_envs)
            self.eval_env.action_space = Tuple(
                (self.eval_env.single_action_space, ) * self.eval_env.num_envs)
            del trajectories, rewards_history
            _ = gc.collect()
            return S_inits[::pointwise_eval_size], true_value_avg, true_Q_dict
        else:
            old_horizon = self.eval_env.T
            if eval_horizon is not None:
                self.eval_env.T = eval_horizon
            print('eval_V: generate trajectories')
            self.eval_env.T = old_horizon
            true_value_list = []
            if S_inits is None and S_inits_kwargs:
                S_inits_sample = []
            for i in range(eval_size):
                rewards_histories_list = []
                for j in range(pointwise_eval_size):
                    traj = self.gen_single_traj(
                        policy=policy,
                        seed=None,
                        S_init=S_inits[i] if S_inits is not None else None,
                        S_init_kwargs={
                            k: v[i]
                            for k, v in S_inits_kwargs.items()
                        } if S_inits_kwargs else {},
                        A_init=None,
                        burn_in=0,
                        evaluation=True)
                    rewards_history = self.eval_env.rewards_history
                    rewards_histories_list.append(rewards_history)
                rewards_histories_arr = np.array(rewards_histories_list) # (pointwise_eval_size, eval_T)
                true_value = np.matmul(
                    rewards_histories_arr,
                    self.gamma**np.arange(
                        start=0, stop=rewards_histories_arr.shape[1]).reshape(
                            -1, 1)).reshape(-1)  # (pointwise_eval_size,)
                true_value_list.append(true_value)
                if S_inits is None and S_inits_kwargs:
                    S_inits_sample.append(traj[0][0])
            true_value_arr = np.array(true_value_list) # (eval_size,pointwise_eval_size)
            if S_inits is None and S_inits_kwargs:
                S_inits_sample = np.array(S_inits_sample)
                print(f'S_inits_sample: {S_inits_sample}')
            true_value_avg = true_value_arr.mean(
                axis=1).tolist()  # (eval_size,)
            self.eval_env.reset()
            del rewards_history, rewards_histories_list, rewards_histories_arr
            _ = gc.collect()
            # get true Q for each action
            true_Q_dict = {}
            for a in range(self.num_actions):
                print('eval_Q: generate trajectories')
                true_Q_list = []
                for i in range(eval_size):
                    rewards_histories_list = []
                    for j in range(pointwise_eval_size):
                        traj = self.gen_single_traj(
                            policy=policy,
                            seed=None,  # seed
                            S_init=S_inits[i] if S_inits is not None else None,
                            A_init=a,
                            burn_in=0,
                            evaluation=True)
                        rewards_history = self.eval_env.rewards_history
                        rewards_histories_list.append(rewards_history)
                    rewards_histories_arr = np.array(rewards_histories_list) # (pointwise_eval_size, eval_T)
                    true_Q = np.matmul(
                        rewards_histories_arr,
                        self.gamma**np.arange(
                            start=0,
                            stop=rewards_histories_arr.shape[1]).reshape(
                                -1))  # (pointwise_eval_size,)
                    true_Q_list.append(true_Q)
                true_Q_arr = np.array(true_Q_list) # (eval_size,pointwise_eval_size)
                true_Q_avg = true_Q_arr.mean(axis=1).tolist()  # (eval_size,)
                true_Q_dict[a] = true_Q_avg
            self.eval_env.T = old_horizon
            del rewards_history, rewards_histories_list, rewards_histories_arr
            _ = gc.collect()
            if S_inits is None and S_inits_kwargs:
                S_inits = S_inits_sample
            return S_inits, true_value_avg, true_Q_dict

    def get_target_value(self, target_policy, S_inits=None, MC_size=None):
        """Wrapper function of V_int
        
        Args:
            target_policy (callable): target policy to be evaluated.
            S_inits (np.ndarray): initial states for policy evaluation. If both S_inits and MC_size are not 
                specified, use initial observations from data.
            MC_size (int): sample size for Monte Carlo approximation.

        Returns:
            est_V (float): integrated value of target policy
        """
        raise NotImplementedError

