import torch

class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            # critic use same obs
            # self.critic_observations = None   
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_probs = None
            # self.hidden_states = None
        
        def clear(self):
            self.__init__()
            self.observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_probs = None
            torch.cuda.empty_cache()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, actions_shape, device="cuda" if torch.cuda.is_available() else "cpu"):

        self.device = device

        self.obs_shape = obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        # only the chosen action's log prob
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        # self.action_probs = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.action_probs = torch.zeros(num_transitions_per_env, num_envs, 27, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0


    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions.clone().detach())
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.action_probs[self.step].copy_(transition.action_probs)
        self.step += 1

    def clear(self):
        self.step = 0
        self.observations = torch.zeros(self.num_transitions_per_env, self.num_envs, *self.obs_shape, device=self.device)
        self.actions = torch.zeros(self.num_transitions_per_env, self.num_envs, *self.actions_shape, device=self.device)
        self.rewards = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.dones = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.values = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.actions_log_prob = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.action_probs= torch.zeros(self.num_transitions_per_env, self.num_envs, 27, device=self.device)
        self.returns = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.advantages = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        torch.cuda.empty_cache()


    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD learning update state value
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)


    # get some statistics , not influcing the training process ----visualization
    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()


    def mini_batch_generator(self, num_mini_batches, num_learning_epochs):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

        # 打乱索引，生成小批次
        for epoch in range(num_learning_epochs):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, mini_batch_size):
                obs_batch = self.observations.view(-1, *self.obs_shape)[indices[start:start + mini_batch_size]]
                actions_batch = self.actions.view(-1, *self.actions_shape)[indices[start:start + mini_batch_size]]
                rewards_batch = self.rewards.view(-1)[indices[start:start + mini_batch_size]]
                dones_batch = self.dones.view(-1)[indices[start:start + mini_batch_size]]
                values_batch = self.values.view(-1)[indices[start:start + mini_batch_size]]
                actions_log_prob_batch = self.actions_log_prob.view(-1)[indices[start:start + mini_batch_size]]
                action_probs_batch = self.action_probs.view(-1,27)[indices[start:start + mini_batch_size]]  # 确保形状匹配

                yield obs_batch, actions_batch, rewards_batch, dones_batch, values_batch, actions_log_prob_batch, action_probs_batch

                del (obs_batch, actions_batch, rewards_batch, dones_batch,
                     values_batch, actions_log_prob_batch, action_probs_batch)
                torch.cuda.empty_cache()
