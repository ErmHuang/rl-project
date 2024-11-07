import torch
import torch.nn as nn
import torch.optim as optim

from .network import ActorCritic
from .rollout_storage_v1 import RolloutStorage


class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        # self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, action_shape, self.device)

    # def test_mode(self):
    #     self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        # self.transition.action_mean = self.actor_critic.action_mean.detach()
        # self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        # self.transition.critic_observations = obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)
    
    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        obs_batch = self.storage.observations.flatten(0, 1)
        actions_batch = self.storage.actions.flatten(0, 1)
        target_values_batch = self.storage.values.flatten(0, 1)
        returns_batch = self.storage.returns.flatten(0, 1)
        old_actions_log_prob_batch = self.storage.actions_log_prob.flatten(0, 1)
        advantages_batch = self.storage.advantages.flatten(0, 1)

        # Perform forward pass to get updated log probabilities and values
        logits = self.actor_critic.actor(obs_batch)
        dist = torch.distributions.Categorical(logits=logits)
        actions_log_prob_batch = dist.log_prob(actions_batch)
        value_batch = self.actor_critic.evaluate(obs_batch)

        # Surrogate loss calculation
        ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
        surrogate = ratio * advantages_batch
        surrogate_clipped = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages_batch
        surrogate_loss = -torch.min(surrogate, surrogate_clipped).mean()

        # Value function loss
        value_loss = (returns_batch - value_batch).pow(2).mean()

        # Total loss
        loss = surrogate_loss + self.value_loss_coef * value_loss

        # Gradient update
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        mean_value_loss = value_loss.item()
        mean_surrogate_loss = surrogate_loss.item()

        # Clear storage after update
        self.storage.clear()
        return mean_value_loss, mean_surrogate_loss



    

