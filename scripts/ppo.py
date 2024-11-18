import torch
import torch.nn as nn
import torch.optim as optim

from network import ActorCritic
from rollout_storage_v1 import RolloutStorage


class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 rollout_storage,
                 num_learning_epochs=1,
                 num_mini_batches=1,
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
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = rollout_storage
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss


    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)


    def test_mode(self):
        self.actor_critic.test()


    def train_mode(self):
        self.actor_critic.train()


    def act(self, obs, critic_obs):
        # Get the action and the corresponding probability distribution
        action, action_probs = self.actor_critic.act(obs)

        # Save the action to transition
        self.transition.actions = action.detach()
        self.transition.action_probs = action_probs.detach()

        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions, self.transition.action_probs).detach()

        # self.transition.action_mean = self.actor_critic.action_mean.detach()
        # self.transition.action_sigma = self.actor_critic.action_std.detach()

        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs

        # Clear variables to free memory after recording
        del obs, critic_obs # 清理变量
        torch.cuda.empty_cache()  # 释放显存缓存
        return self.transition.actions


    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.actor_critic.reset(dones)


    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)


    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, actions_batch, rewards_batch, dones_batch, values_batch, actions_log_prob_batch, action_probs_batch in generator:
            actions_batch = actions_batch.to(torch.int64)
            # 动作 log_prob 计算
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch, action_probs_batch)
            value_batch = self.actor_critic.evaluate(obs_batch)

            rewards_batch = rewards_batch.unsqueeze(1)
            # KL 散度和熵项可以忽略
            ratio = torch.exp(actions_log_prob_batch - actions_log_prob_batch.detach())
            surrogate_loss1 = ratio * rewards_batch  # 使用 advantage 代替 reward 更好
            surrogate_loss2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * rewards_batch
            surrogate_loss = -torch.min(surrogate_loss1, surrogate_loss2).mean()

            # 值函数损失
            value_loss = (rewards_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss

            # 反向传播和梯度更新
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

            del obs_batch, actions_batch, rewards_batch, dones_batch, values_batch
            del actions_log_prob_batch, action_probs_batch
            torch.cuda.empty_cache()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()
        self.transition.clear()
        torch.cuda.empty_cache()

        return mean_value_loss, mean_surrogate_loss