import torch
import os
from robotic_arm_gym_v1 import RoboticArmEnv
from network import ActorCritic
from ppo import PPO
from rollout_storage_v1 import RolloutStorage  

from logger import Logger
from collections import defaultdict


class LearningAgent:
    def __init__(self, env, train_cfg, device="cpu",pretrained_model_path=None):
        # Initialize environment, policy networks and algorithms

        self.env = env
        self.device = device
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]

        self.actor_critic = ActorCritic(
            state_dim=self.env.num_obs,
            action_dim=self.env.num_actions,
            hidden_layers=self.policy_cfg["hidden_layers"],
            activation=self.policy_cfg["activation"]
        ).to(self.device)

        # 如果指定了预训练模型路径，加载预训练权重
        if pretrained_model_path:
            self.actor_critic.load_state_dict(torch.load(pretrained_model_path, map_location=self.device))
            print(f"Loaded pretrained model from {pretrained_model_path}")

        # Set up RolloutStorage to store data
        self.rollout_storage = RolloutStorage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.cfg["num_steps_per_env"],  # number of steps in each environment
            obs_shape=(self.env.num_obs,),
            actions_shape=(self.env.num_actions,),
            device=self.device
        )

        self.ppo = PPO(self.actor_critic, rollout_storage=self.rollout_storage, device=self.device, **self.alg_cfg)

        # Other parameters
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.log_dir = self.cfg.get("log_dir", None)
        self.tot_timesteps = 0
        self.current_learning_iteration = 0

        self.logger = Logger(dt=0.02)

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = torch.tensor(self.env.reset(), dtype=torch.float32).unsqueeze(0).to(self.device)
            self.rollout_storage.clear()  # Clear storage at the beginning of each episode


            episode_rewards = defaultdict(float)

            # Data collection
            for step in range(self.num_steps_per_env):
                action = self.ppo.act(state, state)  # Select action based on state
                next_state, reward, done, _ = self.env.step(action.item())

                 # Record state and action in logger
                self.logger.log_state("state", state.cpu().numpy())
                self.logger.log_state("action", action.cpu().numpy())
                episode_rewards["step_reward"] += reward  # Track total reward for episode

                # debug
                if step % 1000 == 0:
                    print(f"step:{step}, reward:{reward}")

                self.env.render()

                # Store current step data into RolloutStorage
                transition = self.rollout_storage.Transition()
                transition.observations = state
                transition.actions = action.clone().detach().to(torch.long).to(self.device)
                transition.rewards = torch.tensor([reward], dtype=torch.float32).to(self.device)
                transition.dones = torch.tensor([done], dtype=torch.float32).to(self.device)
                transition.values = self.ppo.transition.values
                transition.actions_log_prob = self.ppo.transition.actions_log_prob
                transition.action_probs = self.ppo.transition.action_probs

                self.rollout_storage.add_transitions(transition)

                # Update state
                state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
                if done:
                    break

            # Log rewards for this episode
            self.logger.log_rewards(episode_rewards, num_episodes=1)

            # Add total reward to state_log for plotting
            if "episode_rewards" not in self.logger.state_log:
                self.logger.state_log["episode_rewards"] = []
            self.logger.state_log["episode_rewards"].append(episode_rewards["step_reward"])

            # Calculate advantages and returns
            last_value = self.ppo.actor_critic.evaluate(state)
            self.rollout_storage.compute_returns(last_values=last_value, gamma=self.alg_cfg["gamma"],
                                                 lam=self.alg_cfg["lam"])

            # Update policy
            self.ppo.update()

            print(f"Episode {episode + 1}/{num_episodes} completed.")
            if self.log_dir and not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            if (episode + 1) % self.save_interval == 0:
                torch.save(self.actor_critic.state_dict(), f"{self.log_dir}/model_checkpoint_{episode + 1}.pth")
                print(f"Model checkpoint saved at episode {episode + 1}.")
        




train_cfg = {
    "runner": {
        "num_steps_per_env": 2048,
        "save_interval": 2000,
        "log_dir": "./logs"
    },
    "algorithm": {
        "gamma": 0.99,
        "lam": 0.95
    },
    "policy": {
        "hidden_layers": [256, 256],  # number of neurons per hidden layer
        "activation": "ReLU",  # type of activation function
    }
}

pretrained_model_path = "./logs/model_checkpoint_2000.pth"

# Initialize environment and LearningAgent, then start training
env = RoboticArmEnv()
# agent = LearningAgent(env, train_cfg, device="cuda" if torch.cuda.is_available() else "cpu")
agent = LearningAgent(env, train_cfg, device="cpu", pretrained_model_path=pretrained_model_path)
agent.train(num_episodes=10001)
agent.logger.print_rewards()
agent.logger.plot_states()




# play.py  ---- play the trained policy
# logger.py ---- what needs to be visualized to help training
