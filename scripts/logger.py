
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value

class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            self.log_state(key, value)

    def log_rewards(self, dict, num_episodes):
        current_episode_reward = 0
        for key, value in dict.items():
            if "step_reward" in key:
                reward = value.item() * num_episodes
                self.rew_log[key].append(reward)
                current_episode_reward += reward
            else:
                print("not found step_reward")
        self.num_episodes += num_episodes
        print(f"Episode {self.num_episodes}: Total Reward = {current_episode_reward}")

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self):
        self.plot_process = Process(target=self._plot)
        self.plot_process.start()

    def _plot(self):
        nb_rows = 1
        nb_cols = 1
        fig, axs = plt.subplots(nb_rows, nb_cols)

        # Time variable for plotting
        log = self.state_log
        if log["episode_rewards"]:
            time = np.arange(len(log["episode_rewards"]))  # index for every episode

            # Plot episode rewards
            a = axs[0, 0]
            a.plot(time, log["episode_rewards"], label='Episode Reward')
            a.set(xlabel='Episode', ylabel='Reward', title='Episode Reward')
            a.legend()

        # Existing plots (assuming log has the following data)
        if "dof_pos" in log:
            time = np.linspace(0, len(log["dof_pos"]) * self.dt, len(log["dof_pos"]))
            a = axs[1, 0]
            if log["dof_pos"]: a.plot(time, log["dof_pos"], label='measured')
            if log["dof_pos_target"]: a.plot(time, log["dof_pos_target"], label='target')
            a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
            a.legend()

        plt.tight_layout()
        plt.show()


    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")
    
    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()