import matplotlib.pyplot as plt
import numpy as np

# 读取数据
file_path = "/home/erm/Desktop/rewards_05.txt"
rewards = []

# 提取 rewards 数据
with open(file_path, 'r') as file:
    for line in file:
        if "Total Reward" in line:
            reward = float(line.split("=")[-1].strip())
            rewards.append(reward)

# 计算平均值
average_reward = np.mean(rewards)
print(f"平均 Reward: {average_reward:.2f}")

# 累积平均值 (100 episode 窗口)
window_size = 100
cumulative_avg_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

# 绘图
episodes = np.arange(1, len(rewards) + 1)

plt.figure(figsize=(12, 6))

# 绘制原始 reward 曲线
plt.plot(episodes, rewards, label="Reward per Episode", alpha=0.7)

# 绘制累积平均曲线
cumulative_avg_episodes = np.arange(window_size, len(rewards) + 1)
plt.plot(cumulative_avg_episodes, cumulative_avg_rewards, label=f"Moving Average ({window_size} episodes)", color='orange', linewidth=2)

plt.axhline(y=average_reward, color='red', linestyle='--', linewidth=2, label=f"Average Reward ({average_reward:.2f})")

# 图形美化
plt.title("Reward vs Episodes", fontsize=16)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Reward", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)

# 显示图像
plt.tight_layout()
plt.show()
