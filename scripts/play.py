import torch
from robotic_arm_gym_v1 import RoboticArmEnv
from network import ActorCritic

# 配置
model_path = "./logs/model_checkpoint_6500.pth"  # 模型检查点文件路径

# 初始化环境和模型
env = RoboticArmEnv()
actor_critic = ActorCritic(
    state_dim=env.num_obs,
    action_dim=env.num_actions,
    hidden_layers=[256, 256],  # 确保与训练时的配置一致
    activation="ReLU"
)

# 加载保存的模型权重
actor_critic.load_state_dict(torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
# actor_critic.load_state_dict(torch.load(model_path, map_location="cpu"))
actor_critic.eval()  # 设置模型为评估模式

# 运行环境并使用加载的模型
state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
done = False

while not done:
    with torch.no_grad():
        action, _ = actor_critic.act(state)  # 假设 actor_critic 输出一个 tuple，取第一个元素
    next_state, reward, done, _ = env.step(action)
    
    env.render()  # 渲染环境（可选）
    state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)

