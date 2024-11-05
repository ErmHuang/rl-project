import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim=54, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # 定义策略网络 (Actor)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # 输入层，状态维度到隐藏层
            nn.ReLU(),                         # 激活函数
            nn.Linear(hidden_dim, hidden_dim),  # 隐藏层
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),  # 输出层，输出动作的概率
            nn.Softmax(dim=-1)                  # 使用Softmax将输出转换为概率
        )
        
        # 定义值函数网络 (Critic)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),   # 输入层，状态维度到隐藏层
            nn.ReLU(),                          # 激活函数
            nn.Linear(hidden_dim, hidden_dim),  # 隐藏层
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)            # 输出层，输出当前状态的值
        )

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        
    def forward(self, state):
        # 前向传播
        policy_dist = self.actor(state)  # 策略网络输出动作的概率分布
        value = self.critic(state)       # 值函数网络输出状态的值
        return policy_dist, value
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        probabilities = self.actor(observations)  # 输出是 [batch_size, num_actions]
        self.distribution = probabilities  # 存储概率分布


    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        # 使用 multinomial 进行采样，选择一个动作
        return torch.multinomial(self.distribution, num_samples=1)  # 采样

    
    def get_actions_log_prob(self, actions):
        return (self.distribution.gather(1, actions.unsqueeze(1)).log()).squeeze()  # 计算 log 概率


    def act_inference(self, observations):
        actions_distribution = self.actor(observations)
        return actions_distribution

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    

    

# 示例
if __name__ == "__main__":
    # 状态维度是19，动作空间有54个离散动作
    state_dim = 19
    action_dim = 54
    model = ActorCritic(state_dim, action_dim)
    
    # 生成一个随机状态，测试前向传播
    test_state = torch.randn(1, state_dim)  # 一个batch大小为1的随机状态
    policy, value = model(test_state)

    action = model.act(test_state)

    
    print(f"Policy (action probabilities): {policy}")
    print(f"Value (state value): {value}")
    print(f"chosen action: {action}")
