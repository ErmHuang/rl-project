import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim=54, hidden_layers=[256, 256], activation="ReLU"):
        super(ActorCritic, self).__init__()

        # Actor network definition
        actor_layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            actor_layers.append(nn.Linear(input_dim, hidden_dim))
            actor_layers.append(getattr(nn, activation)())  # Dynamic activation function
            input_dim = hidden_dim
        # Combine hidden layers into a single network
        self.actor_body = nn.Sequential(*actor_layers)

        # Actor output layer
        self.actor_output = nn.Linear(input_dim, action_dim)
        self.actor_softmax = nn.Softmax(dim=-1)

        # Critic network definition (independent layers)
        critic_layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            critic_layers.append(nn.Linear(input_dim, hidden_dim))
            critic_layers.append(getattr(nn, activation)())
            input_dim = hidden_dim
        self.critic_body = nn.Sequential(*critic_layers)
        self.critic_output = nn.Linear(input_dim, 1)
        
    def forward(self, state):
        # Forward propagation
        # Actor part
        actor_features = self.actor_body(state)
        action_probs = self.actor_softmax(self.actor_output(actor_features))

        # Critic part
        critic_features = self.critic_body(state)
        state_value = self.critic_output(critic_features)

        return action_probs, state_value
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)


    def act(self, observations, **kwargs):
        # Compute action probability distribution
        actor_features = self.actor_body(observations)  # Pass through hidden layers
        action_probs = self.actor_softmax(self.actor_output(actor_features))   # Output layer and softmax for probabilities

        assert action_probs.shape[0] == observations.shape[0], "Batch size of action_probs does not match observations"
        # Sample an action based on the probability distribution
        action = torch.multinomial(action_probs, num_samples=1).squeeze(-1)

        # Store action_probs for log_prob calculation
        self.action_probs = action_probs
        return action, action_probs  # Return both action and probability distribution


    def get_actions_log_prob(self, actions, action_probs):
        actions = actions.to(torch.int64)
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)  # 确保 actions 维度为 [batch_size, 1]

        log_prob = action_probs.gather(1, actions).log()  # 使用 gather 索引
        return log_prob

        # return (self.action_probs.gather(1, actions.unsqueeze(1)).log()).squeeze()  # Calculate the log probability


    def act_inference(self, observations):
        # Compute action probability distribution
        actor_features = self.actor_body(observations)  # Pass through hidden layers
        actions_distribution = self.actor_softmax(self.actor_output(actor_features))  # Output layer and softmax for probabilities

        return actions_distribution

    def evaluate(self, critic_observations, **kwargs):
        critic_features = self.critic_body(critic_observations)
        value = self.critic_output(critic_features)
        return value
    
