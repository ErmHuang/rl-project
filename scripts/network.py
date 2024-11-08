import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim=54, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Define the Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),  # Input layer
            nn.ReLU(),                         # Activation function
            nn.Linear(hidden_dim, hidden_dim),  # Hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),  # Output layer(the probability of an action)
            nn.Softmax(dim=-1)                  # Convert output to probability using Softmax
        )
        
        # Define the Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),   # Input layer
            nn.ReLU(),                          # Activation function
            nn.Linear(hidden_dim, hidden_dim),  # Hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)            # Output layer(the value of the current state)
        )

        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")
        
    def forward(self, state):
        # Forward propagation
        
        # Strategy network outputs probability distributions of actions
        policy_dist = self.actor(state)  
        # Value function network outputs the value of the state
        value = self.critic(state)       
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
        probabilities = self.actor(observations)  # Output is [batch_size, num_actions]
        self.distribution = probabilities  # Storing probability distributions


    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        
        # Sample using multinomial, select an action
        return torch.multinomial(self.distribution, num_samples=1)

    
    def get_actions_log_prob(self, actions):
        # Check the shape of actions and distribution to ensure compatibility
        if actions.dim() == 1:
            # If actions is of shape [batch_size], we need to add an extra dimension for gathering
            actions = actions.unsqueeze(1)

        # Gather the log probabilities for the actions taken
        action_log_probs = self.distribution.gather(1, actions).log()

        # Remove the unnecessary dimension, if required
        return action_log_probs.squeeze()

    def act_inference(self, observations):
        actions_distribution = self.actor(observations)
        return actions_distribution

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
    

    

# Demo
if __name__ == "__main__":
    # The state dimension is 19 and the action space has 54 discrete actions
    state_dim = 19
    action_dim = 54
    model = ActorCritic(state_dim, action_dim)
    
    # Generate a random state to test forward propagation
    test_state = torch.randn(1, state_dim)  # A random state with a batch size of 1
    policy, value = model(test_state)

    action = model.act(test_state)

    
    print(f"Policy (action probabilities): {policy}")
    print(f"Value (state value): {value}")
    print(f"chosen action: {action}")