import torch
from network import ActorCritic
from ppo import PPO

# Define state and action dimensions for the environment
state_dim = 19
action_dim = 54

# Initialize ActorCritic model
model = ActorCritic(state_dim, action_dim)

# Initialize PPO instance using the model
ppo = PPO(actor_critic=model)

# Initialize RolloutStorage
num_envs = 1  # Assuming only one environment
num_transitions_per_env = 200  # Number of transitions to store
actor_obs_shape = (state_dim,)
action_shape = (action_dim,)

# Initialize the storage for PPO
ppo.init_storage(num_envs, num_transitions_per_env, actor_obs_shape, action_shape)

# Generate a random state to test the PPO workflow
test_state = torch.randn(1, state_dim)  # A random state with batch size of 1

# Simulate an action selection process using PPO
print("--- PPO Workflow Demonstration ---")

# Actor-Critic: Forward pass
policy, value = model(test_state)
print(f"Policy (action probabilities): {policy}")
print(f"Value (state value): {value}")

# PPO Action selection
action = ppo.act(test_state, test_state).squeeze()  # Ensure action tensor has the correct dimensions
print(f"Chosen action: {action}")

# Simulate PPO environment step and processing (assuming dummy values for reward and done flag)
reward = torch.tensor([1.0])
done = torch.tensor([0])
ppo.process_env_step(reward, done, {})

# Compute returns for PPO
last_value = torch.zeros(1, state_dim)  # Assuming last value is 0
ppo.compute_returns(last_value)

print("--- PPO Update Step ---")
# Perform a PPO update step (assuming training with dummy data)
mean_value_loss, mean_surrogate_loss = ppo.update()
print(f"Mean Value Loss: {mean_value_loss}")
print(f"Mean Surrogate Loss: {mean_surrogate_loss}")
