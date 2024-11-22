import torch
from robotic_arm_gym_v1 import RoboticArmEnv
from network import ActorCritic

# Configuration
model_path = "./logs/model_checkpoint_5000.pth"  # Path to the model checkpoint file

# Initialize the environment and model
env = RoboticArmEnv()
actor_critic = ActorCritic(
    state_dim=env.num_obs,
    action_dim=env.num_actions,
    hidden_layers=[256, 256],  # Ensure it matches the training configuration
    activation="ReLU"
)

# Load the saved model weights
# actor_critic.load_state_dict(torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
actor_critic.load_state_dict(torch.load(model_path, map_location="cpu"))
actor_critic.eval()  # Set the model to evaluation mode

# Run the environment using the loaded model
state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
done = False

while not done:
    with torch.no_grad():
        action, _ = actor_critic.act(state)  # Assuming actor_critic outputs a tuple, take the first element
    next_state, reward, done, _ = env.step(action)
    
    env.render()  # Render the environment (optional)
    state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)


