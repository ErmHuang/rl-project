# import mujoco_py
# import os

# # Set your .xml file path
# xml_path = os.getcwd() + "/model/robotic_arm.xml"

# # Create the Mujoco model
# model = mujoco_py.load_model_from_path(xml_path)
# sim = mujoco_py.MjSim(model)

# # Create the renderer
# viewer = mujoco_py.MjViewer(sim)

# # Run the simulation and render
# for _ in range(10000): # 1000 simulation steps
#     sim.step()  # Simulate one step
#     viewer.render()  # Render the simulation environment



import gym
import robotic_arm_gym_v1  # Import the custom environment module

# Create the environment
# env = gym.make('RoboticArm-v0')
env = gym.make('RoboticArm-v1')

# Reset the environment and get the initial observations
observation = env.reset()

count = 0
# Test the environment's execution
for _ in range(10000):  # Simulate 1000 time steps
    env.render()  # Render the simulation environment

    action = env.action_space.sample()  # Randomly generate an action
    print(f"Action: ", action)

    observation, reward, done, info = env.step(action)  # Execute the action and get feedback
    # print("Observation: ", observation)
    print("Get Reward: ", reward)

    count += 1

    if count % 1000 == 0:
        env.reset()


    if done:  # Check if done
        observation = env.reset()  # Reset environment
env.close()  # Close environment




# import gym

# # Create the  Reacher environment
# env = gym.make('Reacher-v2') 

# # Reset the environment and obtain the initial observation
# observation = env.reset()  # Only get one return value

# # Perform the simulation
# for _ in range(1000):  # Set the number of simulation steps
#     action = env.action_space.sample()   # Randomly generate an action
#     observation, reward, done, info = env.step(action)  # Apply the action

#     if done:
#         observation = env.reset()  # Reset the environment if the termination condition is reached

#     env.render() # Render the simulation environment

# env.close()  # Close environment