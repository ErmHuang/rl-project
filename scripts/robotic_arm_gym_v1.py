import numpy as np
from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Discrete, Box
from gym.envs.registration import register
import os


"""
    ### Action Space
    The action space is a `Discrete(27)`. An action represents the increase or decrease torques applied at the hinge joints.

    | Num | Action                                                                          | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit |
    |-----|---------------------------------------------------------------------------------|-------------|-------------|--------------------------|-------|------|
    | 0   |  Torque applied at the first hinge (connecting the link to the point of fixture)| -10.0 | 10.0 | link_1_joint  | hinge | torque (N m) |
    | 1   |  Torque applied at the second hinge (connecting the two links)                  | -10.0 | 10.0 | link_2_joint  | hinge | torque (N m) |
    | 2   |  Torque applied at the third hinge (connecting the two links)                   | -10.0 | 10.0 | link_3_joint  | hinge | torque (N m) |


    | Num | Observation                                                                                    | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
    | --- | ---------------------------------------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | cosine of the angle of the first arm                                                           | -Inf | Inf | cos(link_1_joint)                | hinge | unitless                 |
    | 1   | cosine of the angle of the second arm                                                          | -Inf | Inf | cos(link_2_joint)                | hinge | unitless                 |
    | 2   | cosine of the angle of the third arm                                                           | -Inf | Inf | cos(link_3_joint)                | hinge | unitless                 |
    | 3   | sine of the angle of the first arm                                                             | -Inf | Inf | sin(link_1_joint)                | hinge | unitless                 |
    | 4   | sine of the angle of the second arm                                                            | -Inf | Inf | sin(link_2_joint)                | hinge | unitless                 |
    | 5   | sine of the angle of the third arm                                                             | -Inf | Inf | sin(link_3_joint)                | hinge | unitless                 |
    | 6   | x-coordinate of the target                                                                     | -Inf | Inf | target_x                         | slide | position (m)             |
    | 7   | y-coordinate of the target                                                                     | -Inf | Inf | target_y                         | slide | position (m)             |
    | 8   | z-coordinate of the target                                                                     | -Inf | Inf | target_z                         | slide | position (m)             |
    | 9   | x-coordinate of the end-effector                                                               | -Inf | Inf | target_x                         | slide | position (m)             |
    | 10  | y-coordinate of the end-effector                                                               | -Inf | Inf | target_y                         | slide | position (m)             |
    | 11  | z-coordinate of the end-effector                                                               | -Inf | Inf | target_z                         | slide | position (m)             |
    | 12  | angular velocity of the first arm                                                              | -Inf | Inf | link_1_joint                     | hinge | angular velocity (rad/s) |
    | 13  | angular velocity of the second arm                                                             | -Inf | Inf | link_2_joint                     | hinge | angular velocity (rad/s) |
    | 14  | angular velocity of the third arm                                                              | -Inf | Inf | link_3_joint                    | hinge | angular velocity (rad/s) |
    | 15  | x-value of position_fingertip - position_target                                                | -Inf | Inf | NA                               | slide | position (m)             |
    | 16  | y-value of position_fingertip - position_target                                                | -Inf | Inf | NA                               | slide | position (m)             |
    | 17  | z-value of position_fingertip - position_target                                                | -Inf | Inf | NA                               | slide | position (m)             |
    | 18  | end-effector push state                                                                        | 0    | 1   | NA                               | slide | NA                       |         

"""

# xml_path = os.getcwd() + "./model/robotic_arm.xml"
project_root = os.path.dirname(os.path.dirname(__file__))

# 使用相对路径指向 model/robotic_arm.xml
xml_path = os.path.join(project_root, "model", "robotic_arm.xml")

class RoboticArmEnv(MujocoEnv, utils.EzPickle):
    """
    Custom Gym Environment for a 3-DOF robotic arm with discrete action space
    and an additional action for pushing a button with the end-effector.
    """
    def __init__(self, **kwargs):
        # Call EzPickle initialization
        utils.EzPickle.__init__(self, **kwargs)

        # Define the observation space
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64)
        self.num_obs = self.observation_space.shape[0]

        # Discrete action space: 27 possible actions (3^3 for torques, plus 1 for the button)
        self.action_space = Discrete(27)
        self.num_actions = self.action_space.n

        # Initial torques (in the range [-1, 1]) for the three joints
        self.torques = np.zeros(3)
        # self.button_state = 0
        # self.end_effector_pushed = 0
        self.num_envs = 1
        # Initialize the Mujoco environment, passing the .xml model file path
        MujocoEnv.__init__(self, xml_path, 2, **kwargs)


    def _set_action_space(self):
        self.action_space = Discrete(27)
    

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        self.done = False  # Ensure the 'done' flag is initialized to False
        return ob


    def step(self, action):
        
        # Reset the end-effector push state before every step
        # self.end_effector_pushed = 0

        # Action mapping for torque changes on the joints
        action_map = [
            (-0.1, -0.1, -0.1), (-0.1, -0.1, 0), (-0.1, -0.1, 0.1),
            (-0.1, 0, -0.1), (-0.1, 0, 0), (-0.1, 0, 0.1),
            (-0.1, 0.1, -0.1), (-0.1, 0.1, 0), (-0.1, 0.1, 0.1),
            (0, -0.1, -0.1), (0, -0.1, 0), (0, -0.1, 0.1),
            (0, 0, -0.1), (0, 0, 0), (0, 0, 0.1),
            (0, 0.1, -0.1), (0, 0.1, 0), (0, 0.1, 0.1),
            (0.1, -0.1, -0.1), (0.1, -0.1, 0), (0.1, -0.1, 0.1),
            (0.1, 0, -0.1), (0.1, 0, 0), (0.1, 0, 0.1),
            (0.1, 0.1, -0.1), (0.1, 0.1, 0), (0.1, 0.1, 0.1)
        ]
        
        # # If action is < 27, it means it’s a torque action
        # if action < 27:
        #     torque_delta = np.array(action_map[action])
        #     self.torques = np.clip(self.torques + torque_delta, -10.0, 10.0)
        # else:
        #     # Action 27-53 represents the torque + button operation (action-27)
        #     torque_delta = np.array(action_map[action - 27])
        #     self.torques = np.clip(self.torques + torque_delta, -10.0, 10.0)
        #     self._attempt_push_button()  # Simulate the button push action

        end_effector_pos = self.get_body_com("end_effector")
        target_pos = self.get_body_com("target")
        relative_distance = np.linalg.norm(end_effector_pos - target_pos)

        torque_delta = np.array(action_map[action])
        self.torques = np.clip(self.torques + torque_delta, -10.0, 10.0)


        # Perform simulation with the updated torques
        self.do_simulation(self.torques, self.frame_skip)
        
        # Observation
        ob = self._get_obs()
        
        # Calculate the reward based on the current state
        reward, done = self._calculate_reward(relative_distance)

        # Return observations, reward, done flag, and additional info
        return ob, reward, done, relative_distance

    # def _attempt_push_button(self):
    #     """
    #     Handles the logic for the button push and updates the end-effector state.
    #     """
    #     end_effector_pos = self.get_body_com("end_effector")
    #     target_pos = self.get_body_com("target")
    #     relative_distance = np.linalg.norm(end_effector_pos - target_pos)

    #     self.end_effector_pushed = 1  # End-effector is in the "pushed" state

    #     # If the end-effector is close enough, mark it as pushed
    #     if relative_distance < 0.05:
    #         self.button_state = 1  # Button is switched to "on"
    #     else:
    #         self.button_state = 0  # Invalid push, button remains "off"

    def _calculate_reward(self, old_relative_distance):
        """
        Computes the reward based on the current state of the environment.
        """
        end_effector_pos = self.get_body_com("end_effector")
        target_pos = self.get_body_com("target")
        relative_distance = np.linalg.norm(end_effector_pos - target_pos)
        angular_velocity = self.data.qvel.flat[:3]

        reward = 0
        
        # 1. Dense negative reward based on the relative distance
        reward += - relative_distance ** 2 
        
        # 2. Positive reward of 1 when the end-effector is close enough to the target
        # if relative_distance < 0.05:
        #     reward += 10
        
        # 3. Penalty for invalid button press
        # if self.end_effector_pushed == 1 and relative_distance >= 0.05:
        #     reward -= 5  # Penalty for invalid push action

        # 4. Penalty for strong torques
        if np.sum(np.square(self.torques)) > 50.0:
            reward -= 0.25 + np.sum(np.square(self.torques)) * 0.01

        # 5. Penalty for high angular velocities
        if np.sum(np.square(angular_velocity)) > 1:
            reward -= np.sum(np.maximum(angular_velocity,-angular_velocity)) * 0.1 

        # 6. Penalty for not moving
        if np.sum(np.maximum(angular_velocity,-angular_velocity)) <= 0.1:
            reward -= 1.0

        #7. Penalty or award for getting closer
        if relative_distance < old_relative_distance:
            reward += 2.0
        else:
            reward -= 1.0

        # 7. Large positive reward when the button is successfully pushed
        if relative_distance <= 0.05:
            reward += 1000
            done = True
        else:
            done = False

        return reward, done
    
    def reset_model(self):
        # Randomly initialize the robotic arm position

        qpos = self.init_qpos.copy()  # Copy init_qpos to avoid modifying the original data
        qpos[:3] = self.np_random.uniform(low=-0.1, high=0.1, size=3)  # Update the positions of the first 3 joints       
        # qvel = self.np_random.uniform(low=-0.005, high=0.005, size=3) + self.init_qvel
        qvel = self.init_qvel.copy()  # Copy init_qvel to avoid modifying the original data
        qvel[:3] = self.np_random.uniform(low=-0.1, high=0.1, size=3)  # Update the velocities of the first 3 joints       
        
        # Randomly set the target position
        # self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=3)
        goal_x = self.np_random.uniform(-0.6, -0.2) if self.np_random.uniform() < 0.5 else self.np_random.uniform(0.2, 0.6)
        goal_y = self.np_random.uniform(-0.6, -0.2) if self.np_random.uniform() < 0.5 else self.np_random.uniform(0.2, 0.6)
        goal_z = self.np_random.uniform(low=0.1, high=0.6)
        self.goal = np.array([goal_x, goal_y, goal_z])
        self.sim.model.body_pos[self.model.body_name2id('target')] = self.goal
        
        self.set_state(qpos, qvel)
        # Reset torques, button state, and end-effector push state
        self.torques = np.zeros(3)
        # self.button_state = 0
        # self.end_effector_pushed = 0

        return self._get_obs()

    def _get_obs(self):
        # Get observations, including joint positions, velocities, and the distance between the end effector and the target
        theta = self.data.qpos.flat[:3]
        angular_velocity = self.data.qvel.flat[:3]  # Get the velocities of the 3 joints
        end_effector_pos = self.get_body_com("end_effector")
        target_pos = self.sim.model.body_pos[self.model.body_name2id('target')]

        position_diff = end_effector_pos - target_pos
        
        # Add the end-effector push state to the observation space
        observation = np.concatenate([
            np.cos(theta),              # Cosine of joint angles
            np.sin(theta),              # Sine of joint angles
            target_pos,                 # X and Y coordinates of the target
            end_effector_pos,           # X and Y coordinates of the end-effector
            angular_velocity,           # Angular velocities
            position_diff,              # Difference in positions (x, y, z)
            # [self.end_effector_pushed]  # End-effector push state (0 or 1)
        ])
        return observation

    def viewer_setup(self):
        # Set up the view
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0


register(
    id='RoboticArm-v1',
    entry_point='robotic_arm_gym_v1:RoboticArmEnv',  # Replace with the module name that defines the environment
)