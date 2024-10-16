import numpy as np
from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Discrete, Box
from gym.envs.registration import register



"""
    ### Action Space
    The action space is a `Discrete(54)`. An action represents the increase or decrease torques applied at the hinge joints.

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




class RoboticArmEnv(MujocoEnv, utils.EzPickle):
    """
    Custom Gym Environment for a 3-DOF robotic arm with discrete action space
    and an additional action for pushing a button with the end-effector.
    """
    def __init__(self, **kwargs):
        # 调用 EzPickle 初始化
        utils.EzPickle.__init__(self, **kwargs)
        
        # 定义观察空间
        observation_space = Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float64)

        # Discrete action space: 54 possible actions (3^3 for torques, plus 1 for the button)
        action_space = Discrete(54)

        # Initial torques (in the range [-1, 1]) for the three joints
        self.torques = np.zeros(3)
        self.button_state = 0
        self.end_effector_pushed = 0

        # 初始化 Mujoco 环境，传入 .xml 文件路径
        MujocoEnv.__init__(self, "robotic_arm.xml", 2, observation_space=observation_space, action_space=action_space, **kwargs)

    def step(self, action):
        print("Applied action: ", action)
        # Reset the end-effector push state before every step
        self.end_effector_pushed = 0

        # Action mapping for torque changes on the joints
        action_map = [
            (-1, -1, -1), (-1, -1, 0), (-1, -1, 1),
            (-1, 0, -1), (-1, 0, 0), (-1, 0, 1),
            (-1, 1, -1), (-1, 1, 0), (-1, 1, 1),
            (0, -1, -1), (0, -1, 0), (0, -1, 1),
            (0, 0, -1), (0, 0, 0), (0, 0, 1),
            (0, 1, -1), (0, 1, 0), (0, 1, 1),
            (1, -1, -1), (1, -1, 0), (1, -1, 1),
            (1, 0, -1), (1, 0, 0), (1, 0, 1),
            (1, 1, -1), (1, 1, 0), (1, 1, 1)
        ]
        
        # If action is < 27, it means it’s a torque action
        if action < 27:
            torque_delta = np.array(action_map[action])
            self.torques = np.clip(self.torques + torque_delta, -10.0, 10.0)
        else:
            # Action 27-53 represents the torque + button operation (action-27)
            torque_delta = np.array(action_map[action - 27])
            self.torques = np.clip(self.torques + torque_delta, -10.0, 10.0)
            self._attempt_push_button()  # Simulate the button push action

        # Perform simulation with the updated torques
        self.do_simulation(self.torques, self.frame_skip)
        
        # Observation
        ob = self._get_obs()
        
        # Calculate the reward based on the current state
        reward, done = self._calculate_reward()

        # Return observations, reward, done flag, and additional info
        return ob, reward, done, {}

    def _attempt_push_button(self):
        """
        Handles the logic for the button push and updates the end-effector state.
        """
        end_effector_pos = self.get_body_com("end_effector")
        target_pos = self.get_body_com("target")
        relative_distance = np.linalg.norm(end_effector_pos - target_pos)

        self.end_effector_pushed = 1  # End-effector is in the "pushed" state

        # If the end-effector is close enough, mark it as pushed
        if relative_distance < 0.05:
            self.button_state = 1  # Button is switched to "on"
        else:
            self.button_state = 0  # Invalid push, button remains "off"

    def _calculate_reward(self):
        """
        Computes the reward based on the current state of the environment.
        """
        end_effector_pos = self.get_body_com("end_effector")
        target_pos = self.get_body_com("target")
        relative_distance = np.linalg.norm(end_effector_pos - target_pos)

        reward = 0
        
        # 1. Dense negative reward based on the relative distance
        reward += -0.01 * relative_distance
        
        # 2. Positive reward of 1 when the end-effector is close enough to the target
        if relative_distance < 0.05:
            reward += 1
        
        # 3. Penalty for invalid button press
        if self.end_effector_pushed == 1 and relative_distance >= 0.05:
            reward -= 1  # Penalty for invalid push action

        # 4. Large positive reward when the button is successfully pushed
        if self.button_state == 1:
            reward += 100
            done = True
        else:
            done = False

        return reward, done
    
    def reset_model(self):
        # 随机初始化机械臂位置

        qpos = self.init_qpos.copy()  # 复制 init_qpos 避免修改原始数据
        qpos[:3] = self.np_random.uniform(low=-0.1, high=0.1, size=3)  # 更新前 3 个关节的位置       
        # qvel = self.np_random.uniform(low=-0.005, high=0.005, size=3) + self.init_qvel
        qvel = self.init_qvel.copy()  # 复制 init_qpos 避免修改原始数据
        qvel[:3] = self.np_random.uniform(low=-0.1, high=0.1, size=3)  # 更新前 3 个关节的位置       
        
        # 设置目标点————区域要更改一下
        self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=3)
        qpos[-3:] = self.goal
        
        self.set_state(qpos, qvel)
        # Reset torques, button state, and end-effector push state
        self.torques = np.zeros(3)
        self.button_state = 0
        self.end_effector_pushed = 0

        return self._get_obs()

    def _get_obs(self):
        # 获取观察值，包括关节位置、速度和末端执行器与目标的距离
        theta = self.data.qpos.flat[:3]
        angular_velocity = self.data.qvel.flat[:3]  # 获取 3 个关节的速度
        end_effector_pos = self.get_body_com("end_effector")
        target_pos = self.get_body_com("target")

        position_diff = end_effector_pos - target_pos
        
        # Add the end-effector push state to the observation space
        observation = np.concatenate([
            np.cos(theta),              # Cosine of joint angles
            np.sin(theta),              # Sine of joint angles
            target_pos,                 # X and Y coordinates of the target
            end_effector_pos,           # X and Y coordinates of the end-effector
            angular_velocity,           # Angular velocities
            position_diff,              # Difference in positions (x, y, z)
            [self.end_effector_pushed]  # End-effector push state (0 or 1)
        ])
        return observation

    def viewer_setup(self):
        # 设置视图
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0


register(
    id='RoboticArm-v0',
    entry_point='robotic_arm_gym:RoboticArmEnv',  # 替换为定义环境的模块名
)

