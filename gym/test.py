# import mujoco_py
# import os

# # 设置你的 .xml 文件路径
# xml_file_path = os.path.expanduser('/home/erm/.mujoco/mujoco210/model/robotic_arm.xml')

# # 创建 Mujoco 模型
# model = mujoco_py.load_model_from_path(xml_file_path)
# sim = mujoco_py.MjSim(model)

# # 创建渲染器
# viewer = mujoco_py.MjViewer(sim)

# # 运行仿真并渲染
# for _ in range(10000):  # 1000 次仿真步长
#     sim.step()  # 仿真一步
#     viewer.render()  # 渲染仿真环境



import gym
import robotic_arm_gym_v1  # 引入自定义的环境模块
import robotic_arm_gym_v0  # 引入自定义的环境模块

# 创建环境
# env = gym.make('RoboticArm-v0')
env = gym.make('RoboticArm-v1')

# 重置环境，得到初始观察值
observation = env.reset()

count = 0
# 测试环境的运行
for _ in range(10000):  # 模拟1000个时间步
    env.render()  # 渲染环境

    action = env.action_space.sample()  # 随机生成一个动作
    print(f"Action: {action}, Type: {type(action)}")

    observation, reward, done, info = env.step(action)  # 执行动作并得到反馈
    print("observation: ", observation)

    count += 1

    if count % 1000 == 0:
        env.reset()


    if done:  # 检查是否完成
        observation = env.reset()  # 重置环境
env.close()  # 关闭环境




# import gym

# # 创建 Reacher 环境
# env = gym.make('Reacher-v2') 

# # 重置环境，获得初始观测
# observation = env.reset()  # 只获取一个返回值

# # 进行仿真
# for _ in range(1000):  # 设置仿真步数
#     action = env.action_space.sample()  # 随机选择一个动作
#     observation, reward, done, info = env.step(action)  # 施加动作

#     if done:
#         observation = env.reset()  # 如果达到终止条件，则重置环境

#     env.render()  # 渲染环境

# env.close()  # 关闭环境