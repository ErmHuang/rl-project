## Installation
1. install MuJoco 210
   
        pip install gym[mujoco]
   or
        https://github.com/google-deepmind/mujoco/releases
   in your main folder ~/
        mkdir .mujoco
        cd ~/.mujoco/mujoco210/bin
        ./simulate

3. create conda environment via requirements.txt

4. put the robotic_arm.xml  into your gym(mujoco) assets folder so that it can be found by (MujocoEnv.__init__(self, "robotic_arm-v1.xml", 2, **kwargs) )
     for reference , my path to the assets folder is :  /home/mayuxuan/.local/lib/python3.6/site-packages/gym/envs/mujoco/assets
5. Clone the repository
   git clone git@github.com:ErmHuang/rl-project.git
   cd rl-project
6. change the mesh path in robotic_arm.xml so that the render works
7. run test
   cd your_project_folder
   python test.py

       

## Structure
- gym/: Custom gym environments.
- meshes/: 3D models for simulations.
- model/: Models used in the reinforcement learning algorithms.
- mujoco-py/: Wrapper for MuJoCo physics engine.

## License
This project is licensed under the MIT License.


