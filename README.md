## Installation
1. Install MuJoco 210
        (https://github.com/google-deepmind/mujoco/releases)
   in your main folder ~/
   
        mkdir .mujoco
   
        cd ~/.mujoco/mujoco210/bin
   
        ./simulate

2. Create conda environment via environment.yml

        conda env create -f environment.yml
   
   
3. Clone the repository
  
        git clone git@github.com:ErmHuang/rl-project.git
   
        cd rl-project

   
5. Activate conda env and run the test
    
        cd YOUR_PROJECT_FOLDER/rl-project

        conda activate me5418_group42

        python scripts/gym_test.py

6. Train your model

        python scripts/learning_agent.py

if youn want to use your pre-trained model for continuouse training, change pretrained_model_path

7. Test your model

        python scripts/play.py
       

## Structure
- scripts/: python files.
- meshes/: 3D models stl files for simulations.
- model/: Models used in the reinforcement learning algorithms.
-model: 3-DOF Robot arm configurations for simulation:
        – meshes: STL files of every links for the robot arm
        – manipulator.urdf: URDF file of the robot arm
        – robot arm.xml: Parameters set-up of the robot arm
- scripts: Codes
        – robotic arm gym v1.py: gym environment set-up
        – gym test.py: test demo of gym environment for random action execution
        – network.py: Actor-Critic network frameworks
        – ppo.py: PPO algorithm implementation
        – rollout storage v1.py: storage and transition data management
        – learning agent.py: model training script
        – play.py: model testing script
        – logger.py: code for visualization
- logs: Trained models
- environment.yml: requirement for building virtual environment

## License
This project is licensed under the MIT License.




