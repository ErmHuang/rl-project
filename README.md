## Installation
1. install MuJoco 210
        (https://github.com/google-deepmind/mujoco/releases)
   in your main folder ~/
   
        mkdir .mujoco
   
        cd ~/.mujoco/mujoco210/bin
   
        ./simulate

3. create conda environment via requirements.txt
   
   
7. clone the repository
  
        git clone git@github.com:ErmHuang/rl-project.git
   
        cd rl-project

   
9. change the mesh path in robotic_arm.xml so that the render works

   
11. activate conda env and run the test
    
        cd YOUR_PROJECT_FOLDER

        python gym/test.py

       

## Structure
- gym/: Custom gym environments.
- meshes/: 3D models for simulations.
- model/: Models used in the reinforcement learning algorithms.
- mujoco-py/: Wrapper for MuJoCo physics engine.

## License
This project is licensed under the MIT License.


