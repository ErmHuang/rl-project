## Installation
1. install MuJoco 210
        (https://github.com/google-deepmind/mujoco/releases)
   in your main folder ~/
   
        mkdir .mujoco
   
        cd ~/.mujoco/mujoco210/bin
   
        ./simulate

2. create conda environment via environment.yml

        conda env create -f environment.yml
   
   
3. clone the repository
  
        git clone git@github.com:ErmHuang/rl-project.git
   
        cd rl-project

   
4. change the meshdir path in `model/robotic_arm.xml` so that the render works

   
5. activate conda env and run the test

        conda activate me5418_group42
    
        cd YOUR_PROJECT_FOLDER/rl-project

        python gym/test.py

       

## Structure
- gym/: Custom gym environments and test demo.
- meshes/: 3D models stl files for simulations.
- model/: Models used in the reinforcement learning algorithms.

## License
This project is licensed under the MIT License.

## Troubleshooting
### Error 1: 
When running "python gym/test.py", it reports:
"File "/home/user/.local/lib/python3.6/site-packages/gym/envs/mujoco/mujoco_env.py", line 72, in __init__
    assert not done
AssertionError" 

### Solution: 
Try to run "python gym/test.py" several times


