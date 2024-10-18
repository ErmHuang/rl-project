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

   
4. Modify the meshdir path in `model/robotic_arm.xml` to point to your own directory so that the render works, such as

        <meshdir="$HOME/YOUR_PROJECT_FOLDER/rl-project/meshes/"/>
   
5. Activate conda env and run the test
    
        cd YOUR_PROJECT_FOLDER/rl-project

        conda activate me5418_group42

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


