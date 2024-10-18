## Installation
1. install MuJoco 210
        (https://github.com/google-deepmind/mujoco/releases)
   in your main folder ~/
   
        mkdir .mujoco
   
        cd ~/.mujoco/mujoco210/bin
   
        ./simulate

2. create conda environment via requirements.txt
   
   
3. clone the repository
  
        git clone git@github.com:ErmHuang/rl-project.git
   
        cd rl-project

   
4. change the mesh path in robotic_arm.xml so that the render works

   
5. activate conda env and run the test
    
        cd YOUR_PROJECT_FOLDER

        python gym/test.py

       

## Structure
- gym/: Custom gym environments and test demo.
- meshes/: 3D models stl files for simulations.
- model/: Models used in the reinforcement learning algorithms.

## License
This project is licensed under the MIT License.

## Troubleshooting
Error 1: "File "/home/erm/.local/lib/python3.6/site-packages/gym/envs/mujoco/mujoco_env.py", line 72, in __init__
    assert not done
AssertionError" 
when running "python gym/test.py"

Solution: Try to run "python gym/test.py" several times


