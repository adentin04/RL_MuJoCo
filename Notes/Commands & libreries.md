``` bash

Conda install conda-forge::mujoco-python
Sudo apt-get install libglfw3 libglew2.0 libgl1-mesa-glx libosmesa6
Conda activate mujoco_env 
Conda install -c conda-forge glew
Conda install -c conda-forge mesalib
Conda install -c anaconda mesa-libgl-cos6-x86_64
Conda install -c menpoglfw3 
Conda create -n mujoco_env python=3.9
Conda activate mujoco_env
Pip install mujoco
Xvfb
xvfb-run -s "-screen 0 128x729x24" python "file.py"
sudo add-apt-repository universe
```
``` python

Import importlib sys pkgutil 
getattr()
importlib.util.find_spec())
PKGUTILfind_loader())
python3 pip index versions exemple
Install : sofrtware-properties-common-y

env=suite.load(domain-name,task_name) 
Reset: timestep = env.reset() -> timestep.observation(dict),
timestemp.reward(float or None), 
timestemp.last(bool)
Step: timestep = env.step(action) 
action doit respecter env.action_spec() (shape,min,max)

Observation -> vecteur: obs = np.concatenante([r.rave(C) for r in timestep.observation.value()])
Action continues : souvent un vecteur flottant; discret et rare dans dm_control 
Physics : env.physics donne accès direct à qpos, qvel, et env.physics.render(...) pour images

```
## Pour Reinforcement learning

- Deepmind Control
- OpenAI Gym,
- stable Baselines3
