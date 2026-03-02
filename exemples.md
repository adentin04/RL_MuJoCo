``` python
import jax # Library used to accelerate RL research by leveraging GPUs and TPUs.
import acme # Acme is an open-source Python framework for RL.
from acme import specs 
from acme.agents.jax import ddpg
from acme.utils import loggers
from acme.environment_loop import EnvironmentLoop
from dm_control import suite


# 1. Create MuJoCo environment
def make_environment():
    return suite.load(domain_name="pendulum", task_name="swingup")
# The suite organizes environments.
# "pendulum" is the physics model, so this defines what exists in the world.
# For a pendulum, there are different possible tasks:
# - swingup = Swing the pendulum from hanging down to upright
# - balance = Keep it upright

environment = make_environment()
environment_spec = specs.make_environment_spec(environment)
# Creates an environment specification from the simulation environment.

# 2. Build JAX DDPG networks
networks = ddpg.make_networks(
    spec=environment_spec,
    policy_layer_sizes=(256, 256), # Given a state, outputs an action.
    critic_layer_sizes=(256, 256), # Estimates how good an action is in a given state (Q-function).
)
#DDPG = Deep Deterministic Policy Gradient
# This function builds two neural networks.
# Resources: https://dm-acme.readthedocs.io/en/latest/user/components.html
# 3. Create JAX DDPG agent
agent = ddpg.DDPG(
    spec=environment_spec,
    networks=networks,
    batch_size=256,
    learning_rate=1e-3,
)


# 4. Use Acme EnvironmentLoop 
loop = EnvironmentLoop(
    environment,
    agent,
    logger=loggers.TerminalLogger(label="train")
)


# 5. Run training
loop.run(num_episodes=50)
```