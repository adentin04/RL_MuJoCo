``` python
import jax
import acme
from acme import specs
from acme.agents.jax import ddpg
from acme.utils import loggers
from acme.environment_loop import EnvironmentLoop
from dm_control import suite


# 1. Create MuJoCo environment
def make_environment():
    return suite.load(domain_name="pendulum", task_name="swingup")


environment = make_environment()
environment_spec = specs.make_environment_spec(environment)


# 2. Build JAX DDPG networks
networks = ddpg.make_networks(
    spec=environment_spec,
    policy_layer_sizes=(256, 256),
    critic_layer_sizes=(256, 256),
)


# 3. Create JAX DDPG agent
agent = ddpg.DDPG(
    spec=environment_spec,
    networks=networks,
    batch_size=256,
    learning_rate=1e-3,
)


# 4. Use Acme EnvironmentLoop (cleaner than manual loop)
loop = EnvironmentLoop(
    environment,
    agent,
    logger=loggers.TerminalLogger(label="train")
)


# 5. Run training
loop.run(num_episodes=50)
```