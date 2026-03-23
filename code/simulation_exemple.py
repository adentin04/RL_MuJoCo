
import acme
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from acme import specs
from acme.agents.jax.d4pg import builder as d4pg_builder
from acme.jax import networks as networks_lib
from acme.jax.layouts import local_layout
from acme.jax import utils
from dm_control import suite


jax.config.update("jax_enable_x64", True)


def make_environment():
    return suite.load(domain_name="pendulum", task_name="swingup")


def make_networks(
    spec: specs.EnvironmentSpec,
    policy_layer_sizes=(300, 200),
    critic_layer_sizes=(400, 300),
    vmin=-150.0,
    vmax=150.0,
    num_atoms=51,
):
    action_spec = spec.actions
    num_dimensions = int(np.prod(action_spec.shape, dtype=int))
    critic_atoms = jnp.linspace(vmin, vmax, num_atoms)

    def actor_fn(obs):
        network = hk.Sequential([
            utils.batch_concat,
            networks_lib.LayerNormMLP(policy_layer_sizes, activate_final=True),
            networks_lib.NearZeroInitializedLinear(num_dimensions),
            networks_lib.TanhToSpec(action_spec),
        ])
        return network(obs)

    def critic_fn(obs, action):
        network = hk.Sequential([
            utils.batch_concat,
            networks_lib.LayerNormMLP([*critic_layer_sizes, num_atoms]),
        ])
        value = network([obs, action])
        return value, critic_atoms

    policy = hk.without_apply_rng(hk.transform(actor_fn))
    critic = hk.without_apply_rng(hk.transform(critic_fn))

    dummy_action = utils.add_batch_dim(utils.zeros_like(spec.actions))
    dummy_obs = utils.add_batch_dim(utils.zeros_like(spec.observations))

    return d4pg_builder.D4PGNetworks(
        policy_network=networks_lib.FeedForwardNetwork(
            init=lambda rng: policy.init(rng, dummy_obs),
            apply=policy.apply,
        ),
        critic_network=networks_lib.FeedForwardNetwork(
            init=lambda rng: critic.init(rng, dummy_obs, dummy_action),
            apply=critic.apply,
        ),
    )


environment = make_environment()
environment_spec = specs.make_environment_spec(environment)

print("=== Environment spec ===")
print("Observation spec:", environment_spec.observations)
print("Action spec:", environment_spec.actions)
print("Reward spec:", environment_spec.rewards)
print("Discount spec:", environment_spec.discounts)

if hasattr(environment_spec.actions, "minimum") and hasattr(environment_spec.actions, "maximum"):
    action_min = np.asarray(environment_spec.actions.minimum)
    action_max = np.asarray(environment_spec.actions.maximum)
    print("Action minimum:", action_min)
    print("Action maximum:", action_max)
    print("Action span:", action_max - action_min)

networks = make_networks(environment_spec)

config = d4pg_builder.D4PGConfig(
    batch_size=32,
    samples_per_insert=2,
    min_replay_size=32,
    samples_per_insert_tolerance_rate=float("inf"),
)

builder = d4pg_builder.D4PGBuilder(config)
policy_network = d4pg_builder.get_default_behavior_policy(networks, config)

agent = local_layout.LocalLayout(
    seed=0,
    environment_spec=environment_spec,
    builder=builder,
    networks=networks,
    policy_network=policy_network,
    min_replay_size=config.min_replay_size,
    samples_per_insert=config.samples_per_insert,
    batch_size=config.batch_size,
    num_sgd_steps_per_step=config.num_sgd_steps_per_step,
)

loop = acme.EnvironmentLoop(environment, agent)

print("\n=== Short training (5 episodes) ===")
print("Check in logs: episode_return and episode_length.")
for episode in range(5):
    metrics = loop.run_episode()
    print(f"Episode {episode + 1} return:", metrics.get("episode_return"))
