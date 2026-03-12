"""Entraînement UR5e avec Acme D4PG."""

from __future__ import annotations

import sys

import numpy as np

from env_ur5e import UR5eReachEnvDM, plot_training_returns


def _make_networks(environment_spec):
    """Construit les réseaux policy/critic D4PG."""
    import haiku as hk
    import jax.numpy as jnp
    from acme.agents.jax.d4pg import networks as d4pg_networks
    from acme.jax import networks as networks_lib
    from acme.jax import utils

    action_spec = environment_spec.actions
    num_actions = int(np.prod(action_spec.shape))
    num_atoms = 51
    atoms = jnp.linspace(-150.0, 150.0, num_atoms)

    def policy(obs):
        return hk.Sequential([
            utils.batch_concat,
            networks_lib.LayerNormMLP((256, 256), activate_final=True),
            networks_lib.NearZeroInitializedLinear(num_actions),
            networks_lib.TanhToSpec(action_spec),
        ])(obs)

    def critic(obs, action):
        value = hk.Sequential([
            utils.batch_concat,
            networks_lib.LayerNormMLP((256, 256, num_atoms)),
        ])([obs, action])
        return value, atoms

    policy_t = hk.without_apply_rng(hk.transform(policy))
    critic_t = hk.without_apply_rng(hk.transform(critic))

    dummy_obs = utils.add_batch_dim(utils.zeros_like(environment_spec.observations))
    dummy_act = utils.add_batch_dim(utils.zeros_like(environment_spec.actions))

    return d4pg_networks.D4PGNetworks(
        policy_network=networks_lib.FeedForwardNetwork(
            init=lambda rng: policy_t.init(rng, dummy_obs),
            apply=policy_t.apply,
        ),
        critic_network=networks_lib.FeedForwardNetwork(
            init=lambda rng: critic_t.init(rng, dummy_obs, dummy_act),
            apply=critic_t.apply,
        ),
    )


def train(num_episodes: int = 500, render: bool = False, seed: int = 0) -> list[float]:
    import acme
    from acme import specs
    from acme.agents.jax.d4pg import builder as d4pg_builder
    from acme.agents.jax.d4pg import config as d4pg_config
    from acme.agents.jax.d4pg import networks as d4pg_networks
    from acme.jax.layouts import local_layout

    environment = UR5eReachEnvDM(render_mode="human" if render else None)
    environment_spec = specs.make_environment_spec(environment)

    networks = _make_networks(environment_spec)
    config = d4pg_config.D4PGConfig(
        batch_size=256,
        min_replay_size=1000,
        max_replay_size=100_000,
        samples_per_insert=4.0,
        samples_per_insert_tolerance_rate=1000.0,
    )
    builder = d4pg_builder.D4PGBuilder(config)
    policy_network = d4pg_networks.get_default_behavior_policy(networks, config)

    agent = local_layout.LocalLayout(
        seed=seed,
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
    returns: list[float] = []

    print(f"Démarrage entraînement: {num_episodes} épisodes")
    print("NOTE: le premier update (JIT) prend 1-5 min — c'est normal.\n")

    for ep in range(num_episodes):
        metrics = loop.run_episode()
        ret = float(metrics.get("episode_return", np.nan))
        length = metrics.get("episode_length", "?")
        returns.append(ret)
        window = min(10, len(returns))
        avg = float(np.nanmean(returns[-window:]))
        print(f"[{ep+1:>4}/{num_episodes}] return={ret:7.2f}  avg{window}={avg:7.2f}  steps={length}")

    environment.close()
    return returns


if __name__ == "__main__":
    import jax
    print(f"Python : {sys.executable}")
    print(f"JAX backend : {jax.default_backend()}  devices : {jax.devices()}\n")

    returns = train(num_episodes=500, render=True)
    plot_training_returns(returns)

    print(f"\nMeilleur return : {np.nanmax(returns):.2f}")
