"""Entraînement UR5e avec Acme D4PG."""
#Provare a cambiare il codice con SAC
from __future__ import annotations

import sys
import os
import ctypes
import time
from pathlib import Path
from types import ModuleType

import numpy as np

from env_ur5e import UR5eReachEnvDM, plot_training_returns


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}][TRAIN] {msg}", flush=True)


def _bootstrap_conda_shared_libs() -> None:
    """Expose les libs conda (dont libpython) pour les extensions natives Acme."""
    _log("bootstrap conda libs: start")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        exe_path = Path(sys.executable).resolve()
        if exe_path.parent.name == "bin" and exe_path.parent.parent.exists():
            conda_prefix = str(exe_path.parent.parent)

    if not conda_prefix:
        _log("bootstrap conda libs: CONDA_PREFIX introuvable, skip")
        return

    lib_dir = Path(conda_prefix) / "lib"
    if not lib_dir.is_dir():
        _log(f"bootstrap conda libs: dossier absent -> {lib_dir}")
        return

    lib_dir_str = str(lib_dir)
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir_str not in ld_library_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = (
            f"{lib_dir_str}:{ld_library_path}" if ld_library_path else lib_dir_str
        )
        _log(f"LD_LIBRARY_PATH mis à jour avec {lib_dir_str}")

    py_major, py_minor = sys.version_info.major, sys.version_info.minor
    candidates = [
        lib_dir / f"libpython{py_major}.{py_minor}.so.1.0",
        lib_dir / f"libpython{py_major}.{py_minor}.so",
    ]

    for candidate in candidates:
        if candidate.exists():
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                _log(f"libpython préchargée: {candidate}")
            except OSError:
                _log(f"échec chargement libpython: {candidate}")
                pass
            break
    _log("bootstrap conda libs: done")


def _install_launchpad_stub() -> None:
    """Installe un stub launchpad minimal pour Acme local_layout."""
    if "launchpad" in sys.modules:
        _log("launchpad stub: déjà présent, skip")
        return

    stub = ModuleType("launchpad")
    stop_handlers = set()

    def register_stop_handler(handler):
        stop_handlers.add(handler)

    def unregister_stop_handler(handler):
        stop_handlers.discard(handler)

    def stop():
        for handler in list(stop_handlers):
            try:
                handler()
            except Exception:
                pass

    class Program:
        def __init__(self, name: str = "agent"):
            self.name = name

    stub.register_stop_handler = register_stop_handler
    stub.unregister_stop_handler = unregister_stop_handler
    stub.stop = stop
    stub.Program = Program
    sys.modules["launchpad"] = stub
    _log("launchpad stub: installé")


def _install_jax_compat_shims() -> None:
    """Ajoute des alias JAX supprimés (Acme 0.4 attend encore ces symboles)."""
    import jax
    import jax.numpy as jnp

    _log("jax compat shims: start")

    array_type = getattr(jax, "Array", None)
    if array_type is None:
        array_type = object

    if not hasattr(jax, "xla"):
        xla_stub = ModuleType("xla")
        device_type = object
        try:
            local_devices = jax.local_devices()
            if local_devices:
                device_type = type(local_devices[0])
        except Exception:
            pass

        xla_stub.Device = device_type
        xla_stub.DeviceArray = array_type
        jax.xla = xla_stub
        _log("jax compat shims: jax.xla ajouté")

    if not hasattr(jax, "pxla"):
        pxla_stub = ModuleType("pxla")
        pxla_stub.ShardedDeviceArray = array_type
        jax.pxla = pxla_stub
        _log("jax compat shims: jax.pxla ajouté")

    if not hasattr(jnp, "DeviceArray"):
        jnp.DeviceArray = array_type
        _log("jax compat shims: jnp.DeviceArray ajouté")
    _log("jax compat shims: done")


def _install_reverb_compat_shim() -> None:
    """Compat Acme/Reverb: ignore kwargs absents sur certaines builds Reverb."""
    import inspect
    import reverb

    _log("reverb compat shim: inspection signature trajectory_writer")

    signature = inspect.signature(reverb.Client.trajectory_writer)
    if "get_signature_timeout_ms" in signature.parameters:
        _log("reverb compat shim: non nécessaire")
        return

    original = reverb.Client.trajectory_writer

    def wrapped(self, *args, **kwargs):
        kwargs.pop("get_signature_timeout_ms", None)
        return original(self, *args, **kwargs)

    reverb.Client.trajectory_writer = wrapped
    _log("reverb compat shim: installé")


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
    _log(
        f"build networks: action_shape={action_spec.shape}, num_actions={num_actions}, num_atoms={num_atoms}"
    )

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


<<<<<<< HEAD
def train(num_episodes: int = 10000, render: bool = False, seed: int = 0) -> list[float]:
=======
def train(
    num_episodes: int =5000,
    render: bool = False,
    seed: int = 0,
    render_every_n_steps: int = 1,
    max_episode_steps: int = 0,
) -> list[float]:
    _log(
        f"train start: episodes={num_episodes}, render={render}, seed={seed}, "
        f"render_every_n_steps={render_every_n_steps}, max_episode_steps={max_episode_steps}"
    )
    _bootstrap_conda_shared_libs()
    _install_launchpad_stub()
    _install_jax_compat_shims()
    _install_reverb_compat_shim()

>>>>>>> 617ff733cfc30768a8fd143622f20656fdcd3a5e
    import acme
    from acme import specs
    from acme.agents.jax.d4pg import builder as d4pg_builder
    from acme.agents.jax.d4pg import config as d4pg_config
    from acme.agents.jax.d4pg import networks as d4pg_networks
    from acme.jax.layouts import local_layout

    environment = UR5eReachEnvDM(render_mode="human" if render else None)
    environment_spec = specs.make_environment_spec(environment)
    _log(f"environment spec observations={environment_spec.observations}")
    _log(f"environment spec actions={environment_spec.actions}")

    networks = _make_networks(environment_spec)
    config = d4pg_config.D4PGConfig(
        batch_size=256,
        min_replay_size=1000,
        max_replay_size=100_000,
        samples_per_insert=4.0,
        samples_per_insert_tolerance_rate=1000.0,
    )
    _log(
        "d4pg config: "
        f"batch_size={config.batch_size}, min_replay_size={config.min_replay_size}, "
        f"max_replay_size={config.max_replay_size}, samples_per_insert={config.samples_per_insert}"
    )
    builder = d4pg_builder.D4PGBuilder(config)
    policy_network = d4pg_networks.get_default_behavior_policy(networks, config)
    _log("builder et behavior policy prêts")

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
    _log("agent LocalLayout initialisé")

    step_counter = {"count": 0}
    original_reset = environment.reset
    original_step = environment.step

    def reset_with_render():
        step_counter["count"] = 0
        timestep = original_reset()
        if render:
            environment.render()
        return timestep

    def step_with_render(action):
        timestep = original_step(action)
        step_counter["count"] += 1

        if render and render_every_n_steps > 0 and step_counter["count"] % render_every_n_steps == 0:
            environment.render()

        if max_episode_steps > 0 and step_counter["count"] >= max_episode_steps and not timestep.last():
            import dm_env

            _log(
                f"episode forcé en truncation à step={step_counter['count']} "
                f"(max_episode_steps={max_episode_steps})"
            )
            return dm_env.truncation(
                reward=float(timestep.reward or 0.0),
                observation=timestep.observation,
            )

        return timestep

    environment.reset = reset_with_render
    environment.step = step_with_render
    if render:
        _log("viewer MuJoCo activé (render via reset/step wrappers)")

    loop = acme.EnvironmentLoop(environment, agent)
    returns: list[float] = []

    print(f"Démarrage entraînement: {num_episodes} épisodes")
    print("NOTE: le premier update (JIT) prend 1-5 min — c'est normal.\n")

    for ep in range(num_episodes):
        ep_start = time.time()
        _log(f"episode {ep+1}/{num_episodes}: start")
        metrics = loop.run_episode()
        _log(f"episode {ep+1}/{num_episodes}: raw metrics keys={list(metrics.keys())}")
        ret = float(metrics.get("episode_return", np.nan))
        length = metrics.get("episode_length", "?")
        returns.append(ret)
        window = min(10, len(returns))
        avg = float(np.nanmean(returns[-window:]))
        print(f"[{ep+1:>4}/{num_episodes}] return={ret:7.2f}  avg{window}={avg:7.2f}  steps={length}")
        _log(
            f"episode {ep+1}/{num_episodes}: done in {time.time() - ep_start:.2f}s, "
            f"return={ret:.3f}, rolling_avg={avg:.3f}, length={length}"
        )

    environment.close()
    _log("environment fermé")
    _log("train done")
    return returns


if __name__ == "__main__":
    import jax
    _log("entrypoint train_acme.py")
    print(f"Python : {sys.executable}")
    print(f"JAX backend : {jax.default_backend()}  devices : {jax.devices()}\n")

    returns = train(num_episodes=10000, render=True)
    plot_training_returns(returns)

    print(f"\nMeilleur return : {np.nanmax(returns):.2f}")
