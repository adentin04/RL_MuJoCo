"""UR5e Reach avec Acme + DeepMind dm_env + MuJoCo."""

from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import List

import numpy as np

from simulation import UR5eReachEnvDM

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # évite OOM GPU sur RTX 2060 6GB
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jax_xla_cache")  # cache XLA → compile 1 seule fois


def plot_training_returns(returns: List[float], output_path: str | None = None) -> None:
    """Trace une courbe d'apprentissage style learning_curves et sauvegarde un PNG."""
    try:
        import matplotlib
        if not os.environ.get("DISPLAY"):
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[PLOT] matplotlib non disponible: {exc}", flush=True)
        return

    y = np.asarray(returns, dtype=np.float32)
    if y.size == 0:
        print("[PLOT] Aucun return à tracer.", flush=True)
        return

    x = np.arange(y.size)
    window = 20 if y.size >= 20 else max(3, y.size)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    moving_avg = np.convolve(y, kernel, mode="valid")
    moving_x = np.arange(window - 1, y.size)

    running_best = np.maximum.accumulate(y)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, y, color="steelblue", alpha=0.30, linewidth=0.9, label="Return par épisode")
    ax.plot(moving_x, moving_avg, color="steelblue", linewidth=2.2, label=f"Moyenne mobile ({window} épisodes)")
    ax.plot(x, running_best, color="seagreen", linestyle="--", linewidth=1.2, alpha=0.9, label="Meilleur score cumulé")

    ax.set_title("Courbe d'apprentissage — UR5e Reach (Acme D4PG)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Épisode", fontsize=12)
    ax.set_ylabel("Return", fontsize=12)
    ax.grid(True, alpha=0.30)
    ax.legend(fontsize=10)

    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
        margin = 0.08 * (y_max - y_min)
        ax.set_ylim(y_min - margin, y_max + margin)

    fig.tight_layout()

    output_dir = Path(__file__).resolve().parent / "learning_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"learning_curve_{timestamp}.png")

    plt.savefig(output_path, dpi=140)

    latest_path = output_dir / "learning_curve.png"
    if str(latest_path) != output_path:
        plt.savefig(str(latest_path), dpi=140)

    backend = str(getattr(plt, "get_backend", lambda: "")()).lower()
    if "agg" in backend:
        print(f"[PLOT] Environnement sans affichage. Graphiques sauvegardés:", flush=True)
        print(f"       - {output_path}", flush=True)
        print(f"       - {latest_path}", flush=True)
    else:
        plt.show(block=False)
        print(f"[PLOT] Graphiques sauvegardés: {output_path} | {latest_path}", flush=True)
        plt.show()
    plt.close()


def _bootstrap_conda_shared_libs() -> None:
    """Expose les libs conda (dont libpython) pour les extensions natives Acme."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        exe_path = Path(sys.executable).resolve()
        if exe_path.parent.name == "bin" and exe_path.parent.parent.exists():
            conda_prefix = str(exe_path.parent.parent)

    if not conda_prefix:
        return

    lib_dir = Path(conda_prefix) / "lib"
    if not lib_dir.is_dir():
        return

    lib_dir_str = str(lib_dir)
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if lib_dir_str not in ld_library_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = (
            f"{lib_dir_str}:{ld_library_path}" if ld_library_path else lib_dir_str
        )

    py_major, py_minor = sys.version_info.major, sys.version_info.minor
    candidates = [
        lib_dir / f"libpython{py_major}.{py_minor}.so.1.0",
        lib_dir / f"libpython{py_major}.{py_minor}.so",
    ]

    for candidate in candidates:
        if candidate.exists():
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
            break


def _install_launchpad_stub() -> None:
    """Installe un stub launchpad minimal pour les usages Acme en local.

    Acme importe `launchpad` via `acme.utils.signals`, mais en mode local
    (`local_layout`) on n'utilise pas l'infra distribuée Courier/Launchpad.
    """
    if "launchpad" in sys.modules:
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


def _install_jax_compat_shims() -> None:
    """Ajoute des alias JAX supprimés (Acme 0.4 attend encore ces symboles)."""
    import jax
    import jax.numpy as jnp

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

    if not hasattr(jax, "pxla"):
        pxla_stub = ModuleType("pxla")
        pxla_stub.ShardedDeviceArray = array_type
        jax.pxla = pxla_stub

    if not hasattr(jnp, "DeviceArray"):
        jnp.DeviceArray = array_type


def _install_reverb_compat_shim() -> None:
    """Compat Acme/Reverb: ignore kwargs absents sur anciennes builds Reverb."""
    import inspect
    try:
        import reverb
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Module `reverb` introuvable dans cet interpréteur Python. "
            f"Python actif: {sys.executable}. "
            "Tu as probablement `.venv` et `acmeUr5e` actifs en même temps. "
            "Lance avec: /home/hiwi/miniconda3/envs/acmeUr5e/bin/python simulation_acme.py"
        ) from exc

    signature = inspect.signature(reverb.Client.trajectory_writer)
    if "get_signature_timeout_ms" in signature.parameters:
        return

    original = reverb.Client.trajectory_writer

    def wrapped(self, *args, **kwargs):
        kwargs.pop("get_signature_timeout_ms", None)
        return original(self, *args, **kwargs)

    reverb.Client.trajectory_writer = wrapped


def _make_acme_networks(environment_spec):
    """Construit policy/critic D4PG pour l'environnement dm_env."""
    import haiku as hk
    import jax.numpy as jnp
    from acme.agents.jax.d4pg import builder as d4pg_builder
    from acme.agents.jax.d4pg import networks as d4pg_networks
    from acme.jax import networks as networks_lib
    from acme.jax import utils

    action_spec = environment_spec.actions
    num_dimensions = int(np.prod(action_spec.shape, dtype=int))

    vmin = -150.0
    vmax = 150.0
    num_atoms = 51
    critic_atoms = jnp.linspace(vmin, vmax, num_atoms)

    def actor_fn(obs):
        network = hk.Sequential([
            utils.batch_concat,
            networks_lib.LayerNormMLP((256, 256), activate_final=True),
            networks_lib.NearZeroInitializedLinear(num_dimensions),
            networks_lib.TanhToSpec(action_spec),
        ])
        return network(obs)

    def critic_fn(obs, action):
        network = hk.Sequential([
            utils.batch_concat,
            networks_lib.LayerNormMLP((256, 256, num_atoms)),
        ])
        value = network([obs, action])
        return value, critic_atoms

    policy = hk.without_apply_rng(hk.transform(actor_fn))
    critic = hk.without_apply_rng(hk.transform(critic_fn))

    dummy_action = utils.add_batch_dim(utils.zeros_like(environment_spec.actions))
    dummy_obs = utils.add_batch_dim(utils.zeros_like(environment_spec.observations))

    return d4pg_networks.D4PGNetworks(
        policy_network=networks_lib.FeedForwardNetwork(
            init=lambda rng: policy.init(rng, dummy_obs),
            apply=policy.apply,
        ),
        critic_network=networks_lib.FeedForwardNetwork(
            init=lambda rng: critic.init(rng, dummy_obs, dummy_action),
            apply=critic.apply,
        ),
    )


def build_acme_agent(environment: UR5eReachEnvDM, seed: int = 0):
    """Crée un agent Acme local (sans Launchpad distribué)."""
    print("[INIT] Préparation des compatibilités runtime (conda/launchpad/jax/reverb)...", flush=True)
    _bootstrap_conda_shared_libs()
    _install_launchpad_stub()
    _install_jax_compat_shims()
    _install_reverb_compat_shim()

    import jax as _jax
    _backend = _jax.default_backend()
    _devices = _jax.devices()
    print(f"[INIT] JAX backend: {_backend} | devices: {_devices}", flush=True)
    if _backend != "gpu":
        print("[WARN] JAX tourne sur CPU — entraînement sera très lent!", flush=True)

    import acme
    from acme import specs
    from acme.agents.jax.d4pg import builder as d4pg_builder
    from acme.agents.jax.d4pg import config as d4pg_config
    from acme.agents.jax.d4pg import networks as d4pg_networks
    from acme.jax.layouts import local_layout

    environment_spec = specs.make_environment_spec(environment)
    print(f"[INIT] Observation spec: {environment_spec.observations}", flush=True)
    print(f"[INIT] Action spec: {environment_spec.actions}", flush=True)
    networks = _make_acme_networks(environment_spec)

    config = d4pg_config.D4PGConfig(
        batch_size=64,
        samples_per_insert=1.0,
        min_replay_size=100,
        max_replay_size=100000,
        samples_per_insert_tolerance_rate=1000.0,  # large tolerance avoids reverb deadlock
    )
    print(
        "[INIT] Config D4PG: "
        f"batch_size={config.batch_size}, min_replay_size={config.min_replay_size}, "
        f"max_replay_size={config.max_replay_size}",
        flush=True,
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

    print("[INIT] Agent Acme local initialisé.", flush=True)
    print(
        f"[INIT] NOTE: après {config.min_replay_size} steps, JAX va JIT-compiler le learner "
        "(1-5 min la première fois). C'est normal — attendez sans interrompre.",
        flush=True,
    )

    return acme, agent


def train_ur5e_with_acme(
    num_episodes: int=4000,
    render: bool = True,
    render_every_n_steps: int = 1,
    max_episode_steps: int = 10000,
) -> List[float]:
    """Entraîne UR5e avec Acme (D4PG) sur l'environnement dm_env."""
    print(
            f"[TRAIN] Démarrage entraînement: num_episodes={num_episodes}, render={render}, "
            f"render_every_n_steps={render_every_n_steps}, max_episode_steps={max_episode_steps}",
            flush=True,
        )
    environment = UR5eReachEnvDM(render_mode="human" if render else None)

    try:
        acme, agent = build_acme_agent(environment)
    except Exception as exc:
        environment.close()
        message = (
            "Impossible d'initialiser Acme/JAX dans l'environnement courant "
            "(acmeUr5e). Vérifie acme, dm-env, jax, dm-haiku, reverb, tensorflow. "
            f"Cause: {type(exc).__name__}: {exc}"
        )
        if isinstance(exc, ValueError) and "numpy.dtype size changed" in str(exc):
            message += (
                "\n[FIX ABI NumPy] Dans l'env acmeUr5e: "
                "python -m pip uninstall -y numpy && python -m pip install numpy==1.26.4"
            )
        raise RuntimeError(
            message
        ) from exc

    loop = acme.EnvironmentLoop(environment, agent)
    step_counter = {"count": 0}
    original_reset = environment.reset
    original_step = environment.step

    def reset_with_counter():
            step_counter["count"] = 0
            timestep = original_reset()
            if render:
                environment.render()
            return timestep

    def step_with_controls(action):
            timestep = original_step(action)
            step_counter["count"] += 1

            if render and render_every_n_steps > 0 and step_counter["count"] % render_every_n_steps == 0:
                environment.render()

            if max_episode_steps > 0 and step_counter["count"] >= max_episode_steps and not timestep.last():
                import dm_env
                return dm_env.truncation(reward=float(timestep.reward or 0.0), observation=timestep.observation)

            return timestep

    environment.reset = reset_with_counter
    environment.step = step_with_controls

    loop = acme.EnvironmentLoop(environment, agent)
    returns: List[float] = []

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes} en cours...", flush=True)

        stop_heartbeat = threading.Event()

        def heartbeat():
            tick = 0
            while not stop_heartbeat.wait(2.0):
                tick += 1
                print(
                    f"  ... épisode {episode + 1} toujours en cours ({2 * tick}s, steps={step_counter['count']})",
                    flush=True,
                )

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()

        metrics = loop.run_episode()
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=0.1)

        episode_return = float(metrics.get("episode_return", np.nan))
        episode_length = metrics.get("episode_length", "n/a")
        actor_steps = metrics.get("actor_steps", "n/a")
        returns.append(episode_return)
        window = 10 if len(returns) >= 10 else len(returns)
        avg = float(np.nanmean(np.asarray(returns[-window:], dtype=np.float32)))
        print(
            f"Episode {episode + 1}/{num_episodes}, Return: {episode_return:.2f}, "
            f"Moyenne({window}): {avg:.2f}, Longueur: {episode_length}, Actor steps: {actor_steps}",
            flush=True,
        )

        if isinstance(metrics, dict):
            other_metrics = {
                key: value for key, value in metrics.items()
                if key not in {"episode_return", "episode_length", "actor_steps"}
            }
            if other_metrics:
                print(f"  [METRICS] {other_metrics}", flush=True)

    environment.close()
    print("[TRAIN] Entraînement terminé.", flush=True)
    return returns


if __name__ == "__main__":
    print("=" * 60)
    print("UR5e REACH AVEC ACME + DEEPMIND DM_ENV + MUJOCO")
    print("=" * 60)
    print("Agent: Acme D4PG (JAX)")
    print("Environment API: dm_env")
    print("Simulateur: MuJoCo")
    print(f"Python: {sys.executable}")
    print("=" * 60)

    returns: List[float] = []
    try:
        returns = train_ur5e_with_acme(
            num_episodes=4000,
            render=True,
            render_every_n_steps=1,
            max_episode_steps=100,
        )
    except KeyboardInterrupt:
        print("\n[INFO] Entraînement interrompu par l'utilisateur.")
    finally:
        if returns:
            print("[PLOT] Affichage du graphique des returns...", flush=True)
            plot_training_returns(returns)

            print("\nEntraînement Acme terminé!")
            print(f"Meilleur return: {np.nanmax(np.asarray(returns, dtype=np.float32)):.2f}")
        else:
            print("[PLOT] Aucun return disponible pour tracer un graphique.", flush=True)
