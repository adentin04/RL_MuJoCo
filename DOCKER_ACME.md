# Docker Acme Environment (UR5e)

Ce setup isole Acme/JAX/MuJoCo des conflits locaux Conda/Pip.

## 1) Build

```bash
cd /home/hiwi/Desktop/Mujoco/RL_MuJoCo
docker build -f Dockerfile.acme -t rl-mujoco-acme:latest .
```

## 2) Test imports Acme/JAX/MuJoCo

```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  -w /workspace \
  rl-mujoco-acme:latest \
  python -c "import dm_env, jax, mujoco; from acme import core, types; from acme.agents.jax import sac; print('ACME_DOCKER_OK')"
```

## 3) Lancer simulation.py

```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  -w /workspace \
  rl-mujoco-acme:latest \
  python simulation.py
```

## Option docker compose

```bash
docker compose -f docker-compose.acme.yml up --build
```

## Notes

- Ce conteneur est orienté CPU pour la stabilité Acme.
- Si tu veux la variante GPU Docker ensuite, on peut ajouter un `Dockerfile` CUDA + `--gpus all`.
