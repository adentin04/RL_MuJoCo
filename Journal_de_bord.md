# 28/02/26

Étapes pour installer les outils :

- Créer un environnement :
    conda create -n mujoco_env python=3.10 -y
- Installer Miniconda.
- Installer `mujoco-python` :
    conda install conda-forge::mujoco-python
- Installer `dm-acme` :
    pip install dm-acme
- Installer le backend RL :
    pip install "dm-acme[jax]"

# 02/03/26
- git clone https://github.com/adentin04/RL_MuJoCo.git

<- Push de ce que j'ai fait jeudi et vendredi ->

- Objectives for today:
    - Research about the MuJoCo API (**finished**) (`research/Mujoco_API`)
    - Research about MuJoCo Playground (**finished**) (`research/Mujoco_playground`)
    - Understand the logic in the RL code (`dm-acme` (JAX)) 

- I analyzed `examples.md` to better understand the code and added some comments. 

<- J'ai push ->

- J'ai fait un pas en arrière et j'ai commencé à analyser tout ce qui est RL et deepmind sans aller sous les outils plus lourds car je savais que je n'allais rien comprendre , j'ai commencé par faire étape par étape et j'ai détaillé tout mon code pour comprendre ce que je faisais et la dynamique du code ! Mon travail est dans "Gym" .