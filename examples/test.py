"""A basic example showing how to train a DQN agent to play Breakout from pixel information"""
import gymnasium as gym
import numpy as np
import os
import torch


from src.algorithms.ppo import PPO
from src.models.ActorCritic import CarRaceActorCritic
from src.utils.logger import setup_train
from os.path import join
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", help="path to direction with model pth")
args = parser.parse_args()

model_path = args.model_path

with open(join(model_path, "config.yaml")) as f:
    hypes = yaml.safe_load(f)

# ------Env------------------
name = "CarRacing-v2"
save_path = os.path.join("results", "models", name)
env = gym.make(
    name, verbose=0,
    render_mode = "human"
)  # Verbosity off for CarRacing - track generation info can get annoying!

if "CarRacing" in name:
    # DQN needs discrete inputs
    env = env
else:
    env = wrap_deepmind(env, episode_life=False)


n_actions = env.action_space.shape[0]

# -------Models--------------
GPU_NUM = hypes["GPU_NUM"]
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) 
    
ppo_kwargs = hypes["ppo_kwargs"]
ppo_kwargs["ac_kwargs"] = hypes["ac_kwargs"]
ppo_kwargs["env"] = env
ppo_kwargs["actor_critic"] = CarRaceActorCritic
ppo_kwargs["save_path"] = model_path
ppo_kwargs["device"] = device
ppo_trainer = PPO(**ppo_kwargs, train=False)