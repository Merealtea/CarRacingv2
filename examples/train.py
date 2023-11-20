"""A basic example showing how to train a DQN agent to play Breakout from pixel information"""
import gymnasium as gym
import numpy as np
import os
import torch


from src.algorithms.ppo import PPO
from src.models.ActorCritic import CarRaceActorCritic
# from src.algorithms.double_deep_q_learning import DoubleDQNAtariAgent
# from src.models import DDQN
# from src.utils.assessment import AtariEvaluator
# from src.utils.env import DiscreteCarRacing, wrap_deepmind, wrap_box2d
from src.utils.logger import setup_train
import yaml

with open("./config/ppo.yaml") as f:
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


if hypes["save_path"] == None:
    save_path = setup_train(hypes)
else:
    save_path = hypes["save_path"] 
    # with open(os.path.join(save_path, "config.yaml")) as f:
    #     hypes = yaml.safe_load(f)
    
ppo_kwargs = hypes["ppo_kwargs"]
ppo_kwargs["ac_kwargs"] = hypes["ac_kwargs"]
ppo_kwargs["env"] = env
ppo_kwargs["actor_critic"] = CarRaceActorCritic
ppo_kwargs["save_path"] = save_path
ppo_kwargs["device"] = device

print(save_path)

ppo_trainer = PPO(**ppo_kwargs)

ppo_trainer.train()