import os
from torch import save as tsave
from datetime import datetime
import numpy as np
from tensorboardX import SummaryWriter
import yaml

def setup_train(hypes):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    model_name = hypes['name']
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")
    folder_name = model_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')

    full_path = os.path.join(current_path, folder_name)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
            except FileExistsError:
                pass 
        # save the yaml file
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

    return full_path

class Logger:
    def __init__(
        self,
        save_path="",
    ):
        self.save_path = save_path
        self.writer = SummaryWriter(save_path)
        self.episode = 0
        self.update_step = 0
        self.rewards = []

    def clear(self):
        self.episode = 0

    def log_mean_reward(self, reward, done):
        self.rewards.append(reward)
        if done:
            self.writer.add_scalar("Mean Reward", np.mean(self.rewards), self.episode)
            self.writer.add_scalar("Trajectory_Length", len(self.rewards), self.episode)
            
            self.episode += 1
            self.rewards.clear()

    def save_model(self, model, epoch, pi_optim, vf_optim, pi_scheduler, vf_scheduler, name):
        if not name.endswith(".pth"):
            name += ".pth"
        tsave({"ac" : model.state_dict(),
               "pi_optim" : pi_optim.state_dict(),
               "vf_optim" : vf_optim.state_dict(),
               "pi_scheduler" : pi_scheduler.state_dict(),
               "vf_scheduler" : vf_scheduler.state_dict(),
               "epoch" : epoch}
              , os.path.join(self.save_path, name))

    def log_update_step(self,  kl_loss, entropy, clip_frac, steer_std, gas_std):
        self.writer.add_scalar('kl_loss', kl_loss, self.update_step)
        self.writer.add_scalar('Entropy', entropy, self.update_step)
        self.writer.add_scalar('Clip_frac', clip_frac, self.update_step)
        self.writer.add_scalar("steer_std", steer_std, self.update_step)
        self.writer.add_scalar("gas_std", gas_std, self.update_step)
        self.update_step += 1
