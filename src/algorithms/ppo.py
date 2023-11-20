import numpy as np
import torch
from torch.optim import Adam
import scipy
import time
from tqdm import tqdm
from src.utils.logger import Logger
import cv2
from os.path import join
import math
from src.utils.env import Wrapped_Env, FireResetEnv
from torch.optim.lr_scheduler import StepLR
import os

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x: An array containing samples of the scalar to produce statistics
            for.

        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = np.sum(x), len(x)
    mean = global_sum / global_n

    global_sum_sq = np.sum((x - mean)**2)
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = np.min(x) if len(x) > 0 else np.inf
        global_max = np.max(x) if len(x) > 0 else -np.inf
        return mean, std, global_min, global_max
    return mean, std

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, img_h, img_w, history_len, act_dim, size, gamma=0.99, lam=0.95, data_augument_rate = 0):
        self.obs_buf = np.zeros((size, 3 * history_len, img_h, img_w), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.data_augument_rate = data_augument_rate
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        
        if self.data_augument_rate > 0:
            if np.random.choice([0, 1], size=1, p=[1-self.data_augument_rate, self.data_augument_rate]):
                obs, act = self.data_augument(obs, act)
        
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1


    def data_augument(self, obs, act):
        """
            Flip the obs and act from left to right
        """

        obs = np.flip(obs, axis = 3)
        # cv2.imwrite("./test.png", np.uint8(obs[0,:3] *255).transpose(1,2,0))
        act = act * np.array([-1, 1])
        return obs, act

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, val = self.val_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

def to_device(data, device):
    if isinstance(data, dict):
        for key in data:
            data[key] = to_device(data[key], device)
        return data
    if isinstance(data, torch.Tensor):
        return data.to(device)

class PPO():
    def __init__(self, env,actor_critic, history_len, device , ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, save_path = None, schedule_step_size = 5, decay_rate = 0.5 , mini_batchsize = 32, adv_norm = False,
        lr_decay = False, ent_coef = 0, vf_coef = 0, pi_coef = 0, clip_value  = False, data_augument_rate = 0):

        # Random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed

        self.env = Wrapped_Env(env, history_len)
        # self.env.seed(seed)

        # Instantiate environment
        img_h, img_w = ac_kwargs["img_h"], ac_kwargs["img_w"]
        act_dim = self.env.action_space.shape[0] - 1
        self.device = device

        # Create actor-critic module
        self.ac = actor_critic(**ac_kwargs)
        self.ac.to(device)
        self.ac.eval()

        # Create data buffer
        self.buf = PPOBuffer(img_h, img_w, self.env.history_len, act_dim, steps_per_epoch, gamma, lam, data_augument_rate)

        # Training parameters
        self.start_epoch = 0
        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.target_kl = target_kl
        self.train_v_iters = train_v_iters
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.max_ep_len = max_ep_len
        self.adv_norm = adv_norm
        self.batch_szie = mini_batchsize
        self.lr_decay = lr_decay
        self.clip_value = clip_value
        # Train loss coefficient 
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.pi_coef = pi_coef

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.actor.parameters(), lr=pi_lr, eps= 1e-5)
        self.vf_optimizer = Adam(self.ac.critic.parameters(), lr=vf_lr, eps= 1e-5)
        
        self.pi_scheduler = StepLR(self.pi_optimizer, step_size=schedule_step_size, gamma=decay_rate)
        self.vf_scheduler = StepLR(self.vf_optimizer, step_size=schedule_step_size, gamma=decay_rate)

        # Logger
        self.logger = Logger(save_path)
        self.save_path = save_path

        # Load model
        if "best_model.pth" in os.listdir(save_path):
            ckpt = torch.load(join(save_path, "best_model.pth"))
            self.start_epoch = ckpt["epoch"]
            print("Load Previous ckpt and training from epoch {}".format(self.start_epoch))
            
            self.ac.load_state_dict(ckpt["ac"])
            self.pi_optimizer.load_state_dict(ckpt["pi_optim"])
            self.vf_optimizer.load_state_dict(ckpt["vf_optim"])
            self.pi_scheduler.load_state_dict(ckpt["pi_scheduler"])
            self.vf_scheduler.load_state_dict(ckpt["vf_scheduler"])


    # Set up function for computing PPO policy loss, Clip PPO
    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        
        if self.adv_norm:
            adv = (adv - torch.mean(adv)) / torch.std(adv)

        # Policy loss
        pi, logp = self.ac.actor(obs, act)
        std = self.ac.actor.log_std.detach().cpu().numpy()
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac, steer_std = std[0], gas_std = std[1])

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(self,  data):
        obs, ret = data['obs'], data['ret']
        v_loss_unclipped = (self.ac.critic(obs) - ret)**2
        if self.clip_value :
            v_clipped = data["val"] + torch.clamp(
                    self.ac.critic(obs) - data["val"],
                    -self.clip_ratio,
                    self.clip_ratio,
                )
            v_loss_clipped = (v_clipped - ret) ** 2
            v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
        else:
            v_loss = v_loss_unclipped.mean()
        return v_loss
    
    def get_batch_data(self, data, idxs):
        batch_data = {k:data[k][idxs] for k in data}
        return batch_data

    def update(self):
        self.ac.train()
        data = self.buf.get()

        print("Train V")
        # Value function learning
        for i in range(self.train_v_iters):
            idxs = np.arange(len(data['obs']))
            np.random.shuffle(idxs)
            for j in range(len(data['obs'])//self.batch_szie):
                self.vf_optimizer.zero_grad()
                self.pi_optimizer.zero_grad()

                idx_batch = idxs[j*self.batch_szie : (j+1) * self.batch_szie]
                data_batch = self.get_batch_data(data, idx_batch)
                data_batch = to_device(data_batch, self.device)
                
                loss_v = self.compute_loss_v(data_batch)
                loss_pi, pi_info = self.compute_loss_pi(data_batch)

                kl = pi_info['kl']
                
                loss = self.ent_coef * pi_info['ent'] + \
                        self.pi_coef * loss_pi + \
                        self.vf_coef * loss_v   

                # loss_v.backward()
                # loss_pi.backward()         
                loss.backward()      
                torch.nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 0.5) 
                torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 0.5) 
                
                self.vf_optimizer.step()
                self.pi_optimizer.step()

                # Log changes from update
                kl, ent, cf, steer_std, gas_steer = pi_info['kl'], pi_info['ent'].item(), pi_info['cf'], pi_info["steer_std"], pi_info["gas_std"]
                self.logger.log_update_step(kl, ent, cf, steer_std, gas_steer)

                # if kl > 1.5 * self.target_kl:
                #     print("Update stop in {} iteration due to kl is {}".format(i, kl))
                #     break

        self.ac.eval()
        

    def train(self):
        # Prepare for interaction with environment
        o, ep_ret, ep_len = self.env.reset( ), 0, 0

        self.env.render()
        all_ret = 0
        max_ret = -100

        # Main loop: collect experience in env and update/log each epoch
        num_step = 0
        for epoch in range(self.start_epoch, self.epochs):
            print("Sample data for epoch {}".format(epoch))
            for t in tqdm(range(self.steps_per_epoch)):
                with torch.no_grad():
                    num_step += 1
                    a, ori_a, v, logp = self.ac.step(torch.as_tensor(o, dtype=torch.float32).to(self.device))
                    
                    next_o, r, d, _, _ = self.env.step(a)
                    ep_ret += r
                    ep_len += 1
                    all_ret += r

                    timeout = ep_len == self.max_ep_len
                    if timeout:
                        print("Time out for current trajectory")
                    
                    # save to buffer
                    self.buf.store(o, ori_a, r, v, logp)
                    
                    # Update obs (critical!)
                    o = next_o
                    
                    terminal = d or timeout
                    epoch_ended = t==self.steps_per_epoch-1
                    self.logger.log_mean_reward(r, timeout or terminal or epoch_ended)

                    if terminal or epoch_ended:
                        if epoch_ended and not(terminal):
                            print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                        # if trajectory didn't reach terminal state, bootstrap value target
                        if timeout or epoch_ended:
                            _, _, v, _ = self.ac.step(torch.as_tensor(o, dtype=torch.float32).to(self.device))
                        else:
                            v = 0
                        self.buf.finish_path(v)
                        if terminal:
                            print("Cumulative reward is {}".format(ep_ret))
                            # only save EpRet / EpLen if trajectory finished
                            o, ep_ret, ep_len = self.env.reset( ), 0, 0

            # Perform PPO update!
            self.update()
            # Try Next next time
            if self.lr_decay:
                self.pi_scheduler.step()
                self.vf_scheduler.step()
                print("current epoch is {}, current pi lr is {}, vf lr is {}"\
                      .format(epoch, self.pi_optimizer.param_groups[0]['lr'], self.vf_optimizer.param_groups[0]['lr']))
                
            if epoch % 20 == 0:
                self.logger.save_model(self.ac, epoch ,self.pi_optimizer, self.vf_optimizer,
                                       self.pi_scheduler, self.vf_scheduler, "epoch {}.pth".format(epoch))
                if max_ret < all_ret:
                    max_ret = all_ret
                    self.logger.save_model(self.ac, epoch, self.pi_optimizer, self.vf_optimizer,
                                            self.pi_scheduler, self.vf_scheduler, "best_model.pth")
            
            all_ret = 0