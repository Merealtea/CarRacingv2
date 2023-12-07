import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from src.utils.logger import Logger
from os.path import join
from src.utils.replay_buffer import PPOBuffer, to_device
from src.utils.env import Wrapped_Env
from torch.optim.lr_scheduler import StepLR
import os
import cv2

class PPO():
    def __init__(self, env,actor_critic, history_len, device , ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, save_path = None, schedule_step_size = 5, decay_rate = 0.5 , mini_batchsize = 32, adv_norm = False,
        lr_decay = False, ent_coef = 0, vf_coef = 0, pi_coef = 0, clip_value  = False, data_augument_rate = 0, train =True):

        # Random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.seed = seed

        self.env = Wrapped_Env(env, history_len)

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

        self.max_ret = -100

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.actor.parameters(), lr=pi_lr, eps= 1e-5)
        self.vf_optimizer = Adam(self.ac.critic.parameters(), lr=vf_lr, eps= 1e-5)
        
        self.pi_scheduler = StepLR(self.pi_optimizer, step_size=schedule_step_size, gamma=decay_rate)
        self.vf_scheduler = StepLR(self.vf_optimizer, step_size=schedule_step_size, gamma=decay_rate)
        self.save_path = save_path

        # Logger
        if train:
            self.logger = Logger(save_path)
            # Load model
            if "best_model.pth" in os.listdir(save_path):
                ckpt = torch.load(join(save_path, "best_model.pth"))
                self.start_epoch = ckpt["epoch"]
                
                
                self.ac.load_state_dict(ckpt["ac"])
                self.pi_optimizer.load_state_dict(ckpt["pi_optim"])
                self.vf_optimizer.load_state_dict(ckpt["vf_optim"])
                self.pi_scheduler.load_state_dict(ckpt["pi_scheduler"])
                self.vf_scheduler.load_state_dict(ckpt["vf_scheduler"])
                self.max_ret = ckpt["max_ret"]
                print("Load Previous ckpt and training from epoch {}, average reward is {}".format(self.start_epoch, self.max_ret))
        else:
            self.save_path = None
            ckpt = torch.load(join(save_path, "best_model.pth"))
            self.start_epoch = ckpt["epoch"]            
            self.ac.load_state_dict(ckpt["ac"])
            self.validate(self.start_epoch)

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
     
                loss.backward()      
                torch.nn.utils.clip_grad_norm_(self.ac.critic.parameters(), 0.5) 
                torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 0.5) 
                
                self.vf_optimizer.step()
                self.pi_optimizer.step()

                # Log changes from update
                kl, ent, cf, steer_std, gas_steer = pi_info['kl'], pi_info['ent'].item(), pi_info['cf'], pi_info["steer_std"], pi_info["gas_std"]
                self.logger.log_update_step(kl, ent, cf, steer_std, gas_steer)

                # if kl > 1.5 * self.target_kl:
                #     # print("Update stop in {} iteration due to kl is {}".format(i, kl))
                #     break

        self.ac.eval()
        

    def train(self):
        # Prepare for interaction with environment
        o, ep_ret, ep_len = self.env.reset( ), 0, 0

        # self.env.render()
        self.max_ret = -100

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

                    timeout = ep_len == self.max_ep_len
                    if timeout:
                        print("Time out for current trajectory")
                    
                    # save to buffer
                    self.buf.store(o, ori_a, r, v, logp)
                    
                    # Update obs (critical!)
                    o = next_o
                    
                    terminal = d or timeout
                    epoch_ended = t==self.steps_per_epoch-1
                    self.logger.log_mean_reward(r, terminal)

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
                                       self.pi_scheduler, self.vf_scheduler, self.max_ret,"epoch {}.pth".format(epoch))
            
            if epoch % 5 == 0:
                episode_reward = self.validate(epoch)
                if self.max_ret < episode_reward:
                    self.max_ret = episode_reward
                    print("Save best model for epoch {}".format(epoch))
                    self.logger.save_model(self.ac, epoch, self.pi_optimizer, self.vf_optimizer,
                                            self.pi_scheduler, self.vf_scheduler, self.max_ret, "best_model.pth")

    def validate(self, epoch):
        # self.env.render
        episode_num = 10
        all_ret = 0

        # if self.save_path is not None:
        #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #     out = cv2.VideoWriter(join(self.save_path, "val_{}.avi".format(epoch)),fourcc, 50, (96, 96))

        # Main loop: collect experience in env and update/log each epoch
        print("Validation for epoch {}".format(epoch))
        for _ in range(episode_num):
            o = self.env.reset( )
            d = False
            while not d:
                with torch.no_grad():
                    a, ori_a, v, logp = \
                        self.ac.step(torch.as_tensor(o, dtype=torch.float32).to(self.device))
                    
                    next_o, r, d, next_orin_o, _ = self.env.step(a)
                    # if self.save_path is not None:
                    #     next_orin_o = cv2.cvtColor(next_orin_o, cv2.COLOR_BGR2RGB)
                    #     out.write(next_orin_o)
                    all_ret += r
                    o = next_o
        print("Mean episode reward for epoch {} is {}".format(epoch, all_ret / episode_num)) 
        # if self.save_path is not None:
        #     out.release()
        return all_ret / episode_num