import torch
from torch import nn
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions import Beta
import torch.nn.functional as F

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        pass

    def step(self, obs):
        raise NotImplementedError
        
    def act(self, obs):
        raise NotImplementedError
  
    
    
class CarRaceActorCritic(ActorCritic):
    def __init__(self, img_h, img_w, act_dim, conv_channels, mlp_channels):
        super(CarRaceActorCritic, self).__init__()
        self.actor = GaussianActor(img_h, img_w, act_dim, conv_channels, mlp_channels)
        self.critic = Critic(img_h, img_w, conv_channels, mlp_channels)

    def wrap_action(self, act):
        # wrapped action : steering, gas, brake
        act = act.cpu().detach().numpy()[0]

        wrapped_act = np.zeros((3,))
        wrapped_act[0] = act[0]
        wrapped_act[1] = 0 if act[1] < 0 else act[1]
        wrapped_act[2] = 0 if act[1] > 0 else act[1]
        return wrapped_act

    def step(self, obs):
        pi = self.actor._distribution(obs)
        original_action = pi.rsample()
        original_action = torch.clamp(original_action, -1, 1)

        logp_a = self.actor._log_prob_from_distribution(pi, original_action)

        v = self.critic(obs)
        wrapped_action = self.wrap_action(original_action)
        return wrapped_action, original_action.cpu().detach().numpy(), v.cpu().detach().numpy(), logp_a.cpu().detach().numpy()
    
    def act(self, obs):
        return self.step(obs)[0]


class Actor(nn.Module):
    def __init__(self, img_h, img_w, conv_channels):
        super(Actor, self).__init__()
        self.encoder = ImageEncoder(conv_channels)

        img_sample = torch.randn(1, conv_channels[0], img_h, img_w)
        with torch.no_grad():
            x = self.encoder(img_sample)
            self.hidden_size = x.shape[1]

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a
    

class GaussianActor(Actor):
    def __init__(self, img_h, img_w, act_dim, conv_channels, mlp_channels):
        super(GaussianActor, self).__init__(img_h, img_w, conv_channels)
        log_std = 0 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = MLPNetwork([self.hidden_size] + mlp_channels)

        self.act_mean = nn.Linear(mlp_channels[-1], act_dim)
        self.tanh = nn.Tanh()

        orthogonal_init(self.act_mean)

    def _distribution(self, obs):
        encoded_feature = self.encoder(obs)
        mu = self.mu_net(encoded_feature)
        mu = self.act_mean(mu)
        mu = self.tanh(mu)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

class BetaActor(Actor):
    def __init__(self, img_h, img_w, act_dim, conv_channels, mlp_channels):
        super(BetaActor, self).__init__(img_h, img_w, conv_channels)

        self.alpha_layer = MLPNetwork([self.hidden_size] + mlp_channels)
        self.beta_layer = MLPNetwork([self.hidden_size] + mlp_channels)
        orthogonal_init(self.alpha_layer)
        orthogonal_init(self.beta_layer)

    def _distribution(self, obs):
        encoded_feature = self.encoder(obs)
        alpha = F.softplus(self.alpha_layer(encoded_feature)) + 1.0
        beta = F.softplus(self.beta_layer(encoded_feature)) + 1.0
        return Beta(alpha, beta)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution


class Critic(nn.Module):
    def __init__(self, img_h, img_w, conv_channels, mlp_channels):
        super(Critic, self).__init__()

        self.encoder = ImageEncoder(conv_channels)
        img_sample = torch.randn(1, conv_channels[0], img_h, img_w)
        with torch.no_grad():
            x = self.encoder(img_sample)
            self.hidden_size = x.shape[1]

        mlp_channels = [self.hidden_size] + mlp_channels

        self.mlp_layers = nn.Sequential(
            MLPNetwork(mlp_channels),
            nn.Linear(mlp_channels[-1], 1)
        )

    def forward(self, obs):
        encoded_feature = self.encoder(obs)
        v = self.mlp_layers(encoded_feature)
        return v
        # return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.
    

class ImageEncoder(nn.Module):
    def __init__(self, channels : list):
        super(ImageEncoder, self).__init__()
        self.conv = nn.Sequential()
        in_channels, out_channels = channels[:-1], channels[1:]
        for in_channel, out_channel in zip(in_channels, out_channels):
            self.conv.append(ConvLayer(in_channel, out_channel))

        self.flatten = nn.Flatten()

    def forward(self, img):
        x = self.conv(img)
        return self.flatten(x)
    

class MLPNetwork(nn.Module):
    def __init__(self, channels : list):
        super(MLPNetwork, self).__init__()
        self.mlp = nn.Sequential()
        in_channels, out_channels = channels[:-1], channels[1:]
        for in_channel, out_channel in zip(in_channels, out_channels):
            self.mlp.append(LinearLayer(in_channel, out_channel))

    def forward(self, features):
        return self.mlp(features)
    

class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size = 3, stride = 1):
        super(ConvLayer, self).__init__()
        conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride)
        orthogonal_init(conv)
        self.net = nn.Sequential(
            conv,
            nn.Tanh(),
            nn.MaxPool2d(2),
        )
    
    def forward(self, x):
        return self.net(x)


class LinearLayer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LinearLayer, self).__init__()

        self.fc = nn.Linear(in_channel, out_channel)
        self.tanh = nn.Tanh()
        orthogonal_init(self.fc)


    def forward(self, x):
        return self.tanh(self.fc(x))