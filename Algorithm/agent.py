from typing import Optional, List

from utils.color import cprint
from models import Policy
from models import Value

from torch.distributions import Normal
from collections import deque
from scipy.stats import norm
from copy import deepcopy
import numpy as np
import pickle
import random
import torch
import copy
import time
import os

EPS = 1e-8

@torch.jit.script
def normalize(a, maximum, minimum):
    temp_a = 2.0/(maximum - minimum)
    temp_b = (maximum + minimum)/(minimum - maximum)
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b

@torch.jit.script
def unnormalize(a, maximum, minimum):
    temp_a = (maximum - minimum)/2.0
    temp_b = (maximum + minimum)/2.0
    temp_a = torch.ones_like(a)*temp_a
    temp_b = torch.ones_like(a)*temp_b
    return temp_a*a + temp_b

@torch.jit.script
def clip(a, maximum, minimum):
    clipped = torch.where(a > maximum, maximum, a)
    clipped = torch.where(clipped < minimum, minimum, clipped)
    return clipped

class Agent:
    def __init__(self, args):
        # base
        self.device = args.device
        self.name = args.name
        self.checkpoint_dir=f'{args.save_dir}/checkpoint'

        # for env
        self.discount_factor = args.discount_factor
        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.action_bound_min = torch.tensor(args.action_bound_min, device=args.device, dtype=torch.float32)
        self.action_bound_max = torch.tensor(args.action_bound_max, device=args.device, dtype=torch.float32)

        # for policy & value
        self.value_lr = args.value_lr
        self.policy_lr = args.policy_lr
        self.max_grad_norm = args.max_grad_norm
        self.batch_size = args.batch_size
        self.polyak = args.polyak
        self.value_epochs = args.value_epochs

        # for replay buffer
        self.replay_buffer = []

        # declare networks
        self.policy = Policy(args).to(args.device)
        self.reward_value1 = Value(args).to(args.device)
        self.reward_value2 = Value(args).to(args.device)
        self.target_reward_value1 = Value(args).to(args.device)
        self.target_reward_value2 = Value(args).to(args.device)

        # for entropy_alpha
        self.entropy_d = args.entropy_d
        self.entropy_alpha_lr = args.entropy_alpha_lr
        self.log_entropy_alpha = torch.tensor(np.log(args.entropy_alpha_init + EPS), requires_grad=True, device=args.device)

        # for entropy_alpha
        self.disc_d = args.disc_d
        self.disc_alpha_lr = args.disc_alpha_lr
        self.log_disc_alpha = torch.tensor(np.log(args.disc_alpha_init + EPS), requires_grad=True, device=args.device)

        # for conservative
        self.n_action_samples = args.n_action_samples
        self.conservative_d = args.conservative_d
        self.log_conservative_alpha = torch.tensor(np.log(args.conservative_alpha_init + EPS), requires_grad=True, device=args.device)
        self.conservative_alpha_lr = args.conservative_alpha_lr
        self.tot_epoch = 0

        # optimizers
        self.reward_value_params = list(self.reward_value1.parameters()) + list(self.reward_value2.parameters())
        self.reward_value_optimizer = torch.optim.Adam(self.reward_value_params, lr=self.value_lr)
        self.entropy_alpha_optimizer = torch.optim.Adam([self.log_entropy_alpha], lr=self.entropy_alpha_lr)
        self.disc_alpha_optimizer = torch.optim.Adam([self.log_disc_alpha], lr=self.disc_alpha_lr)
        self.conservative_alpha_optimizer = torch.optim.Adam([self.log_conservative_alpha], lr=self.conservative_alpha_lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.policy_lr)
        # self.policy_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.policy_optimizer,
        #                                 lr_lambda = lambda epoch: 0.9995 ** epoch,
        #                                 last_epoch=-1,
        #                                 verbose=False)                                       

        # load
        self._load()


    def addTransition(self, state, action, reward, done, fail, next_state):
        self.replay_buffer.append([state, action, reward, done, fail, next_state])

    def getAction(self, state:torch.Tensor, is_train:bool) -> List[torch.Tensor]:
        '''
        input:
            states:     Tensor(state_dim,)
            is_train:   boolean
        output:
            action:         Tensor(action_dim,)
            cliped_action:  Tensor(action_dim,)
        '''
        mu, pi, logp_pi = self.policy.sample(state)
        if is_train:
            action = self._unnormalizeAction(pi)
        else:
            action = self._unnormalizeAction(mu)
        return action

    def valueUpdate(self, real_states, real_actions, real_rewards, real_fails, real_next_states):
        for i in range(self.value_epochs):
            # get target values
            with torch.no_grad():
                entropy_alpha = self._getEntropyAlpha()
                conservative_alpha = 0.999**self.tot_epoch * self._getConservativeAlpha()
                _, next_pi, log_prob = self.policy.sample(real_next_states)
                next_reward_values_tensor = torch.min(
                    self.target_reward_value1(real_next_states, next_pi), 
                    self.target_reward_value2(real_next_states, next_pi)
                )
                real_target_reward_values_tensor = real_rewards \
                    + (1.0 - real_fails)*self.discount_factor*(next_reward_values_tensor - entropy_alpha*log_prob)

            # get overestimation difference
            policy_values1 = []
            policy_values2 = []
            random_values = []
            for _ in range(self.n_action_samples):
                with torch.no_grad():
                    _, sampled_actions_tensor, _ = self.policy.sample(real_states)
                policy_values1.append(torch.stack([
                    self.reward_value1(real_states, sampled_actions_tensor), 
                    self.reward_value2(real_states, sampled_actions_tensor)
                ]))
                with torch.no_grad():
                    _, sampled_actions_tensor, _ = self.policy.sample(real_next_states)
                policy_values2.append(torch.stack([
                    self.reward_value1(real_states, sampled_actions_tensor), 
                    self.reward_value2(real_states, sampled_actions_tensor)
                ]))
                with torch.no_grad():
                    zero_tensor = torch.zeros((len(real_states), self.action_dim), device=self.device, dtype=torch.float32)
                    sampled_actions_tensor = zero_tensor.uniform_(-1.0, 1.0)
                random_values.append(torch.stack([
                    self.reward_value1(real_states, sampled_actions_tensor), 
                    self.reward_value2(real_states, sampled_actions_tensor)
                ]))
            policy_values = torch.cat([
                torch.stack(policy_values1, dim=2), torch.stack(policy_values2, dim=2), torch.stack(random_values, dim=2)
            ], dim=2)
            pred_values = torch.mean(torch.logsumexp(policy_values, dim=2))
            current_values = torch.mean(torch.stack([
                self.reward_value1(real_states, real_actions), 
                self.reward_value2(real_states, real_actions)
            ]))
            value_difference = pred_values - current_values

            # reward value update
            self.reward_value_optimizer.zero_grad()
            reward_value_loss = torch.mean(torch.square(self.reward_value1(real_states, real_actions) - real_target_reward_values_tensor))
            reward_value_loss += torch.mean(torch.square(self.reward_value2(real_states, real_actions) - real_target_reward_values_tensor))
            reward_value_loss += conservative_alpha*value_difference
            reward_value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reward_value_params, self.max_grad_norm)
            self.reward_value_optimizer.step() 

            # update target network
            self._softUpdate(self.target_reward_value1, self.reward_value1, self.polyak)
            self._softUpdate(self.target_reward_value2, self.reward_value2, self.polyak)

            # conservative alpha update
            self.conservative_alpha_optimizer.zero_grad()
            conservative_alpha_loss = -self.log_conservative_alpha*(value_difference.item() - self.conservative_d)
            conservative_alpha_loss.backward()
            self.conservative_alpha_optimizer.step()
            self.log_conservative_alpha.data = torch.clamp(self.log_conservative_alpha, -8.0, 8.0)

        # self.tot_epoch += 1

        return reward_value_loss.item(), value_difference.item(), conservative_alpha.item()

    def save(self):
        torch.save({
            'target_reward_value1': self.target_reward_value1.state_dict(),
            'target_reward_value2': self.target_reward_value2.state_dict(),
            'reward_value1': self.reward_value1.state_dict(),
            'reward_value2': self.reward_value2.state_dict(),
            'policy': self.policy.state_dict(),
            'log_entropy_alpha': self.log_entropy_alpha,
            'log_disc_alpha': self.log_disc_alpha,
            'log_conservative_alpha': self.log_conservative_alpha,
            'reward_value_optimizer': self.reward_value_optimizer.state_dict(),
            'entropy_alpha_optimizer': self.entropy_alpha_optimizer.state_dict(),
            'disc_alpha_optimizer': self.disc_alpha_optimizer.state_dict(),
            'conservative_alpha_optimizer': self.conservative_alpha_optimizer.state_dict(),
        }, f"{self.checkpoint_dir}/model.pt")
        cprint(f'[{self.name}] save success.', bold=True, color="blue")

    def _softUpdate(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * tau + param.data * (1.0 - tau))

    def _normalizeAction(self, a:torch.Tensor) -> torch.Tensor:
        return normalize(a, self.action_bound_max, self.action_bound_min)

    def _unnormalizeAction(self, a:torch.Tensor) -> torch.Tensor:
        return unnormalize(a, self.action_bound_max, self.action_bound_min)

    def _getEntropyAlpha(self):
        alpha = torch.exp(self.log_entropy_alpha)
        return alpha

    def _getDiscAlpha(self):
        alpha = torch.exp(self.log_disc_alpha)
        return alpha

    def _getConservativeAlpha(self):
        alpha = torch.exp(self.log_conservative_alpha)
        return alpha

    def _load(self):
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = f"{self.checkpoint_dir}/model.pt"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.target_reward_value1.load_state_dict(checkpoint['target_reward_value1'])
            self.target_reward_value2.load_state_dict(checkpoint['target_reward_value2'])
            self.reward_value1.load_state_dict(checkpoint['reward_value1'])
            self.reward_value2.load_state_dict(checkpoint['reward_value2'])
            self.policy.load_state_dict(checkpoint['policy'])
            self.log_entropy_alpha.data = checkpoint['log_entropy_alpha']
            self.log_disc_alpha.data = checkpoint['log_disc_alpha']
            self.log_conservative_alpha.data = checkpoint['log_conservative_alpha']
            self.reward_value_optimizer.load_state_dict(checkpoint['reward_value_optimizer'])
            self.entropy_alpha_optimizer.load_state_dict(checkpoint['entropy_alpha_optimizer'])
            self.disc_alpha_optimizer.load_state_dict(checkpoint['disc_alpha_optimizer'])
            self.conservative_alpha_optimizer.load_state_dict(checkpoint['conservative_alpha_optimizer'])
            cprint(f'[{self.name}] load success.', bold=True, color="blue")
        else:
            self.policy.initialize()
            self.reward_value1.initialize()
            self.reward_value2.initialize()
            self._softUpdate(self.target_reward_value1, self.reward_value1, tau=0)
            self._softUpdate(self.target_reward_value2, self.reward_value2, tau=0)
            cprint(f'[{self.name}] load fail.', bold=True, color="red")
