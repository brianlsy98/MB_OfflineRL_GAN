from torch.nn import functional as F
from torch import jit, nn
import numpy as np
import torch

LOG_STD_MAX = 2
LOG_STD_MIN = -4
EPS = 1e-8

def initWeights(m, init_value=0.0):
    if isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.normal_(init_value, 0.01)

class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()

        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.hidden1_units = args.hidden_dim
        self.hidden2_units = args.hidden_dim
        self.log_std_init = args.log_std_init
        self.activation = args.activation

        self.fc1 = nn.Linear(self.obs_dim, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.activ = eval(f'torch.nn.{self.activation}()')
        self.output_activ = torch.sigmoid

        self.fc_mean = nn.Linear(self.hidden2_units, self.action_dim)
        self.log_std = torch.tensor(
            [self.log_std_init]*self.action_dim, dtype=torch.float32, 
            requires_grad=True, device=args.device
        )
        self.log_std = nn.Parameter(self.log_std)
        self.register_parameter(name="my_log_std", param=self.log_std)


    def forward(self, x):
        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        mean = self.output_activ(self.fc_mean(x))

        log_std = torch.clamp(self.log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        std = torch.ones_like(mean)*torch.exp(log_std)
        return mean, log_std, std

    def initialize(self):
        for m_idx, module in enumerate(self.children()):
            module.apply(initWeights)


class Policy2(nn.Module):
    def __init__(self, args):
        super(Policy2, self).__init__()
        
        self.state_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.hidden1_units = args.hidden_dim
        self.hidden2_units = args.hidden_dim
        self.activation = args.activation
        self.log_std_init = args.log_std_init
        
        self.fc1 = nn.Linear(self.state_dim, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.act_fn = eval(f'F.{self.activation.lower()}')
        
        self.fc_mean = nn.Linear(self.hidden2_units, self.action_dim)
        self.fc_log_std = nn.Linear(self.hidden2_units, self.action_dim)


    def forward(self, state):
        x = self.act_fn(self.fc1(state))
        x = self.act_fn(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, log_std, std

    def initialize(self):
        for m_idx, module in enumerate(self.children()):
            if m_idx == 3:
                initializer = lambda m: initWeights(m, init_value=self.log_std_init)
            else:
                initializer = lambda m: initWeights(m)
            module.apply(initializer)


class Value(nn.Module):
    def __init__(self, args):
        super(Value, self).__init__()

        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.hidden1_units = args.hidden_dim
        self.hidden2_units = args.hidden_dim
        self.activation = args.activation

        self.fc1 = nn.Linear(self.obs_dim, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.fc3 = nn.Linear(self.hidden2_units, 1)
        self.activ = eval(f'torch.nn.{self.activation}()')


    def forward(self, x):
        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        x = self.fc3(x)
        x = torch.reshape(x, (-1,))
        return x

    def initialize(self):
        self.apply(initWeights)


class DiscModel(nn.Module):
    def __init__(self, args):
        super(DiscModel, self).__init__()

        self.input_dim = args.obs_dim + args.action_dim
        self.hidden1_units = args.hidden_dim
        self.hidden2_units = args.hidden_dim
        self.activation = args.activation

        self.fc1 = nn.Linear(self.input_dim, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.fc3 = nn.Linear(self.hidden2_units, 1)
        self.activ = eval(f'torch.nn.{self.activation}()')
        self.output_activ = torch.nn.Sigmoid()


    def forward(self, x):
        x = self.activ(self.fc1(x))
        x = self.activ(self.fc2(x))
        x = self.output_activ(self.fc3(x))
        x = torch.reshape(x, (-1,))
        x = torch.clamp(x, min=0.001, max=0.999)
        return x

    def initialize(self):
        self.apply(initWeights)


class TransitionModel(nn.Module):
    def __init__(self, args):
        super(TransitionModel, self).__init__()

        self.obs_dim = args.obs_dim
        self.action_dim = args.action_dim
        self.hidden1_units = args.hidden_dim
        self.hidden2_units = args.hidden_dim
        self.activation = args.activation

        self.fc1 = nn.Linear(self.obs_dim + self.action_dim, self.hidden1_units)
        self.fc2 = nn.Linear(self.hidden1_units, self.hidden2_units)
        self.fc_state = nn.Linear(self.hidden2_units, self.obs_dim)
        self.fc_reward = nn.Linear(self.hidden2_units, 1)
        self.activ = eval(f'torch.nn.{self.activation}()')


    def forward(self, states, actions):
        sa_pairs = torch.cat([states, actions], dim=-1)
        x = self.activ(self.fc1(sa_pairs))
        x = self.activ(self.fc2(x))
        next_states = self.fc_state(x) + states
        rewards = self.fc_reward(x)
        return next_states, rewards

    def initialize(self):
        self.apply(initWeights)
