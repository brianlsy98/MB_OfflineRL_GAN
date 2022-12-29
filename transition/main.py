# ===== add python path ===== #
import glob
import sys
import os
PATH = os.getcwd()
for dir_idx, dir_name in enumerate(PATH.split('/')):
    dir_path = '/'.join(PATH.split('/')[:(dir_idx+1)])
    file_list = [os.path.basename(sub_dir) for sub_dir in glob.glob(f"{dir_path}/.*")]
    if '.git_package' in file_list:
        PATH = dir_path
        break
if not PATH in sys.path:
    sys.path.append(PATH)
# =========================== #

from transition import EnsembleTransition
from utils.color import cprint

from sklearn.utils import shuffle
import numpy as np
import argparse
import pickle
import torch
import wandb
import gym

EPS = 1e-8

def getParser():
    parser = argparse.ArgumentParser(description='RL')
    # common
    parser.add_argument('--wandb', action='store_true', help='use wandb?')
    parser.add_argument('--slack', action='store_true', help='use slack?')
    parser.add_argument('--test', action='store_true', help='test or train?')
    parser.add_argument('--device', type=str, default='gpu', help='gpu or cpu.')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index.')
    # for env
    parser.add_argument('--env_name', type=str, default='halfcheetah', help='gym environment name.')
    parser.add_argument('--dataset', type=str, default='medium', help='random, medium, expert, etc.')
    return parser

def main(args):
    # parameters
    transition_config = {
        'name': 'EnsembleTransition',
        'activation': 'ReLU',
        'hidden_dim': 512,
        'max_grad_norm': 1.0,
        'lr': 3e-4,
        'n_ensembles': 4,
        'batch_size': 10000,
        'total_epochs': 100,
    }

    if args.env_name == 'hopper':
        env = gym.make('Hopper-v3')
    elif args.env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
    elif args.env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
    elif args.env_name == 'ant':
        env = gym.make('Ant-v3')
    else:
        raise NotImplementedError
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = f'dataset/{args.env_name}-{args.dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # get each component
    observations = np.concatenate([tr['observations'] for tr in trajectories], axis=0)
    actions = np.concatenate([tr['actions'] for tr in trajectories], axis=0)
    next_observations = np.concatenate([tr['next_observations'] for tr in trajectories], axis=0)
    rewards = np.concatenate([tr['rewards'] for tr in trajectories], axis=0)
    rewards = np.reshape(rewards, (-1, 1))

    # normalize
    obs_mean, obs_std = np.mean(observations, axis=0), np.std(observations, axis=0)
    reward_mean, reward_std = np.mean(rewards, axis=0), np.std(rewards, axis=0)
    norm_states = (observations - obs_mean)/(obs_std + EPS)
    norm_rewards = (rewards - reward_mean)/(reward_std + EPS)
    norm_next_states = (next_observations - obs_mean)/(obs_std + EPS)

    # convert dict to namespace
    transition_args = argparse.Namespace()
    for key, value in transition_config.items():
        setattr(transition_args, key, value)
    # set remained variables
    transition_args.save_dir = f"transition/results/{args.env_name}/{args.dataset}/{transition_args.name}"
    transition_args.device = args.device
    transition_args.obs_dim = obs_dim
    transition_args.action_dim = action_dim
    transition = EnsembleTransition(transition_args)

    # wandb
    if args.wandb:
        project_name = '[Offline RL] Ensemble Transition'
        wandb.init(
            project=project_name, 
            config=transition_args,
        )
        run_idx = wandb.run.name.split('-')[-1]
        wandb.run.name = f"{transition_args.name}-{run_idx}"

    # training
    for epoch in range(transition_args.total_epochs):
        norm_states, actions, norm_next_states, norm_rewards = \
            shuffle(norm_states, actions, norm_next_states, norm_rewards, random_state=0)
        cnt = 0
        losses = []
        while True:
            start_idx = cnt
            end_idx = cnt + transition_args.batch_size
            if end_idx > len(norm_states): break
            states_tensor = torch.tensor(norm_states[start_idx:end_idx], device=args.device, dtype=torch.float32)
            actions_tensor = torch.tensor(actions[start_idx:end_idx], device=args.device, dtype=torch.float32)
            next_states_tensor = torch.tensor(norm_next_states[start_idx:end_idx], device=args.device, dtype=torch.float32)
            rewards_tensor = torch.tensor(norm_rewards[start_idx:end_idx], device=args.device, dtype=torch.float32)
            loss = transition.train(states_tensor, actions_tensor, next_states_tensor, rewards_tensor)
            losses.append(loss)
            cnt = end_idx
        loss = np.mean(losses)
        wandb.log({'dynamics/loss': loss})
        print(cnt, loss)
    transition.save()


if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()
    # device
    if torch.cuda.is_available() and args.device == 'gpu':
        device = torch.device(f'cuda:{args.gpu_idx}')
        cprint('[torch] cuda is used.', bold=True, color='orange')
    else:
        device = torch.device('cpu')
        cprint('[torch] cpu is used.', bold=True, color='orange')
    args.device = device
    main(args)
