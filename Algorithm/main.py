# MAIN code for Model based offline rl with GAN
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

from transition.transition import EnsembleTransition
from discriminator import Discriminator
from utils.logger import Logger
from utils.color import cprint
from agent import Agent

from collections import deque
from copy import deepcopy
import numpy as np
import argparse
import pickle
import random
import torch
import wandb
import time
import gym

EPS = 1e-8

def getParser():
    parser = argparse.ArgumentParser(description='RL')
    # common
    parser.add_argument('--wandb',  action='store_true', help='use wandb?')
    parser.add_argument('--slack',  action='store_true', help='use slack?')
    parser.add_argument('--test',  action='store_true', help='test or train?')
    parser.add_argument('--device', type=str, default='gpu', help='gpu or cpu.')
    parser.add_argument('--name', type=str, default='CQL-GAN', help='save name.')
    parser.add_argument('--save_freq', type=int, default=int(1e3), help='# of time steps for save.')
    parser.add_argument('--slack_freq', type=int, default=int(2.5e3), help='# of time steps for slack message.')
    parser.add_argument('--total_epochs', type=int, default=int(2e4), help='total training steps.')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index.')
    parser.add_argument('--seed', type=int, default=1, help='seed number.')
    # for env
    parser.add_argument('--env_name', type=str, default='halfcheetah', help='gym environment name.')
    parser.add_argument('--dataset', type=str, default='medium', help='random, medium, expert, etc.')
    parser.add_argument('--max_episode_steps', type=int, default=1000, help='maximum steps of each episode.')
    parser.add_argument('--n_rollout_steps', type=int, default=1, help='# of rollout steps.')
    # for networks
    parser.add_argument('--activation', type=str, default='ReLU', help='activation function. ReLU, Tanh, Sigmoid...')
    parser.add_argument('--hidden_dim', type=int, default=512, help='the number of hidden layer\'s node.')
    parser.add_argument('--log_std_init', type=float, default=0.0, help='log of initial std.')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='maximum of grad_nrom.')
    parser.add_argument('--value_lr', type=float, default=1e-3, help='value learning rate.')
    parser.add_argument('--policy_lr', type=float, default=1e-4, help='policy learning rate.')
    # for RL
    parser.add_argument('--discount_factor', type=float, default=0.99, help='discount factor.')
    parser.add_argument('--batch_size', type=int, default=1000, help='batch size.')
    parser.add_argument('--polyak', type=float, default=0.995, help='polyak.')
    parser.add_argument('--value_epochs', type=int, default=10, help='value epochs.')
    # for entropy alpha
    parser.add_argument('--entropy_d', type=float, default=-1.0, help='entropy limit value per action dim.')
    parser.add_argument('--entropy_alpha_init', type=float, default=1.0, help='initial alpha value.')
    parser.add_argument('--entropy_alpha_lr', type=float, default=0.01, help='alpha learning rate.')
    # for disc alpha
    parser.add_argument('--disc_d', type=float, default=0.6, help='discriminator limit value per action dim.')
    parser.add_argument('--disc_alpha_init', type=float, default=5.0, help='initial disc alpha value.')
    parser.add_argument('--disc_alpha_lr', type=float, default=0.01, help='disc alpha learning rate.')
    # for conservative
    parser.add_argument('--conservative_d', type=float, default=5.0, help='conservative limit value per action dim.')
    parser.add_argument('--conservative_alpha_init', type=float, default=0.01, help='initial conservative alpha value.')
    parser.add_argument('--conservative_alpha_lr', type=float, default=0.01, help='alpha learning rate.')
    parser.add_argument('--n_action_samples', type=int, default=10, help='CQL action samples.')
    return parser


def main(args):
    # define envs
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

    # load dataset
    dataset_path = f'{PATH}/dataset/{args.env_name}-{args.dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
    observations = np.concatenate([tr['observations'] for tr in trajectories], axis=0)
    rewards = np.concatenate([tr['rewards'] for tr in trajectories], axis=0)
    obs_mean, obs_std = np.mean(observations, axis=0), np.std(observations, axis=0)
    reward_mean, reward_std = np.mean(rewards, axis=0), np.std(rewards, axis=0)

    # set args value for env
    args.obs_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.action_bound_min = env.action_space.low
    args.action_bound_max = env.action_space.high

    # transition parameters
    transition_config = {
        'name': 'EnsembleTransition',
        'activation': 'ReLU',
        'hidden_dim': 512,
        'max_grad_norm': 1.0,
        'lr': 3e-4,
        'n_ensembles': 4,
        'batch_size': 10000,
    }

    # convert dict to namespace
    transition_args = argparse.Namespace()
    for key, value in transition_config.items():
        setattr(transition_args, key, value)

    # load transition model
    transition_args.save_dir = f"{PATH}/transition/results/{args.env_name}/{args.dataset}/{transition_args.name}"
    transition_args.device = args.device
    transition_args.obs_dim = args.obs_dim
    transition_args.action_dim = args.action_dim
    transition = EnsembleTransition(transition_args)

    # discriminator parameters
    disc_config = {
        'name': 'Discriminator',
        'activation': 'ReLU',
        'hidden_dim': 512,
        'max_grad_norm': 1.0,
        'lr': 3e-4,
    }

    # convert dict to namespace
    disc_args = argparse.Namespace()
    for key, value in disc_config.items():
        setattr(disc_args, key, value)

    # load transition model
    disc_args.save_dir = f"results/{args.env_name}/{disc_args.name}_s{args.seed}"
    disc_args.device = args.device
    disc_args.obs_dim = args.obs_dim
    disc_args.action_dim = args.action_dim
    discriminator = Discriminator(disc_args)

    # define agent
    agent = Agent(args)

    # fill replay buffer with offline dataset
    for trajectory in trajectories:
        observations = (trajectory['observations'] - obs_mean)/(obs_std + EPS)
        next_observations = (trajectory['next_observations'] - obs_mean)/(obs_std + EPS)
        actions = trajectory['actions']
        rewards = trajectory['rewards']
        dones = np.zeros(len(observations))
        dones[-1] = 1.0
        fails = trajectory['terminals']
        for t_idx in range(len(trajectory['observations'])):
            agent.addTransition(
                observations[t_idx], actions[t_idx], rewards[t_idx], 
                float(dones[t_idx]), float(fails[t_idx]), next_observations[t_idx],
            )

    # wandb
    if args.wandb:
        project_name = f'[Offline RL] CQL-GAN-{args.env_name}'
        wandb.init(
            project=project_name, 
            config=args,
        )
        run_idx = wandb.run.name.split('-')[-1]
        wandb.run.name = f"{args.name}-{run_idx}"

    # logger
    score_logger = Logger(args.save_dir, 'score')
    eplen_logger = Logger(args.save_dir, 'eplen')

    # train
    for epoch in range(args.total_epochs):
        # sample trajectories
        sampled_trajectory = random.sample(agent.replay_buffer, min(agent.batch_size, len(agent.replay_buffer)))
        real_states = torch.tensor(np.array([t[0] for t in sampled_trajectory]), device=args.device, dtype=torch.float32)
        real_actions = agent._normalizeAction(torch.tensor(np.array([t[1] for t in sampled_trajectory]), device=args.device, dtype=torch.float32))
        real_rewards = torch.tensor(np.array([t[2] for t in sampled_trajectory]), device=args.device, dtype=torch.float32)
        real_fails = torch.tensor(np.array([t[4] for t in sampled_trajectory]), device=args.device, dtype=torch.float32)
        real_next_states = torch.tensor(np.array([t[5] for t in sampled_trajectory]), device=args.device, dtype=torch.float32)

        # generate states
        gen_states_list = []
        gen_actions_list = []
        gen_rewards_list = []
        gen_next_states_list = []
        states = real_states
        for roll_idx in range(args.n_rollout_steps):
            actions = agent.getAction(states, is_train=True)
            next_states, norm_rewards = transition.predict(states, actions)
            rewards = norm_rewards.reshape((-1,))*reward_std + reward_mean
            gen_states_list.append(states)
            gen_actions_list.append(actions)
            gen_rewards_list.append(rewards)
            gen_next_states_list.append(next_states)
            states = next_states

        # update value
        gen_states = torch.cat(gen_states_list, dim=0).detach()
        value_loss, value_difference, conservative_alpha = agent.valueUpdate(
            real_states, real_actions, real_rewards, real_fails, real_next_states,
        )

        # =========== update policy =========== #
        # get last value
        with torch.no_grad():
            entropy_alpha = agent._getEntropyAlpha()
            disc_alpha = agent._getDiscAlpha()
        last_states = gen_next_states_list[-1]
        _, sampled_actions, _ = agent.policy.sample(last_states)
        last_value = torch.mean(torch.min(agent.reward_value1(last_states, sampled_actions), agent.reward_value2(last_states, sampled_actions)))
        # calculate cumulative reward sum
        discount_factors = torch.tensor(
            [args.discount_factor**temp_t_idx for temp_t_idx in range(args.n_rollout_steps)], 
            dtype=torch.float32, device=args.device,
        ).tile((agent.batch_size, 1)).T
        reward_sum = torch.mean(torch.sum(discount_factors*torch.stack(gen_rewards_list, dim=0), dim=0))
        reward_sum += (args.discount_factor**args.n_rollout_steps)*last_value
        # get entropy
        _, _, log_probs = agent.policy.sample(gen_states)
        entropy = -torch.mean(log_probs)
        # calculate policy loss
        total_gen_states = torch.cat(gen_next_states_list, dim=0)
        disc_value = torch.mean(torch.log(discriminator.predict(total_gen_states)))
        # policy_loss = -reward_sum + entropy_alpha*(agent.entropy_d*agent.action_dim - entropy) + disc_alpha*(disc_value - np.log(agent.disc_d))
        policy_loss = -reward_sum + entropy_alpha*(agent.entropy_d*agent.action_dim - entropy)

        # policy update
        agent.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), agent.max_grad_norm)
        agent.policy_optimizer.step()
        # agent.policy_scheduler.step()
        # Entropy Alpha Update
        agent.entropy_alpha_optimizer.zero_grad()
        entropy_constraint = agent.entropy_d*agent.action_dim
        entropy_alpha_loss = -agent.log_entropy_alpha*(entropy_constraint - entropy.item())
        entropy_alpha_loss.backward()
        agent.entropy_alpha_optimizer.step()
        agent.log_entropy_alpha.data = torch.clamp(agent.log_entropy_alpha, -8.0, 8.0)
        # Disc Alpha Update
        agent.disc_alpha_optimizer.zero_grad()
        disc_constraint = np.log(agent.disc_d)
        disc_alpha_loss = -agent.log_disc_alpha*(disc_value.item() - disc_constraint)
        disc_alpha_loss.backward()
        agent.disc_alpha_optimizer.step()
        agent.log_disc_alpha.data = torch.clamp(agent.log_disc_alpha, -8.0, 8.0)
        # ===================================== #

        # update discriminator
        sampled_trajectory = random.sample(agent.replay_buffer, min(total_gen_states.shape[0], len(agent.replay_buffer)))
        total_real_states = torch.tensor(np.array([t[0] for t in sampled_trajectory]), device=agent.device, dtype=torch.float32)
        disc_loss = -torch.mean(torch.log(discriminator.predict(total_gen_states.detach()))) - torch.mean(torch.log(1 - discriminator.predict(total_real_states)))
        gp_loss = discriminator.getGPLoss(total_real_states, total_gen_states)
        # disc_loss += 0.01*gp_loss   # halfcheetah
        # disc_loss += 5*gp_loss   # walker2d
        disc_loss += 0.01*gp_loss   # hopper
        # disc_loss += 0.01*gp_loss   # ant
        discriminator.optimizer.zero_grad()
        disc_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.model.parameters(), discriminator.max_grad_norm)
        discriminator.optimizer.step()
        # discriminator.scheduler.step()

        # get accuracy
        gen_acc = torch.mean((discriminator.predict(total_gen_states) > 0.5).float())
        real_acc = torch.mean((discriminator.predict(total_real_states) <= 0.5).float())

        # ======== for evaluation ======== #
        obs = env.reset()
        obs = (obs - obs_mean)/(obs_std + EPS)
        done = False
        score = 0.0
        step = 0
        while True:
            step += 1
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, device=args.device, dtype=torch.float32)
                action_tensor = agent.getAction(obs_tensor, False)
                action = action_tensor.detach().cpu().numpy()
            obs, reward, done, info = env.step(action)
            obs = (obs - obs_mean)/(obs_std + EPS)
            score += reward
            if done: break
        score_logger.write([1, score])
        eplen_logger.write([1, step])
        # ================================ #

        log = {
            'train/disc_loss': disc_loss.item(), 
            'train/policy_loss': policy_loss.item(), 
            'train/gp_loss': gp_loss.item(),
            'train/value_loss': value_loss,
            'value/gen_reward_sum': (reward_sum - (args.discount_factor**args.n_rollout_steps)*last_value).item(),
            'value/gen_last_value': last_value.item(),
            'metric/gen_acc': gen_acc.item(),
            'metric/real_acc': real_acc.item(),
            'metric/gen_disc_value': disc_value.item() - np.log(agent.disc_d),
            'metric/value_difference': value_difference,
            'metric/entropy': entropy.item(),
            'metric/conservative_alpha': conservative_alpha,
            'metric/disc_alpha': disc_alpha.item(),
            'metric/entropy_alpha': entropy_alpha.item(),
            'rollout/score': score_logger.get_avg(10),
            'rollout/ep_len': eplen_logger.get_avg(10),            
        }
        if args.wandb:
            wandb.log(log)
        print(log)

        if (epoch + 1)%args.save_freq == 0:
            discriminator.save()
            agent.save()


if __name__ == "__main__":
    parser = getParser()
    args = parser.parse_args()
    # save_dir
    args.save_dir = f"results/{args.env_name}/{args.name}_s{args.seed}"
    # device
    if torch.cuda.is_available() and args.device == 'gpu':
        device = torch.device(f'cuda:{args.gpu_idx}')
        cprint('[torch] cuda is used.', bold=True, color='orange')
    else:
        device = torch.device('cpu')
        cprint('[torch] cpu is used.', bold=True, color='orange')
    args.device = device
    main(args)
