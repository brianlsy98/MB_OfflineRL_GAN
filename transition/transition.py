from transition.models import TransitionModel
from utils.color import cprint

import numpy as np
import torch
import os

class EnsembleTransition:
    def __init__(self, args) -> None:
        # set parameters
        self.name = args.name
        self.checkpoint_dir=f'{args.save_dir}/checkpoint'
        self.device = args.device
        self.n_ensembles = args.n_ensembles
        self.lr = args.lr
        self.max_grad_norm = args.max_grad_norm

        # define NN & optimizers
        self.models = []
        self.parameters = []
        for _ in range(self.n_ensembles):
            model = TransitionModel(args).to(self.device)
            self.models.append(model)
            self.parameters += list(model.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.lr)

        # load
        self._load()


    ''' Public Functions
    '''
    def predict(self, states, actions):
        next_states = []
        rewards = []
        for model_idx in range(self.n_ensembles):
            temp_next_states, temp_rewards = self.models[model_idx](states, actions)
            next_states.append(temp_next_states)
            rewards.append(temp_rewards)
        next_states = torch.mean(torch.stack(next_states), dim=0)
        rewards = torch.mean(torch.stack(rewards), dim=0)
        return next_states, rewards

    def train(self, states, actions, next_states, rewards):
        pred_next_states, pred_rewards = self.predict(states, actions)
        loss = torch.mean(0.5 * torch.square(pred_next_states - next_states))
        loss += torch.mean(0.5 * torch.square(pred_rewards - rewards))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, self.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def save(self):
        save_dict = dict()
        for i in range(self.n_ensembles):
            save_dict[f'model_{i}'] = self.models[i].state_dict()
        save_dict[f'optimizer'] = self.optimizer.state_dict()
        torch.save(save_dict, f"{self.checkpoint_dir}/model.pt")
        cprint(f'[{self.name}] save success.', bold=True, color='blue')


    ''' Private Functions
    '''
    def _load(self):
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = f"{self.checkpoint_dir}/model.pt"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            for i in range(self.n_ensembles):
                self.models[i].load_state_dict(checkpoint[f'model_{i}'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            cprint(f'[{self.name}] load success.', bold=True, color='blue')
        else:
            for i in range(self.n_ensembles):
                self.models[i].initialize()
            cprint(f'[{self.name}] load fail.', bold=True, color='red')
