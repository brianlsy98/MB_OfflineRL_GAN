from typing import Optional, List

from utils.color import cprint
from models import DiscModel

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

class Discriminator:
    def __init__(self, args):
        # base
        self.device = args.device
        self.name = args.name
        self.checkpoint_dir=f'{args.save_dir}/checkpoint'
        self.max_grad_norm = args.max_grad_norm
        self.lr = args.lr

        # define model
        self.model = DiscModel(args).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
        #                                 lr_lambda = lambda epoch: 0.999 ** epoch,
        #                                 last_epoch=-1,
        #                                 verbose=False)

        # load model
        self._load()

    def predict(self, states):
        return self.model(states)

    def getGPLoss(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = torch.autograd.Variable(interpolated, requires_grad=True)
        interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.model(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                                create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + EPS)

        # Return gradient penalty
        return ((gradients_norm - 1.0)**2).mean()

    def save(self):
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, f"{self.checkpoint_dir}/model.pt")
        cprint(f'[{self.name}] save success.', bold=True, color="blue")

    def _load(self):
        if not os.path.isdir(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_file = f"{self.checkpoint_dir}/model.pt"
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            cprint(f'[{self.name}] load success.', bold=True, color="blue")
        else:
            self.model.initialize()
            cprint(f'[{self.name}] load fail.', bold=True, color="red")

