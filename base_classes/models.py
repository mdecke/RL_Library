from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from base_classes.utils import get_activation


class Actor(nn.Module):
    def __init__(self, input_dim:int, output_dim:int, action_limit:float, hidden_dims:List[int], lr:float, activation_fct:str, stochastic:bool, seed:int=42):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.action_limit = action_limit

        layers = []
        prev_dim = self.input_dim

        self.stochastic = stochastic

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation_fct))
            prev_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
        
        if self.stochastic:
            self.mu_head = nn.Linear(prev_dim, self.output_dim)
            self.log_sigma_head = nn.Linear(prev_dim, self.output_dim)
        else:
            self.out = nn.Linear(prev_dim, self.output_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input:torch.Tensor)->torch.Tensor:
        logits = self.net(input)
        if self.stochastic:
            mu = self.mu_head(logits)
            log_std = self.log_sigma_head(logits)
            if self.training:
                sigma = torch.exp(log_std) + 1e-5 #ensure not sigma not zero for numerical stibility
                actions = torch.normal(mu,sigma, generator=self.generator)
            else:
                actions =  mu
        else:
            actions = self.out(logits)
        
        bounded_actions = torch.tanh(actions) * self.action_limit
        # bounded_actions = torch.clamp(actions, -self.action_limit, self.action_limit)
        return bounded_actions

    def save(self, filepath:str)->None:
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath:str)->None:
        self.load_state_dict(torch.load(filepath))
        self.eval()


class Critic(nn.Module):
    def __init__(self, input_dim:int, hidden_dims:List[int], lr:float, activation_fct:str, output_dim:int=1):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr

        layers = []
        prev_dim = self.input_dim


        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(get_activation(activation_fct))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, self.output_dim)) # action prediction
        self.net = nn.Sequential(*layers)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, input:torch.Tensor)->torch.Tensor:
        return self.net(input)

    def save(self, filepath:str)->None:
        torch.save(self.state_dict(), filepath)

    def load(self, filepath:str)->None:
        self.load_state_dict(torch.load(filepath))
        self.eval()
