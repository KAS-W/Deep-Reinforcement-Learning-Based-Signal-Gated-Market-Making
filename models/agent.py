import torch
import torch.nn as nn
import numpy as np

class TradingPolicy(nn.Module):
    def __init__(self, state_dim=3, action_dim=6, hidden_dim=64):
        super(TradingPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        with torch.no_grad():
            logits = self.net(x)
            return torch.argmax(logits, dim=1).item()
        
    def get_weights(self):
        return torch.cat([p.data.view(-1) for p in self.parameters()])
    
    def set_weights(self, weights):
        current_idx = 0
        for p in self.parameters():
            numel = p.data.numel()
            p.data.copy_(weights[current_idx:current_idx + numel].view(p.data.shape))
            current_idx += numel

class NeuroEvolution:
    def __init__(self, population_size=50, sigma=0.1):
        self.pop_size = population_size
        self.sigma = sigma 
        self.master_policy = TradingPolicy()

    def ask(self):
        population_weights = []
        master_weights = self.master_policy.get_weights()
        for _ in range(self.pop_size):

            noise = torch.randn_like(master_weights) * self.sigma
            population_weights.append(master_weights + noise)
        return population_weights
    
    def tell(self, population_weights, fitness_scores):
        best_idx = np.argmax(fitness_scores)
        self.master_policy.set_weights(population_weights[best_idx])
        return fitness_scores[best_idx]