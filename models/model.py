import torch
import torch.nn as nn
import numpy as np

class TradingPolicy(nn.Module):
    """DRL Agent"""
    def __init__(self, state_dim=3, action_dim=2, hidden_dim=32):
        super(TradingPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.9)
                nn.init.constant_(m.bias, 0.05)
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.net(x)
        
    def get_weights(self):
        return torch.cat([p.data.view(-1) for p in self.parameters()]).clone().detach()
    
    def set_weights(self, weights):
        current_idx = 0
        for p in self.parameters():
            numel = p.data.numel()
            p.data.copy_(weights[current_idx:current_idx + numel].view(p.data.shape).to(p.device))
            current_idx += numel

class AdversaryPolicy(nn.Module):
    """Adversial DRL Agent"""
    def __init__(self, input_size=3, hidden_size=12):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2), # two noises：delta_a, delta_b
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)
    
    def set_weights(self, weights):
        current_idx = 0
        for p in self.parameters():
            numel = p.data.numel()
            p.data.copy_(weights[current_idx:current_idx + numel].view(p.data.shape).to(p.device))
            current_idx += numel
    
class NeuroEvolution:
    def __init__(self, population_size=50, sigma=0.05):
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