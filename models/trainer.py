import torch
import numpy as np
import pandas as pd
from agent import NeuroEvolution, TradingPolicy

class MarketSimulator:

    def __init__(self, phi=0.01, tick_size=0.001):
        self.phi = phi 
        self.tick_size = tick_size
        self.reset()

    def reset(self):
        self.inventory = 0.0  # q_t
        self.cash = 0.0
        self.wealth = 0.0     # W_t
        self.last_wealth = 0.0
        return 0.0 
    
    def step(self, action_idx, mid_price, ask_1, bid_1):
        action_map = {
            0:(1,1), 1:(2,2), 2:(1,2), 3:(2,1), 4:(3,3), 5:(5,5)
        }
        off_b, off_a = action_map[action_idx]
        
        my_bid = mid_price - off_b * self.tick_size
        my_ask = mid_price + off_a * self.tick_size

        fill_buy = 1 if my_bid >= bid_1 else 0
        fill_sell = 1 if my_ask <= ask_1 else 0

        if fill_buy:
            self.inventory += 1
            self.cash -= my_bid
        if fill_sell:
            self.inventory -= 1
            self.cash += my_ask

        self.wealth = self.cash + self.inventory * mid_price

        reward = (self.wealth - self.last_wealth) - self.phi * (self.inventory ** 2)

        self.last_wealth = self.wealth
        return reward
    
def run_drl_training(sgu1_signals, sgu2_signals, mid_prices, ask_prices, bid_prices):
    evolver = NeuroEvolution(population_size=50, sigma=0.05)
    simulator = MarketSimulator(phi=0.01)
    for gen in range(100):  
        pop_weights = evolver.ask()
        fitness_scores = []

        for weights in pop_weights:
            agent = TradingPolicy()
            agent.set_weights(weights)
            
            total_reward = 0
            simulator.reset()

            for t in range(len(mid_prices)):

                state = torch.tensor([[sgu1_signals[t], sgu2_signals[t], simulator.inventory]], 
                                     dtype=torch.float32)
                
                action = agent.forward(state)

                reward = simulator.step(action, mid_prices[t], ask_prices[t], bid_prices[t])
                total_reward += reward
            
            fitness_scores.append(total_reward)

        best_f = evolver.tell(pop_weights, fitness_scores)
        print(f"Generation {gen} | Best Fitness: {best_f:.4f} | Avg Fitness: {np.mean(fitness_scores):.4f}")

    torch.save(evolver.master_policy.state_dict(), "best_mm_agent.pth")