import torch
import numpy as np
import os
from multiprocessing import Pool
from functools import partial
from .market_env import FTPEnv
from models.model import TradingPolicy, NeuroEvolution, AdversaryPolicy

def evaluate_individual(mm_weights, adv_weights, bundle, phi, tick_size, fee_rate, train_stats, use_arl=False):
    # Unpack the data bundle containing price paths and execution bounds 
    s1, s2, mid_next, best_ask, best_bid, buy_max, sell_min = bundle

    # Initialize policies and load weights for the MM and, if enabled, the adversary 
    policy = TradingPolicy()
    policy.set_weights(mm_weights)
    
    if use_arl and adv_weights is not None:
        adv_policy = AdversaryPolicy()
        adv_policy.set_weights(adv_weights)
    else:
        adv_policy = None

    # Initialize the FPT environment with specific risk and fee parameters 
    env = FTPEnv(phi=phi, tick_size=tick_size, fee_rate=fee_rate)

    total_reward, trades = 0, 0
    # Initialize execution flags for the adversary's state (clovervoyance/historical execution) 
    fill_buy_flag, fill_sell_flag = 0.0, 0.0
    
    with torch.no_grad():
        for t in range(len(mid_next)):
            # Market Maker State: [Normalized S1, Normalized S2, Normalized Inventory] 
            mm_state = torch.tensor([[(s1[t] - train_stats['s1_m']) / train_stats['s1_s'], 
                                      (s2[t] - train_stats['s2_m']) / train_stats['s2_s'], 
                                      env.inventory / 2.0]], dtype=torch.float32)
            
            # Generate and scale the Market Maker action 
            mm_raw = policy.forward(mm_state).squeeze().cpu().numpy()
            mm_action = np.round(mm_raw * 5.0).astype(int)

            # Adversary Logic: Observe state and generate perturbation if ARL is active 
            adv_action = None
            if adv_policy is not None:
                # Adversary State: [Inventory, Ask Execution Flag, Bid Execution Flag] 
                adv_state = torch.tensor([[env.inventory / 2.0, fill_sell_flag, fill_buy_flag]], dtype=torch.float32)
                adv_raw = adv_policy.forward(adv_state).squeeze().cpu().numpy()
                # Scale adversary action to influence quotes (typically +/- 1 tick) 
                adv_action = np.round(adv_raw * 1.0).astype(int)

            # Execute step in environment with potential adversarial interference 
            reward, info = env.step(mm_action, mid_next[t], best_ask[t], best_bid[t], 
                                    buy_max[t], sell_min[t], adv_action=adv_action)

            total_reward += reward
            
            # Update execution flags for the next step's adversary state 
            fill_buy_flag = 1.0 if info['fill_buy'] else 0.0
            fill_sell_flag = 1.0 if info['fill_sell'] else 0.0

            if info['fill_buy'] or info['fill_sell']:
                trades += 1

    # Penalize the policy if no trades occurred to prevent degenerate idle behavior 
    if trades == 0: 
        total_reward -= 50.0
        
    return total_reward, trades

class DRLEngine:
    def __init__(self, pop_size=50, sigma=0.05, phi=0.01, tick_size=0.01, fee_rate=0.0, use_arl=False, save_dir="checkpoints/drl"):
        self.phi = phi
        self.tick_size = tick_size
        self.fee_rate = fee_rate
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.use_arl = use_arl
        self.mm_evolver = NeuroEvolution(population_size=pop_size)

        # add adversial if we need it
        if self.use_arl:
            self.adv_evolver = NeuroEvolution(population_size=pop_size)

    def train(self, train_bundle, val_bundle, train_stats, generations=100, output_prefix="agent"):
        best_val_reward = -float('inf')
        no_improvement_gens = 0 
        history = {
            'gen': [], 'train_f': [], 'val_f': [], 
            'train_trades': [], 'val_trades': []
        }

        with Pool(processes=8) as pool:
            for gen in range(generations):
                # 1. Generate population weights
                mm_pop = self.mm_evolver.ask()
                
                # Generate adversary population if ARL is enabled; otherwise use None
                if self.use_arl:
                    adv_pop = self.adv_evolver.ask()
                else:
                    adv_pop = [None] * len(mm_pop)

                # 2. Parallel Evaluation using starmap to pair MM and Adversary weights
                # The evaluate_individual function must accept both mm_weights and adv_weights
                eval_func = partial(
                    evaluate_individual, 
                    bundle=train_bundle, 
                    phi=self.phi, 
                    tick_size=self.tick_size, 
                    fee_rate=self.fee_rate, 
                    train_stats=train_stats,
                    use_arl=self.use_arl
                )
                
                # Execute competitive evaluation in parallel
                results = pool.starmap(eval_func, zip(mm_pop, adv_pop))

                # 3. Extract fitness and update populations
                # Market Maker seeks to maximize the reward 
                fitness_scores = [res[0] for res in results]
                best_train_f = self.mm_evolver.tell(mm_pop, fitness_scores)

                # Adversary seeks to minimize MM reward (maximize -reward) 
                if self.use_arl:
                    adv_fitness = [-res[0] for res in results]
                    self.adv_evolver.tell(adv_pop, adv_fitness)

                # 4. Validate the best MM individual in a standard environment (No Adversary)
                # This measures the robustness of the agent under normal conditions
                best_idx = np.argmax(fitness_scores)
                best_weights_this_gen = mm_pop[best_idx]
                val_reward, val_trades = evaluate_individual(
                    best_weights_this_gen, 
                    None, # No adversarial weights during validation
                    val_bundle, 
                    self.phi, 
                    self.tick_size, 
                    self.fee_rate,
                    train_stats,
                    use_arl=False # Explicitly disable adversary during testing
                )

                # 5. Model Saving and Sigma Decay Logic 
                improvement_tag = ""
                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    no_improvement_gens = 0 
                    save_path = os.path.join(self.save_dir, f"{output_prefix}_best_val_{self.phi}.pth")
                    # Save the master policy of the Market Maker
                    torch.save(self.mm_evolver.master_policy.state_dict(), save_path)
                    improvement_tag = "*"
                else:
                    no_improvement_gens += 1

                # Apply Sigma Decay if no improvement is found for 15 generations 
                if no_improvement_gens >= 15:
                    self.mm_evolver.sigma *= 0.5
                    if self.use_arl:
                        self.adv_evolver.sigma *= 0.5
                    print(f">>> Sigma decayed to {self.mm_evolver.sigma:.4f} due to no improvement")
                    no_improvement_gens = 0 

                # Log training history
                history['gen'].append(gen)
                history['train_f'].append(best_train_f)
                history['val_f'].append(val_reward)
                history['train_trades'].append(results[best_idx][1])
                history['val_trades'].append(val_trades)

                if gen % 5 == 0:
                    arl_status = "ARL:ON" if self.use_arl else "ARL:OFF"
                    print(f"Gen {gen:03d} | {arl_status} | Best Train: {best_train_f:.2f} | Val: {val_reward:.2f}{improvement_tag}")
        
        # Load the best weights discovered during the training session
        best_path = os.path.join(self.save_dir, f"{output_prefix}_best_val_{self.phi}.pth")
        if os.path.exists(best_path):
            self.mm_evolver.master_policy.load_state_dict(torch.load(best_path, weights_only=True))
            
        return self.mm_evolver.master_policy, history