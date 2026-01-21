import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class StrategyAnalytics:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._returns = self.df['wealth'].diff().fillna(0)
        self._wealth = self.df['wealth']
        self._trades = self.df[self.df['is_trade'] == True]

    @property
    def total_pnl(self) -> float:
        """Terminal wealth"""
        return self._wealth.iloc[-1] - self._wealth.iloc[0]
    
    @property
    def max_drawdown(self) -> float:
        """Max drawdown"""
        cumulative_max = self._wealth.cummax()
        drawdown = self._wealth - cumulative_max
        return drawdown.min()
    
    @property
    def mean_absolute_position(self) -> float:
        """Mean absolute position"""
        return self.df['inventory'].abs().mean()
    
    @property
    def pnl_to_map_ratio(self) -> float:
        map_val = self.mean_absolute_position
        if map_val == 0: return 0.0
        return self.total_pnl / map_val
    
    @property
    def sharpe_ratio(self) -> float:
        if len(self._trades) < 2:
            return 0.0
        
        trade_pnls = self._trades['wealth'].diff().dropna()
        std = trade_pnls.std()
        if std == 0:
            return 0.0
        
        return trade_pnls.mean() / std
    
    @property
    def summary_dict(self) -> dict:
        return {
            'Total PnL': self.total_pnl,
            'MAP (Risk)': self.mean_absolute_position,
            'PnLMAP (Eff)': self.pnl_to_map_ratio,
            'Max DD': self.max_drawdown,
            'Sharpe': self.sharpe_ratio,
            'Trades': len(self._trades)
        }
    
class BacktestVisualizer:
    @staticmethod
    def plot_professional_report(df, metrics, save_path='report.png', show_fees=False):
        n_base_plots = 10
        n_rows = n_base_plots + (1 if show_fees else 0) + 1
        fig, axes = plt.subplots(n_rows, 1, figsize=(16, 4 * n_rows))

        # 1. Mid Price
        axes[0].plot(df['step'], df['mid'], color='black', alpha=0.8, label='Mid Price')
        axes[0].set_title('Market Mid Price', fontweight='bold')

        # 2. PnL & Position
        ax2_pnl = axes[1].plot(df['step'], df['wealth'], color='#1f77b4', marker='*', markersize=3, 
                               linewidth=1, alpha=0.7, label='Terminal Wealth')
        axes[1].set_title('Wealth & Inventory History', fontweight='bold')
        axes[1].set_ylabel('Wealth')
        ax2_pos = axes[1].twinx()
        ax2_pos.step(df['step'], df['inventory'], where='post', color='red', alpha=0.4, label='Inventory')
        ax2_pos.set_ylabel('Position')
        ax2_pos.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

        # 3. (Un)Realized PnL
        axes[2].plot(df['step'], df.get('realized_pnl', 0), 'g-', label='Realized PnL', alpha=0.7)
        axes[2].plot(df['step'], df.get('unrealized_pnl', 0), 'r-', label='Unrealized PnL', alpha=0.7)
        axes[2].set_title('(Un)Realized PnL', fontweight='bold')

        # 4. Step Reward
        axes[3].bar(df['step'], df['reward'], color='#FF6B6B', alpha=0.6, label='Step Reward')
        axes[3].set_title('Step Reward', fontweight='bold')

        # 5. Reward Components
        reward_components = ['inventory_reward', 'pnl_reward', 'spread_reward']
        colors = ['#4ECDC4', '#FFD166', '#06D6A0']
        bottom = np.zeros(len(df))
        for i, comp in enumerate(reward_components):
            if comp in df.columns:
                axes[4].bar(df['step'], df[comp], bottom=bottom, color=colors[i], alpha=0.7, label=comp)
                bottom += df[comp].values
        axes[4].set_title('Reward Components Decomposition', fontweight='bold')

        # 6. Spread
        axes[5].plot(df['step'], df.get('spread', df['ask'] - df['bid']), color='#9D4EDD', label='Market Spread')
        axes[5].set_title('Market Spread (Ticks)', fontweight='bold')

        # 7. Cash
        axes[6].plot(df['step'], df['cash'], color='purple', label='Cash Balance')
        axes[6].set_title('Cash Flow', fontweight='bold')

        # 8. Cum Rewards
        axes[7].plot(df['step'], df['cum_reward'], color='orange', label='Cumulative Reward')
        axes[7].set_title('Cumulative Reward Evolution', fontweight='bold')

        # 9. Skew
        axes[8].scatter(df['step'], df['skew'], s=8, color='brown', alpha=0.5, label='Skew (off_b - off_a)')
        axes[8].set_title('Policy Skewness (Asymmetry)', fontweight='bold')

        # 10. Bid & Ask Offsets
        axes[9].scatter(df['step'], df['off_b'], s=10, color='green', label='Bid Offset', alpha=0.4)
        axes[9].scatter(df['step'], df['off_a'], s=10, color='red', label='Ask Offset', alpha=0.4)
        axes[9].set_title('Agent Action: Tick Offsets from Best Bid/Ask', fontweight='bold')

        # 11. fee rates if consider transaction costs
        if show_fees:
            
            ax_fee = axes[10]
            ax_fee.plot(df['step'], -df['cum_fees'], color='brown', label='Total Fees')
            ax_fee.fill_between(df['step'], 0, -df['cum_fees'], color='brown', alpha=0.1)
            ax_fee.set_title("Transaction Cost Erosion")
            ax_fee.set_ylabel("Currency Unit")
            ax_fee.legend(loc='upper left')
            ax_fee.grid(True, alpha=0.3)

        # 12. Metrics Summary Panel
        summary_ax = axes[-1]
        summary_ax.axis('off')
        
        metric_str = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        summary_ax.text(0.5, 0.5, metric_str, transform=summary_ax.transAxes, 
                              fontsize=15, ha='center', va='center', fontweight='bold', 
                              bbox=dict(facecolor='#f0f0f0', alpha=0.9, boxstyle='round,pad=1'))
        
        for ax in axes[:-1]:
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper left', fontsize=8)
            ax.set_xlabel('Environmental Step')

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()