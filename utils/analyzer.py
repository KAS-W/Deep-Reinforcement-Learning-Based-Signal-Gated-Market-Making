import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def evaluate_sgu1_comparison(model, X_val, y_val, graph_pth):
    preds = model.predict(X_val)
    
    pearson_corr = np.corrcoef(preds, y_val)[0, 1]
    spearman_corr, _ = spearmanr(preds, y_val)
    
    print(f"--- S2 Comparison Metrics ---")
    print(f"Pearson Correlation:  {pearson_corr:.4f}")
    print(f"Spearman Correlation: {spearman_corr:.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.regplot(x=y_val, y=preds, ax=axes[0], 
                scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red'})
    axes[0].set_title("Actual vs Predicted RR")
    axes[0].set_xlabel("Actual Log RR")
    axes[0].set_ylabel("Predicted Log RR")
    axes[1].plot(y_val.values[:300], label='Actual RR', alpha=0.7)
    axes[1].plot(preds[:300], label='Predicted RR', color='orange', linewidth=2)
    axes[1].set_title("Time-Series Comparison (Subset of S2)")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(graph_pth)

def plot_importance(model_obj, graph_pth):
    booster = model_obj.model.get_booster()
    importance = booster.get_score(importance_type='gain')
    df_imp = pd.DataFrame(list(importance.items()), columns=['feature', 'gain'])
    df_imp = df_imp.sort_values('gain', ascending=False)
    plt.figure(figsize=(10, 8))
    plt.barh(df_imp['feature'], df_imp['gain'])
    plt.gca().invert_yaxis()
    plt.title("SGU1 Feature Importance: Gain")
    plt.savefig(graph_pth)