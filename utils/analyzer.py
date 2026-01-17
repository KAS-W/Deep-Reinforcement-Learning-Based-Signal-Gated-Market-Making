import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_sgu1_comparison(model, X_val, y_val, graph_pth):
    preds = model.predict(X_val)
    y_true = np.asarray(y_val).flatten()
    preds = np.asarray(preds).flatten()

    mse = np.mean((preds - y_true)**2)
    rmse = np.sqrt(mse)

    baseline_mse = np.mean((0 - y_true)**2)
    improvement = (1 - mse / (baseline_mse + 1e-9)) * 100

    std_ratio = np.std(preds) / (np.std(y_true) + 1e-9)

    print(f"--- SGU1 Paper Replication Metrics ---")
    print(f"Validation MSE:        {mse:.6f}")
    print(f"Validation RMSE:       {rmse:.6f}")
    print(f"Relative Improvement:  {improvement:.2f}% (vs Zero-Baseline)")
    print(f"Signal Std Ratio:      {std_ratio:.4f} (Ideal: 0.3-0.7)")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.regplot(x=y_true, y=preds, ax=axes[0], 
                scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red'})
    axes[0].set_title(f"Numerical Fit (Improvement: {improvement:.1f}%)")
    axes[0].set_xlabel("Actual Slope")
    axes[0].set_ylabel("Predicted Slope")

    axes[1].plot(y_true[:200], label='Actual Slope', alpha=0.6)
    axes[1].plot(preds[:200], label='Predicted Slope (SGU2)', color='orange', linewidth=2)
    axes[1].set_title("Time-Series Realization (First 200 Samples)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(graph_pth)
    plt.close()

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

def plot_sgu2_loss(history, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss (MSE)', color='blue')
    plt.plot(history['val_loss'], label='Val Loss (MSE)', color='red', linestyle='--')
    plt.title('SGU2 LSTM Training: Loss Reduction')
    plt.xlabel('Epochs')
    plt.ylabel('MSE (BP^2)') 
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def evaluate_sgu1(model, X_val, y_val, graph_pth, is_xgb=True):
    if is_xgb:
        preds = model.predict(X_val) 
    else:
        preds = model.predict(X_val).flatten() 
        
    y_true = np.asarray(y_val).flatten()
    preds = np.asarray(preds).flatten()

    mse = np.mean((preds - y_true)**2)
    rmse = np.sqrt(mse)
    baseline_mse = np.mean((0 - y_true)**2)
    improvement = (1 - mse / (baseline_mse + 1e-9)) * 100
    std_ratio = np.std(preds) / (np.std(y_true) + 1e-9)
    corr = np.corrcoef(preds, y_true)[0, 1]

    print(f"\n--- SGU1 (Range Prediction) Metrics ---")
    print(f"Correlation:         {corr:.4f}")
    print(f"Validation RMSE:     {rmse:.6f}")
    print(f"Relative Improvement: {improvement:.2f}% (vs Zero-Baseline)")
    print(f"Signal Std Ratio:    {std_ratio:.4f} (Ideal: 0.3-0.7)")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.regplot(x=y_true, y=preds, ax=axes[0], scatter_kws={'alpha':0.2, 's':10}, line_kws={'color':'red'})
    axes[0].set_title(f"SGU1: Range Fit (RI: {improvement:.1f}%)")
    axes[0].set_xlabel("Actual Price Range")
    axes[0].set_ylabel("Predicted Price Range")

    axes[1].plot(y_true[:200], label='Actual Range', alpha=0.6)
    axes[1].plot(preds[:200], label='Predicted Range', color='orange')
    axes[1].set_title("SGU1: Time-Series (First 200)")
    axes[1].legend()
    
    plt.savefig(graph_pth)
    plt.close()

def evaluate_sgu2(model, X_val, y_val, graph_pth):
    preds = model.predict(X_val).flatten()
    y_true = np.asarray(y_val).flatten()

    pearson_corr = np.corrcoef(preds, y_true)[0, 1]
    
    mse = np.mean((preds - y_true)**2)
    baseline_mse = np.mean((0 - y_true)**2)
    improvement = (1 - mse / (baseline_mse + 1e-9)) * 100
    std_ratio = np.std(preds) / (np.std(y_true) + 1e-9)

    lag_corr = np.corrcoef(preds[:-1], y_true[1:])[0, 1]

    print(f"\n--- SGU2 (Trend Prediction) Metrics ---")
    print(f"Pearson Correlation:  {pearson_corr:.4f}")
    print(f"Lag-1 Correlation:    {lag_corr:.4f} (Should be < Pearson)")
    print(f"Relative Improvement: {improvement:.2f}%")
    print(f"Signal Std Ratio:     {std_ratio:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.regplot(x=y_true, y=preds, ax=axes[0], scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red'})
    axes[0].set_title(f"SGU2: Returns Fit (Corr: {pearson_corr:.3f})")
    axes[0].set_xlabel("Actual Returns (BP)")
    axes[0].set_ylabel("Predicted Returns (BP)")

    axes[1].plot(y_true[:200], label='Actual Returns', alpha=0.5)
    axes[1].plot(preds[:200], label='Predicted Returns', color='green', linewidth=1.5)
    axes[1].set_title("SGU2: Trend Realization (First 200)")
    axes[1].set_ylabel("Basis Points (BP)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(graph_pth)
    plt.close()