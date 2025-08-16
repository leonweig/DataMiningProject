import os, random, gc, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, mean_squared_error, mean_absolute_error

from arch import arch_model


def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

#to create directories 
def ensure_dirs(*paths):
    import os
    for p in paths:
        if not os.path.exists(p):
            os.makedirs(p)

#find pairs in dataset
def discover_pairs(zip_path):
    z = zipfile.ZipFile(zip_path, "r")
    names = z.namelist()
    z.close()
    
    #find gzip files
    gzip_files = []
    for n in names:
        if n.endswith(".gzip"):
            gzip_files.append(n)
    
    #find pair names
    pairs = []
    for n in gzip_files:
        pair_name = n.split("/")[0].replace(".csv", "")
        if pair_name not in pairs:
            pairs.append(pair_name)
    
    pairs.sort()
    return pairs

#create a df for a pair
def load_pair_df(pair, zip_path):
    z = zipfile.ZipFile(zip_path, "r")
    file_path = pair + ".csv/" + pair + ".csv.gzip"
    f = z.open(file_path)
    df = pd.read_csv(f)
    f.close()
    z.close()
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    df.drop(columns=["Unnamed: 0","close_time","ignore"], errors="ignore", inplace=True)
    return df

#create feature for pair (we want log return, rolling volatility, taker ratio)
def make_features(df):
    x = df.copy()
    x["log_return"] = np.log(x["close"] / x["close"].shift(1))
    x["rolling_vol_30min"] = x["log_return"].rolling(30).std()
    x["taker_ratio"] = x["tb_quote_av"] / x["quote_av"]
    return x.dropna()

#garch moodel 
def garch_rolling_forecast(returns, window_minutes, horizon_minutes):
    #*1000 because of error messages in optimizer
    r = (returns * 1000).dropna()  
    preds, idx = [], []
    for start in range(0, len(r) - window_minutes - horizon_minutes, horizon_minutes):
        train = r.iloc[start: start + window_minutes]
        t_idx = r.index[start + window_minutes + horizon_minutes - 1]
        try:
            res = arch_model(train, vol="GARCH", p=1, q=1).fit(disp="off")
            var = res.forecast(horizon=horizon_minutes).variance.iloc[-1].values[-1]
            #return to nomral scale
            preds.append(np.sqrt(var) / 1000)  
        except Exception:
            preds.append(np.nan)
        idx.append(t_idx)
    return pd.Series(preds, index=idx)


#evaluate forecast with rmse, mae, directional accuracy
def eval_forecast(true_returns, vol_fc, horizon_minutes):
    realized = true_returns.rolling(window=horizon_minutes).std().reindex(vol_fc.index)
    a, b = realized.dropna(), vol_fc.dropna()
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return float("nan"), float("nan"), float("nan")
    a, b = a.loc[common], b.loc[common]
    rmse = float(np.sqrt(mean_squared_error(a, b)))
    mae  = float(mean_absolute_error(a, b))
    da   = float((np.sign(b.diff().dropna()) == np.sign(a.diff().dropna())).mean())
    return rmse, mae, da

#find anomalies and outliers using isolation forest and local outlier factor
def detect_anomalies(df, seed=42, if_contam='auto', lof_contam='auto'):
    use = df[["log_return","rolling_vol_30min","volume","taker_ratio"]].dropna()
    out = pd.DataFrame(index=df.index)
    if use.empty:
        out["anomaly_score"] = np.nan
        out["lof_score"] = np.nan
        out["anomaly_score_proba"] = np.nan
        out["lof_score_proba"] = np.nan
        return out
    
    Xs = StandardScaler().fit_transform(use)
    
    out["anomaly_score"] = 1
    out["lof_score"] = 1
    out["anomaly_score_proba"] = 0.0
    out["lof_score_proba"] = 0.0
    
    iso = IsolationForest(n_estimators=200, contamination=if_contam, random_state=seed)
    iso_predictions = iso.fit_predict(Xs)
    iso_scores = iso.decision_function(Xs)
    
    out.loc[use.index, "anomaly_score"] = iso_predictions
    out.loc[use.index, "anomaly_score_proba"] = iso_scores
    
    lof = LocalOutlierFactor(n_neighbors=20, contamination=lof_contam)
    lof_predictions = lof.fit_predict(Xs)
    lof_scores = lof.negative_outlier_factor_
    
    out.loc[use.index, "lof_score"] = lof_predictions
    out.loc[use.index, "lof_score_proba"] = lof_scores
    
    return out

#KMEans clustering
def kmeans_quality(df, n_clusters=3, seed=42):
    use = df[["log_return","rolling_vol_30min","volume","taker_ratio"]].dropna()
    if len(use) < n_clusters:
        return np.nan, np.nan, pd.Series(dtype=int, name="kmeans_label")
    Xs = StandardScaler().fit_transform(use)
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=seed)
    labels = pd.Series(km.fit_predict(Xs), index=use.index, name="kmeans_label")
    sil = float(silhouette_score(Xs, labels))
    dbi = float(davies_bouldin_score(Xs, labels))
    return sil, dbi, labels

#find pump and dump events
def pump_dump_flags(df, pump_window=10, dump_window=30, price_up=0.01, price_down=0.01, vol_mult=2.0):
    x = df.copy()
    x["price_pct_change"] = x["close"].pct_change(pump_window)
    x["future_min"] = x["close"].rolling(dump_window, min_periods=1).min().shift(-pump_window)
    x["price_drop_after_pump"] = (x["close"] - x["future_min"]) / x["close"]
    x["volume_roll_median"] = x["volume"].rolling(60).median()
    x["volume_spike"] = x["volume"] > (x["volume_roll_median"] * vol_mult)
    x["pump_and_dump"] = (x["price_pct_change"] > price_up) & x["volume_spike"] & (x["price_drop_after_pump"] > price_down)
    return x[["pump_and_dump"]]

#backtest to evaluate strategy
def backtest(df, signal_index, hold=30, cooldown=60):
    sig = sorted(set(signal_index))
    trades, last = [], None
    for t in sig:
        if last is not None and (t - last).total_seconds() < cooldown * 60:
            continue
        try:
            exit_t = df.index[df.index.get_loc(t) + hold]
        except Exception:
            continue
        entry, exitp = df.loc[t, "close"], df.loc[exit_t, "close"]
        trades.append((exitp - entry) / entry); last = t
    if not trades:
        return 0, float("nan"), float("nan"), float("nan")
    arr = np.array(trades)
    return len(arr), float(arr.mean()), float((arr > 0).mean()), float(np.prod(1 + arr) - 1)



#only run this on the top 10 pairs
def get_top_crypto_pairs():
    top_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "SOLUSDT", "XRPUSDT", "DOTUSDT", "DOGEUSDT","AVAXUSDT","MATICUSDT"]
    return top_pairs

#only get pais for the top 10 from dataset
def discover_top_pairs(zip_path):
    all_pairs = discover_pairs(zip_path)
    
    top_pairs = get_top_crypto_pairs()
    
    available_top_pairs = []
    for pair in top_pairs:
        if pair in all_pairs:
            available_top_pairs.append(pair)
    
    print(f"Found {len(available_top_pairs)}")
    for pair in available_top_pairs:
        print(f"  - {pair}")
    
    return available_top_pairs



#plotting 



#creating graphs 
def plot_kmeans_quality(summary, outpath, dpi=160):
    plt.figure(figsize=(10, 6))
    
    good_data = summary.dropna(subset=["kmeans_silhouette","kmeans_dbi"])
    
    #colr points based on clustering qual
    colors = []
    for dbi in good_data["kmeans_dbi"]:
        if dbi < 1.1:
            colors.append('green')  
        elif dbi < 1.15:
            colors.append('orange')  
        else:
            colors.append('red')  
    
    plt.scatter(good_data["kmeans_silhouette"], good_data["kmeans_dbi"], 
                c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    for i, row in good_data.iterrows():
        plt.annotate(row["pair"], (row["kmeans_silhouette"], row["kmeans_dbi"]), 
                    fontsize=9, alpha=0.8, ha='center', va='bottom')
    
    plt.xlabel("Silhouette Score (higher = better)", fontsize=12)
    plt.ylabel("Davies-Bouldin Index (lower = better)", fontsize=12)
    plt.title("Cryptocurrency Clustering Quality Analysis", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_anomaly_overview(summary, outpath, dpi=160):
    plt.figure(figsize=(12, 6))
    
    pairs = summary['pair']
    if_anoms = summary['if_anoms']
    
    sorted_data = summary.sort_values('if_anoms', ascending=False)
    max_anoms = sorted_data['if_anoms'].max()
    colors = []
    for count in sorted_data['if_anoms']:
        intensity = count / max_anoms
        colors.append((1.0, 1.0 - intensity, 1.0 - intensity)) 
    
    bars = plt.bar(sorted_data['pair'], sorted_data['if_anoms'], color=colors, 
                   edgecolor='black', linewidth=0.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Cryptocurrency Trading Pairs', fontsize=12)
    plt.ylabel('Number of Anomalies Detected', fontsize=12)
    plt.title('Anomaly Detection Results Across Cryptocurrency Pairs', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.show()

def plot_backtest_results(summary, outpath, dpi=160):
    valid_data = summary.dropna(subset=['backtest_mean'])
    
    if len(valid_data) == 0:
        plt.figure()
        plt.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=16)
        plt.savefig(outpath, dpi=dpi)
        plt.show()
        return
    
    sorted_data = valid_data.sort_values('backtest_mean', ascending=False)
    pairs = sorted_data['pair']
    returns = sorted_data['backtest_mean']
    win_rates = sorted_data['backtest_win']
    
    plt.figure(figsize=(12, 6))
    

    colors = []
    for r in returns:
        if r > 0.001:
            colors.append('darkgreen')
        elif r > 0:
            colors.append('lightgreen')
        elif r > -0.001:
            colors.append('orange')
        else:
            colors.append('red')
    
    bars = plt.bar(pairs, returns * 100, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=0.5)
    
    for i, (bar, win_rate) in enumerate(zip(bars, win_rates)):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., 
                height + (0.01 if height >= 0 else -0.05),
                f'{height:.2f}%\n(Win: {win_rate:.1%})', 
                ha='center', va='bottom' if height >= 0 else 'top', 
                fontsize=9)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Cryptocurrency Trading Pairs', fontsize=12)
    plt.ylabel('Average Return (%)', fontsize=12)
    plt.title('Trading Strategy Backtest Results', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.show()

def plot_forecast_metrics(summary, outpath, dpi=160):
    valid_data = summary.dropna(subset=['directional_acc'])
    
    if len(valid_data) == 0:
        plt.figure()
        plt.text(0.5, 0.5, 'No Data Available', ha='center', va='center', fontsize=16)
        plt.savefig(outpath, dpi=dpi)
        plt.show()
        return
    
    sorted_data = valid_data.sort_values('directional_acc', ascending=False)
    pairs = sorted_data['pair']
    accuracy = sorted_data['directional_acc']
    
    plt.figure(figsize=(12, 6))
    
    colors = []
    for acc in accuracy:
        if acc > 0.45:
            colors.append('darkgreen')
        elif acc > 0.40:
            colors.append('lightgreen')
        elif acc > 0.35:
            colors.append('orange')
        else:
            colors.append('red')
    
    bars = plt.bar(pairs, accuracy * 100, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    


    plt.axhline(y=50, color='red', linestyle='--', alpha=0.7, 
                label='Random Chance (50%)')
    
    plt.xlabel('Cryptocurrency Trading Pairs', fontsize=12)
    plt.ylabel('Directional Forecast Accuracy (%)', fontsize=12)
    plt.title('Price Direction Prediction Accuracy', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.legend()
    plt.ylim(30, 50)
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.show()

def plot_correlation_analysis(summary, outpath, dpi=160):
    plt.figure(figsize=(10, 8))
    
    if 'if_anoms' in summary.columns and 'pump_events' in summary.columns:
        plt.subplot(2, 2, 1)
        plt.scatter(summary['if_anoms'], summary['pump_events'], 
                   c='red', alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
        for i, row in summary.iterrows():
            plt.annotate(row['pair'], (row['if_anoms'], row['pump_events']), 
                        fontsize=8, alpha=0.8, ha='center', va='bottom')
        plt.xlabel('Anomalies Detected')
        plt.ylabel('Pump Events')
        plt.title('Anomalies vs Pump Events')
        plt.grid(True, alpha=0.3)
        
        if 'directional_acc' in summary.columns:
            plt.subplot(2, 2, 2)
            plt.scatter(summary['if_anoms'], summary['directional_acc'] * 100,
                       c='blue', alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
            plt.xlabel('Anomalies Detected')
            plt.ylabel('Forecast Accuracy (%)')
            plt.title('Anomalies vs Prediction Accuracy')
            plt.grid(True, alpha=0.3)
        
        if 'backtest_mean' in summary.columns:
            plt.subplot(2, 2, 3)
            plt.scatter(summary['if_anoms'], summary['backtest_mean'] * 100,
                       c='green', alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
            plt.xlabel('Anomalies Detected')
            plt.ylabel('Average Return (%)')
            plt.title('Anomalies vs Trading Returns')
            plt.grid(True, alpha=0.3)
        

        if 'kmeans_dbi' in summary.columns and 'rmse' in summary.columns:
            plt.subplot(2, 2, 4)
            plt.scatter(summary['rmse'], summary['kmeans_dbi'],
                       c='purple', alpha=0.7, s=100, edgecolors='black', linewidth=0.5)
            plt.xlabel('Price Volatility (RMSE)')
            plt.ylabel('Clustering Quality (DBI)')
            plt.title('Volatility vs Clustering')
            plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'Insufficient Data for Correlation Analysis', 
                ha='center', va='center', fontsize=16)
    
    plt.suptitle('Cryptocurrency Market Analysis Correlations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.show()

def plot_market_performance_overview(summary, outpath, dpi=160):
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 3, 1)
    sorted_returns = summary.sort_values('backtest_mean', ascending=True)
    colors = ['red' if x < 0 else 'green' for x in sorted_returns['backtest_mean']]
    plt.barh(range(len(sorted_returns)), sorted_returns['backtest_mean'] * 100, color=colors, alpha=0.7)
    plt.yticks(range(len(sorted_returns)), sorted_returns['pair'], fontsize=10)
    plt.xlabel('Average Return (%)')
    plt.title('Trading Performance Ranking')
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.subplot(2, 3, 2)
    plt.scatter(summary['rmse'], summary['backtest_mean'] * 100, 
               c=summary['directional_acc'], cmap='RdYlGn', 
               s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    plt.colorbar(label='Forecast Accuracy')
    plt.xlabel('Risk (RMSE)')
    plt.ylabel('Return (%)')
    plt.title('Risk vs Return Profile')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    plt.hist(summary['if_anoms'], bins=8, color='orange', alpha=0.7, edgecolor='black')
    plt.xlabel('Anomalies Detected')
    plt.ylabel('Number of Pairs')
    plt.title('Anomaly Distribution')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(2, 3, 4)
    win_rates = summary['backtest_win'] * 100
    colors = ['red' if x < 45 else 'orange' if x < 50 else 'green' for x in win_rates]
    plt.bar(summary['pair'], win_rates, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    plt.axhline(y=50, color='black', linestyle='--', alpha=0.5, label='Break-even')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Win Rate (%)')
    plt.title('Trading Win Rates')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(2, 3, 5)
    acc_data = summary['directional_acc'] * 100
    plt.bar(summary['pair'], acc_data, 
           color=['red' if x < 40 else 'orange' if x < 45 else 'green' for x in acc_data],
           alpha=0.7, edgecolor='black', linewidth=0.5)
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Random Chance')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy (%)')
    plt.title('Prediction Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(2, 3, 6)
    if 'kmeans_silhouette' in summary.columns:
        plt.bar(summary['pair'], summary['kmeans_silhouette'], 
               color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Silhouette Score')
        plt.title('Market Structure Quality')
        plt.grid(True, alpha=0.3, axis='y')
    else:
        plt.text(0.5, 0.5, 'No Clustering Data', ha='center', va='center')
    
    plt.suptitle('Cryptocurrency Market Analysis Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.show()
