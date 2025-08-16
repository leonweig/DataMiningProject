import numpy as np
import pandas as pd

from config import (
    SEED, DATA_DIR, RESULTS_DIR, PLOTS_DIR, ZIP_PATH,
    DATE_SLICE, N_CLUSTERS, IF_CONTAM, LOF_CONTAM,
    ROLLING_WINDOW_MIN, HORIZON_MIN, SAVE_FIG_DPI
)
from helpers import (
    set_seed, ensure_dirs, discover_pairs, discover_top_pairs, load_pair_df, make_features,
    garch_rolling_forecast, eval_forecast,
    detect_anomalies, kmeans_quality,
    pump_dump_flags, backtest,
    plot_kmeans_quality, plot_anomaly_overview, plot_backtest_results, 
    plot_forecast_metrics, plot_correlation_analysis, plot_market_performance_overview
)

def run(pairs, date_slice, n_clusters):
    set_seed(SEED); ensure_dirs(RESULTS_DIR, PLOTS_DIR)
    rows = []

    for pair in pairs:
        raw = load_pair_df(pair, ZIP_PATH)
        df  = make_features(raw)

        #slice if month exists to only run on the month
        if date_slice:
            months = df.index.strftime("%Y-%m").unique()
            if date_slice in months:
                df = df.loc[date_slice].copy()

        #forecasting
        vol_fc = garch_rolling_forecast(df["log_return"], ROLLING_WINDOW_MIN, HORIZON_MIN)
        rmse, mae, da = eval_forecast(df["log_return"], vol_fc, HORIZON_MIN)

        #anomalis
        anoms = detect_anomalies(df, seed=SEED, if_contam=IF_CONTAM, lof_contam=LOF_CONTAM)

        #KMeans
        sil, dbi, km = kmeans_quality(df, n_clusters=n_clusters, seed=SEED)
        if not km.empty:
            df.loc[km.index, "kmeans_label"] = km

        #Heuristic and overlaps
        heur = pump_dump_flags(df)
        pumps = int(heur["pump_and_dump"].sum())
        
        # Count anomalies 
        if_anoms = int((anoms["anomaly_score"] == -1).sum())
        lof_anoms = int((anoms["lof_score"] == -1).sum())

        # Calculate overlaps  pump and dump events
        if_anomaly_idx = anoms.index[anoms["anomaly_score"] == -1]
        lof_anomaly_idx = anoms.index[anoms["lof_score"] == -1]
        pump_idx = heur.index[heur["pump_and_dump"]]
        
        if_overlap = len(if_anomaly_idx.intersection(pump_idx))
        lof_overlap = len(lof_anomaly_idx.intersection(pump_idx))

        #backtest isolation forest AND pumpdump 
        sig_idx = if_anomaly_idx.intersection(pump_idx)
        ntr, mean_ret, win, cumret = backtest(df, sig_idx)

        rows.append(dict(
            pair=pair,
            rmse=rmse, mae=mae, directional_acc=da,
            if_anoms=if_anoms, lof_anoms=lof_anoms, pump_events=pumps,
            if_pump_overlap=if_overlap, lof_pump_overlap=lof_overlap,
            kmeans_silhouette=sil, kmeans_dbi=dbi,
            backtest_trades=ntr, backtest_mean=mean_ret, backtest_win=win, backtest_cum=cumret
        ))
        print(f"[OK] {pair}")

    summary = pd.DataFrame(rows)
    out_csv = RESULTS_DIR + "/summary_metrics.csv"
    summary.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    return summary

if __name__ == "__main__":
    pairs = discover_top_pairs(ZIP_PATH)
    print(f"Running analysis on {len(pairs)} top crypto pairs")
    
    summary = run(pairs, DATE_SLICE, 3) 
    
    # make plot
    if "kmeans_silhouette" in summary.columns and "kmeans_dbi" in summary.columns:
        plot_file = PLOTS_DIR + "/kmeans_quality.png"
        plot_kmeans_quality(summary, plot_file, dpi=SAVE_FIG_DPI)
    
    plot_file = PLOTS_DIR + "/anomaly_overview.png"
    plot_anomaly_overview(summary, plot_file, dpi=SAVE_FIG_DPI)
    
    plot_file = PLOTS_DIR + "/backtest_results.png"
    plot_backtest_results(summary, plot_file, dpi=SAVE_FIG_DPI)
    
    plot_file = PLOTS_DIR + "/forecast_metrics.png"
    plot_forecast_metrics(summary, plot_file, dpi=SAVE_FIG_DPI)
    
    plot_file = PLOTS_DIR + "/correlation_matrix.png"
    plot_correlation_analysis(summary, plot_file, dpi=SAVE_FIG_DPI)
    
    plot_file = PLOTS_DIR + "/market_overview.png"
    plot_market_performance_overview(summary, plot_file, dpi=SAVE_FIG_DPI)
    
    print("Analysis complete!")
