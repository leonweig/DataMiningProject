#use a seed for reproducibility of randomness
SEED = 42

#paths
DATA_DIR = "data"
RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"
ZIP_PATH = "binance-futures-market-data-1-minute-frequency.zip"

#we only run this code for January 2022
DATE_SLICE = "2022"
N_CLUSTERS = 3
IF_CONTAM = 'auto'
LOF_CONTAM = 'auto'

#forecast parameters
ROLLING_WINDOW_MIN = 1440 * 7
HORIZON_MIN = 30

#graphs
SAVE_FIG_DPI = 200  
