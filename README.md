# Analyzing Structural Anomalies in Cryptocurrency Futures

## Description 
This project aimed to find manipulative pump and dump events, and structural anomalies in Cryptocurrency Futures. The data used is from the "Binance USDT-MArgined Futures Market Data - 1-minute Frequency" (Kaggle, 2024)(https://www.kaggle.com/datasets/siavashraz/binance-futures-market-data-1-minute-frequency). The data considered were from the top 10 cryptocurrencies (BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, SOLUSDT, XRPUSDT, DOTUSDT, DOGEUSDT, AVAXUSDT, MATICUSDT) and for the month of January 2022. The range can be changed in the config.py file. To run this project python with following libaries is needed: pandas, numpy, matplotlib, scikit-learn, arch

## Questions and answers
The project tried to figure out if we can forecast 30 min volatility from 1 min futures data, which turned out to be partially ture. Unsupervised anomalies flag manipulation like events showed some overlap with low precision. We also concluded that anomaly informed signals only help trading marginally and inconsistently. 

## Final project paper


## Video demonstration

