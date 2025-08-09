Time Series Forecasting with LSTM & GRU

This project implements deep learning models (LSTM and GRU) to forecast time series data, with configurable preprocessing, training, evaluation, and forecasting scripts.
Features

    - Data Preparation: Sliding window creation from raw series (single or multi-ticker).
    - Model Training: LSTM and GRU architectures with early stopping and model checkpointing.
    - Evaluation: Residual analysis, prediction-vs-actual plots, and error metrics.

Usage
1. Prepare Windows from Series

python -m src.prepare_windows \
  --from-long data/processed/series_long.csv \
  --ticker AAPL \
  --win 30 \
  --horizon 1

2. Train a Model

Example for LSTM:

python -m src.train_lstm \
  --data data/processed/dataset.npz \
  --out-dir output

Example for GRU:

python -m src.train_gru \
  --data data/processed/dataset.npz \
  --out-dir output

3. Evaluate Model

python -m src.evaluate \
  --model-path output/lstm_best.keras \
  --data data/processed/dataset.npz

4. Forecast Future Values

python -m src.forecast \
  --model-path output/lstm_best.keras \
  --series data/processed/series_long.csv \
  --ticker AAPL \
  --n-steps 20 \
  --freq B \
  --start-date 2023-01-01 \
  --history-points -1 \
  --out-dir output/forecast

Requirements
    Python 3.9+
    TensorFlow 2.15+
    NumPy
    Pandas
    Matplotlib
    Seaborn
    scikit-learn