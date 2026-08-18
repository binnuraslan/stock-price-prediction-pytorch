# Stock Price Prediction with PyTorch (LSTM vs GRU)

A beginner machine learning project that predicts Amazon (AMZN) stock closing prices using two recurrent neural network architectures — **LSTM** and **GRU** — implemented in PyTorch, and compares their performance.

This project follows the general approach described in [Stock price prediction with PyTorch](https://medium.com/swlh/stock-price-prediction-with-pytorch-37f52ae84632) by Rodolfo Saldanha.

## Overview

- **Task**: time series regression — predict the next day's closing price from the previous N days
- **Data**: AMZN daily closing price, ~14 years (2012–present), fetched via [yfinance](https://pypi.org/project/yfinance/)
- **Models**: a 2-layer LSTM and a 2-layer GRU, trained and evaluated under the same conditions
- **Metric**: MSE / RMSE on a held-out test set

## Method

1. **Data collection**: daily OHLCV data for AMZN downloaded with `yfinance`, using `auto_adjust=True` to correct for the 2022 stock split.
2. **Preprocessing**:
   - Only the `Close` price is used.
   - Missing rows are dropped (`dropna`).
   - Prices are scaled to `[-1, 1]` with `MinMaxScaler`.
3. **Sliding window**: a lookback window of **20 days** is used to predict day 21 (sequences generated with a custom `create_sequences` function).
4. **Train/test split**: 80% / 20%, split **chronologically** (no shuffling) to avoid look-ahead bias — the model is only ever tested on data that comes after its training period.
5. **Models**: both models share the same hyperparameters — `input_dim=1`, `hidden_dim=32`, `num_layers=2`, `output_dim=1` — so the comparison isolates the effect of the architecture (LSTM vs GRU) rather than model capacity.
6. **Training**: MSE loss, Adam optimizer (`lr=0.01`). The LSTM was trained for 300 epochs and the GRU for 100 — the LSTM required more epochs to converge to a comparable loss, likely due to its extra gate (and therefore more parameters to learn).

## Results

| Model | Test MSE | Test RMSE | Training time |
|-------|---------:|----------:|---------------:|
| LSTM  | 165.72   | 12.87     | — |
| GRU   | 100.26   | 10.01     | ~5.5s |

**GRU outperformed LSTM** on this dataset, both in accuracy and training speed — consistent with the reference article's findings. This is likely because GRU's simpler gating mechanism (no separate cell state, no output gate) has fewer parameters to learn, so it converges faster for a comparable amount of training.

Both models track the overall trend of the stock reasonably well, but — as expected for stock price prediction — they lag behind sharp, sudden moves rather than anticipating them.

## Project structure

```
.
├── stockpricepredictionwithPyTorch.ipynb   # main notebook (data, models, training, evaluation)
├── README.md
└── requirements.txt
```

## How to run

1. Clone this repository.
2. Create a virtual environment and install the dependencies:
   ```bash
   python3 -m venv ml_env
   source ml_env/bin/activate   # Windows: ml_env\Scripts\activate
   pip3 install -r requirements.txt
   ```
3. Launch Jupyter and run the notebook top to bottom:
   ```bash
   jupyter notebook
   ```
4. Open `stockpricepredictionwithPyTorch.ipynb`, then **Kernel → Restart Kernel and Run All Cells**.

Results will vary slightly run to run unless a random seed is fixed (this notebook fixes it with `torch.manual_seed(42)` before each model is created).

## Limitations & reflection

Stock price prediction from historical prices alone is a genuinely hard problem — real markets are influenced by countless factors (news, macroeconomics, sentiment) that aren't present in this dataset. The relatively low RMSE here mostly reflects that the model has learned to track the recent trend, not that it can anticipate future moves. This project was built for learning purposes (time series preprocessing, RNNs, and PyTorch training loops), not as a trading tool.

## Acknowledgements

- Methodology reference: [Rodolfo Saldanha — "Stock price prediction with PyTorch"](https://medium.com/swlh/stock-price-prediction-with-pytorch-37f52ae84632)
- Data: [Yahoo Finance](https://finance.yahoo.com/) via the `yfinance` Python package
