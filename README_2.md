# Stock Price Prediction with PyTorch (LSTM vs GRU)

*Diğer diller: [Türkçe](README.tr.md)*

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
   - Prices are scaled to `[-1, 1]` with `MinMaxScaler`, **fit only on the training portion of the raw series** and then applied to the full series — the test period is never seen during fitting, so no future information leaks into the scaling step.
3. **Sliding window**: a lookback window of **20 days** is used to predict day 21 (sequences generated with a custom `create_sequences` function).
4. **Train/test split**: 80% / 20%, split **chronologically** (no shuffling) to avoid look-ahead bias — the model is only ever tested on data that comes after its training period.
5. **Models**: both models share the same hyperparameters — `input_dim=1`, `hidden_dim=32`, `num_layers=2`, `output_dim=1` — so the comparison isolates the effect of the architecture (LSTM vs GRU) rather than model capacity.
6. **Training**: MSE loss, Adam optimizer (`lr=0.01`). The LSTM was trained for 300 epochs and the GRU for 100 — the LSTM required more epochs to converge to a comparable loss, likely due to its extra gate (and therefore more parameters to learn).

## Results

| Model | Test MSE | Test RMSE | Training time |
|-------|---------:|----------:|---------------:|
| LSTM  | 83.69    | 9.15      | — |
| GRU   | 195.68   | 13.99     | ~5.5s |
| **Persistence baseline** (naive "tomorrow = today") | **17.62** | **4.20** | 0.00s |

The **persistence baseline clearly beats both LSTM and GRU** on this dataset. A model that does nothing but repeat today's price is a hard bar to clear for a stock this trend-heavy and autocorrelated — both RNNs learned to track the general shape of the series, but their extra flexibility let them drift further from the very next value than the naive guess does. This is a common, humbling result in stock price prediction, and it's a more honest takeaway than the RMSE numbers alone: **neither model demonstrated real predictive value over the trivial baseline.**

Between the two RNNs, LSTM outperformed GRU here — the opposite of an earlier run of this same project, before the scaling was made leakage-safe (fitting `MinMaxScaler` on the full series instead of the training period only). That reversal is itself a useful finding: which architecture "wins" was sensitive to a preprocessing detail, not just the number of epochs or hidden units. This is a reminder not to over-read a single comparison — see the note on [AI Snake Oil](https://www.aisnakeoil.com/) below.

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

Stock price prediction from historical prices alone is a genuinely hard problem — real markets are influenced by countless factors (news, macroeconomics, sentiment) that aren't present in this dataset. The RMSE values here mostly reflect how closely each model tracks the recent trend, not any real predictive power: as the results above show, a trivial "tomorrow = today" baseline beat both trained models. A low error number on its own doesn't mean a model has learned something useful — it needs to be measured against a baseline to mean anything, and even then, results can flip based on preprocessing choices that have nothing to do with model "intelligence." This project was built for learning purposes (time series preprocessing, RNNs, and PyTorch training loops), not as a trading tool, and this result is a good reminder to stay skeptical of headline accuracy numbers in general — see Narayanan & Kapoor's [AI Snake Oil](https://www.aisnakeoil.com/) for more on this.

## Acknowledgements

- Methodology reference: [Rodolfo Saldanha — "Stock price prediction with PyTorch"](https://medium.com/swlh/stock-price-prediction-with-pytorch-37f52ae84632)
- Data: [Yahoo Finance](https://finance.yahoo.com/) via the `yfinance` Python package
- Critical perspective on AI predictive claims: Narayanan & Kapoor, [AI Snake Oil](https://www.aisnakeoil.com/)
