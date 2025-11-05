# MWU-on-GANs-for-Financial-Markets

A research project implementing Multiplicative Weights Update (MWU) algorithm on multiple GAN architectures for financial market anomaly detection and algorithmic trading.

## Overview

This project combines deep learning (GANs) with online learning algorithms (MWU) to detect anomalies in financial time series data and generate profitable trading signals. The system trains multiple GAN architectures as "experts" that learn to detect market anomalies, then uses the MWU algorithm to dynamically weight these experts' predictions for optimal trading performance.

### Key Features 

- **Multiple GAN Architectures:** Paper GAN, LSTM-CNN Sequential, LSTM-CNN Parallel, and CNN-LSTM Sequential
- **Dual Input Types:** Price-only and OHLCAV (Open, High, Low, Close, Adjusted Close, Volume)
- **Online Learning:** MWU algorithm with regret bounds for expert weighting
- **Hyperparameter Optimization:** Optuna-based Bayesian optimization
- **Comprehensive Backtesting:** VectorBT-powered portfolio simulation
- **SLURM Support:** Designed for HPC cluster execution with MPI parallelization

## Project Structure

```
MWU-on-GANs-for-Financial-Markets/
├── common/                          # Shared utilities
│   ├── anomaly_detector.py          # Unified anomaly detection interface
│   ├── backtest_common.py           # VectorBT backtesting engine
│   ├── data_utils.py                # Data loading and preprocessing
│   ├── io_utils.py                  # Configuration and file I/O
│   ├── logger_utils.py              # Logging utilities
│   ├── optuna_backtest.py           # Trading signal optimization
│   ├── optuna_common.py             # Base Optuna optimizer
│   └── plot_utils.py                # Visualization utilities
│
├── paper_gan_price/                 # Paper GAN (price-only)
│   ├── gan_model.py                 # Model implementation
│   └── gan_train_optuna.py          # Training and optimization
│
├── paper_gan_ohlcav/                # Paper GAN (OHLCAV features)
│   ├── gan_model.py
│   └── gan_train_optuna.py
│
├── cnn_lstm_seq_price/              # CNN→LSTM Sequential (price)
│   ├── cnn_lstm_gan_model_sequential.py
│   └── gan_train_optuna.py
│
├── cnn_lstm_seq_ohlcav/             # CNN→LSTM Sequential (OHLCAV)
│   ├── cnn_lstm_gan_model_sequential.py
│   └── gan_train_optuna.py
│
├── lstm_cnn_seq_price/              # LSTM→CNN Sequential (price)
│   ├── lstm_cnn_gan_model_sequential.py
│   └── gan_train_optuna.py
│
├── lstm_cnn_seq_ohlcav/             # LSTM→CNN Sequential (OHLCAV)
│   ├── lstm_cnn_gan_model_sequential.py
│   └── gan_train_optuna.py
│
├── lstm_cnn_parallel_price/         # LSTM-CNN Parallel (price)
│   ├── lstm_cnn_gan_model_parallel.py
│   └── gan_train_optuna.py
│
├── lstm_cnn_parallel_ohlcav/        # LSTM-CNN Parallel (OHLCAV)
│   ├── lstm_cnn_gan_model_parallel.py
│   └── gan_train_optuna.py
|
├── mwu_price/                       # MWU algorithm (price-only)
│   ├── hedge_algorithm_python.py    # Core MWU implementation
│   └── hedge_algorithm_optuna.py    # MWU hyperparameter tuning
│
├── mwu_ohlcav/                      # MWU algorithm (OHLCAV)
│   ├── hedge_algorithm_python.py
│   └── hedge_algorithm_optuna.py
|
├── data/                            # Data used
│   ├── lstm_cnn_gan_model_parallel.py
│   └── gan_train_optuna.py
|
├── anomaly_backtest_vectorbt.py     # Main backtesting script
├── workflow.sh                      # End-to-end SLURM workflow
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended)
- SLURM cluster (optional, for distributed training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MWU-on-GANs-for-Financial-Markets.git
cd MWU-on-GANs-for-Financial-Markets
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Prepare your financial data as CSV files with the following columns:

**For price-only models:**
- `date`: Timestamp (epoch seconds, milliseconds, or 'YYYY-MM-DD')
- `log_adj_close`: Log-transformed adjusted close price
- `adj_close`: Adjusted close price

**For OHLCAV models:**
- `date`: Timestamp
- `open`, `high`, `low`, `close`, `adj_close`, `volume`
- `log_adj_close`: Log-transformed adjusted close

### 2. Training Individual GAN Models

#### Train with default parameters:
```bash
python -m paper_gan_price.gan_train_optuna train \
    --tickers AAPL \
    --data-dir ~/Data/OHLC \
    --out-dir ./runs
```

#### Hyperparameter optimization:
```bash
python -m paper_gan_price.gan_train_optuna optimize \
    --ticker AAPL \
    --data-dir ~/Data/OHLC \
    --out-dir ./optuna_results \
    --n-trials 50 \
    --timeout 3600 \
    --train-best
```

### 3. Single-Model Backtesting

Test a single GAN model's anomaly detection trading strategy:

```bash
python anomaly_backtest_vectorbt.py \
    --input data/AAPL_backtest.csv \
    --output ./results \
    --model-path ./models/paper_gan_price \
    --model-type paper_gan_price \
    --start-date 2018-01-01 \
    --end-date 2023-12-31 \
    --short-ma 10 \
    --long-ma 50
```

#### With optimization:
```bash
python anomaly_backtest_vectorbt.py \
    --input data/AAPL_backtest.csv \
    --output ./results \
    --model-path ./models/paper_gan_price \
    --model-type paper_gan_price \
    --optimize \
    --n-trials 100 \
    --start-date 2018-01-01 \
    --end-date 2023-12-31
```

### 4. Multi-Expert MWU Trading

Combine multiple GAN experts using the MWU algorithm:

```bash
python -m mwu_price.hedge_algorithm_python \
    --input data/AAPL_backtest.csv \
    --output ./mwu_results \
    --paper-gan-model models/paper_gan_price \
    --lstm-cnn-gan-model models/lstm_cnn_parallel_price \
    --start 2018-01-01 \
    --end 2023-12-31 \
    --max-iterations 50 \
    --short-window 5 \
    --long-window 20
```

#### MWU hyperparameter tuning:
```bash
python -m mwu_price.hedge_algorithm_optuna \
    --input data/AAPL_backtest.csv \
    --output ./mwu_optuna \
    --paper-gan-model models/paper_gan_price \
    --lstm-cnn-gan-model models/lstm_cnn_parallel_price \
    --n-trials 50 \
    --start 2018-01-01 \
    --end 2023-12-31
```

### 5. Complete Workflow (SLURM)

Run the entire pipeline on a SLURM cluster:

```bash
sbatch workflow.sh paper_gan_price
```

This workflow:
1. Optimizes GAN hyperparameters with Optuna
2. Trains the best model
3. Optimizes trading signal parameters
4. Performs final backtesting

## Model Architectures

### Paper GAN
- **Generator**: MLP-CNN with upsampling (LeakyReLU activation)
- **Discriminator**: CNN with spectral normalization
- **Key Parameters**: 
  - Base channels: 16 (G), 64 (D)
  - Kernel sizes: [10, 5, 3, 2] (G), [3, 3, 3, 3] (D)
  - Learning rates: 1.5e-4 (G), 1e-5 (D)

### LSTM-CNN Parallel
- **Generator**: Parallel LSTM and CNN branches with fusion
- **Discriminator**: Parallel CNN and LSTM feature extraction
- **Key Features**: 
  - Residual connections
  - Squeeze-and-Excitation blocks
  - Channel-wise dropout

### Sequential Architectures
Similar to parallel but with sequential processing of LSTM → CNN or CNN → LSTM.

## Anomaly Detection Method

The project uses **TAnoGAN (Time-series Anomaly GAN)**:

1. **Training**: GANs learn the distribution of normal market behavior
2. **Detection**: For each test window:
   - Find optimal latent code `z` that reconstructs the input
   - Compute residual loss: `L_R = ||x - G(z)||`
   - Compute discrimination loss: `L_D = ||D(x) - D(G(z))||`
   - Final anomaly score: `L = (1-λ)L_R + λL_D` (λ=0.9)
3. **Signals**: Generate trading signals from anomaly score moving averages

## MWU Algorithm

The Multiplicative Weights Update algorithm dynamically weights expert predictions:

```
Initialize: w_i = 1/N for all experts
For each time period t:
  1. Get expert predictions and compute losses l_i(t)
  2. Update weights: w_i(t+1) = w_i(t) * exp(-η * l_i(t))
  3. Normalize weights: w_i(t+1) = w_i(t+1) / Σw_j(t+1)
  
Learning rate: η = sqrt(2 ln(N) / T) (adaptive)
Regret bound: R(T) ≤ sqrt(2T ln(N))
```

## Output Structure

Each run produces:

```
results/
├── backtest_results.json          # Performance metrics
├── backtest_plots.png             # Visualization
├── daily_data.csv                 # Per-day statistics
├── trades.csv                     # Trade log
├── config.json                    # Configuration used
└── optimization_results.json      # Optuna results (if optimizing)

mwu_results/
├── overall_results.json           # Combined performance
├── final_weights.json             # Expert weights
├── iteration_details.json         # Per-iteration statistics
├── online_mwu_results.png         # Strategy visualization
└── online_mwu_weight_evolution.png # Weight evolution plot
```

## Key Metrics

- **Total Return**: Strategy cumulative return
- **Sharpe Ratio**: Risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Calmar Ratio**: Return / Maximum Drawdown
- **Regret**: MWU algorithm regret vs. best expert

## Configuration

### Model Training Parameters
- `sequence_length`: Time window size (default: 25)
- `latent_dim`: GAN latent space dimension (default: 100)
- `epochs`: Training epochs (default: 200)
- `batch_size`: Batch size (default: 64)
- `scaler_type`: Data scaling method (robust/standard/minmax)

### Trading Parameters
- `short_window`: Short MA window for anomaly scores (default: 5)
- `long_window`: Long MA window for anomaly scores (default: 20)
- `initial_capital`: Starting portfolio value (default: 100,000)
- `transaction_cost`: Trading fee fraction (default: 0.005)

### MWU Parameters
- `learning_rate`: Fixed η or None for adaptive (default: None)
- `max_loss`: Maximum loss per round (default: 1.0)
- `max_iterations`: Number of MWU update rounds (default: 50)

## References

- Schlegl et al. (2019). "f-AnoGAN: Fast unsupervised anomaly detection with generative adversarial networks"
- Arora et al. (2012). "The Multiplicative Weights Update Method"
- VectorBT Documentation: https://vectorbt.dev/

## Acknowledgments

- TAnoGAN methodology for anomaly detection
- VectorBT for efficient backtesting
- Optuna for hyperparameter optimization
- PyTorch for deep learning framework

---

**Note**: This is research code. Always validate results and perform proper risk management before using in live trading.