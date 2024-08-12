# Evaluation of Technical Indicators in High-Frequency Trading Data

This repository contains the code and resources for evaluating the effectiveness of various technical indicators when used as input variables for machine learning (ML) models in high-frequency trading (HFT) data. The project assesses whether these indicators add predictive value or introduce noise into the models.

## Table of Contents
- [Introduction](#introduction)
- [Data Preparation](#data-preparation)
- [Technical Indicators](#technical-indicators)
- [Machine Learning Models](#machine-learning-models)
- [Methodology](#methodology)
- [Preliminary Results](#preliminary-results)
- [Conclusion and Next Steps](#conclusion-and-next-steps)
- [Requirements](#requirements)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this research is to evaluate the effectiveness of technical indicators when used as input variables for ML models in high-frequency trading data. Specifically, we aim to test the hypothesis presented by Lo et al. in their paper *Foundations of Technical Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation* to determine if these indicators add predictive value or introduce noise.

## Data Preparation

For this study, one quarter of high-frequency stock data (e.g., 1-minute intervals) is used. The dataset includes the following features:
- Open, High, Low, Close prices
- Volume

## Technical Indicators

The following technical indicators are computed on the high-frequency data:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
- Moving Average Convergence Divergence (MACD)
- Relative Strength Index (RSI)
- Bollinger Bands
- Stochastic Oscillator
- Average Directional Index (ADX)
- Commodity Channel Index (CCI)
- On-Balance Volume (OBV)
- Fibonacci Retracement

## Machine Learning Models

The performance of the following ML models for time series financial forecasting is evaluated:
- Random Forest
- Gradient Boosting Machines (GBM)
- Support Vector Machines (SVM)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Convolutional Neural Networks (CNN)
- ARIMA
- Prophet
- XGBoost
- Transformer Models

## Methodology

### Baseline Models

Models are trained and evaluated using only the raw stock data (Open, High, Low, Close, Volume). The models are assessed using appropriate metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared.

### Models with Technical Indicators

Next, the models are trained and evaluated using the enhanced dataset, which includes both the raw stock data and the computed technical indicators. The same evaluation metrics are used.

### Hypothesis Testing

To determine if the technical indicators add value, we compare the performance of the models with and without the technical indicators. Statistical tests, such as paired t-tests, are conducted to assess the significance of the differences in performance.

## Preliminary Results

A preliminary test run using Random Forest Regressor (RFR) models with some popular technical indicators has been completed. The table below summarizes the plan for testing various ML models with different technical indicators.

| **Technical Indicator / Model** | **Random Forest** | **GBM** | **SVM** | **LSTM** | **GRU** | **CNN** | **ARIMA** | **Prophet** | **XGBoost** | **Transformer** |
|---------------------------------|-------------------|---------|---------|----------|---------|----------|------------|-------------|-------------|-----------------|
| **SMA**                         | ✔                | ✔       | ✔       | ✔        | ✔       | ✔        | ✔          | ✔           | ✔           | ✔               |
| **EMA**                         | ✔                | ✔       | ✔       | ✔        | ✔       | ✔        | ✔          | ✔           | ✔           | ✔               |
| **MACD**                        | ✔                | ✔       | ✔       | ✔        | ✔       | ✔        | ✔          | ✔           | ✔           | ✔               |
| **RSI**                         | ✔                | ✔       | ✔       | ✔        | ✔       | ✔        | ✔          | ✔           | ✔           | ✔               |
| **Bollinger Bands**             | ✔                | ✔       | ✔       | ✔        | ✔       | ✔        | ✔          | ✔           | ✔           | ✔               |
| **Stochastic Oscillator**       | ✔                | ✔       | ✔       | ✔        | ✔       | ✔        | ✔          | ✔           | ✔           | ✔               |
| **ADX**                         | ✔                | ✔       | ✔       | ✔        | ✔       | ✔        | ✔          | ✔           | ✔           | ✔               |
| **CCI**                         | ✔                | ✔       | ✔       | ✔        | ✔       | ✔        | ✔          | ✔           | ✔           | ✔               |
| **OBV**                         | ✔                | ✔       | ✔       | ✔        | ✔       | ✔        | ✔          | ✔           | ✔           | ✔               |
| **Fibonacci Retracement**       | ✔                | ✔       | ✔       | ✔        | ✔       | ✔        | ✔          | ✔           | ✔           | ✔               |

## Conclusion and Next Steps

The initial findings suggest potential value in incorporating technical indicators into ML models for high-frequency trading data. The next steps involve a comprehensive evaluation across all selected ML models and technical indicators, followed by a detailed analysis of the results.

## Requirements

To run this project, you will need the following libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

## Setup and Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/hft-technical-indicators.git
    cd hft-technical-indicators
    ```

2. **Install the required libraries**:
    You can install the necessary libraries using `pip`:
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```

## Usage

1. **Open the Jupyter notebook**:
    ```bash
    jupyter notebook hft_technical_indicators.ipynb
    ```

2. **Run the notebook**:
    Execute the cells in the notebook to load data, compute technical indicators, train the models, and visualize the results.

## Project Structure

- `hft_technical_indicators.ipynb`: The main Jupyter notebook containing the entire workflow.
- `q1_filtered_dataset.csv`: The dataset used for training and testing the models.

## Results

The notebook includes visualizations of the models' performance, including plots comparing actual vs. predicted stock prices for both training and validation datasets. Performance metrics such as MSE, MAE, RMSE, and R² are also calculated and displayed.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or new features to suggest.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
