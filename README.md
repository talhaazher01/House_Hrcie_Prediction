# House Price Prediction

This repository contains a machine learning project that predicts house prices based on various features such as size, location, number of rooms, and more. The project uses different regression models and techniques to provide an accurate prediction for house prices.

## Project Overview

The goal of this project is to predict the house prices in Bengaluru (or any other dataset) using machine learning algorithms. It involves the following steps:

1. Data Collection
2. Data Preprocessing
3. Feature Engineering
4. Model Selection
5. Model Evaluation
6. Saving the Model for Future Use

### Key Features:
- **Exploratory Data Analysis (EDA)**: Understand the relationship between different variables and house prices.
- **Multiple Models**: Linear Regression, XGBoost, etc.
- **Saved Models**: Models are saved as `.pkl` files for future prediction.

## Dataset

The dataset used in this project is `Bengaluru_House_Data.csv`, which contains various features like:
- `Location`
- `Size`
- `Price`
- `Area Type`
- `Availability`
- `Total Square Feet`
- `Balcony`
- `Bathroom`

## Files in this Repository

- **`Bengaluru_House_Data.csv`**: The dataset used for training the model.
- **`main.py`**: The main script for data preprocessing, model training, and evaluation.
- **`beng.ipynb`**: Jupyter Notebook containing exploratory data analysis and model building steps.
- **`linear_regression_model.pkl`**: The saved Linear Regression model.
- **`xg.pkl`**: The saved XGBoost model.
- **`feature_matrix.pkl`**: Feature matrix used for training the models.

## Installation and Setup

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/talhaazher01/House_Price_Prediction.git
   cd House_Price_Prediction
