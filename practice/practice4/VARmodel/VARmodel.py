import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error
import os
from datetime import datetime


# Cấu hình log sẽ được thiết lập trong main() sau khi tạo thư mục logs/images.



# Read data
def load_data(file_path):
    df = pd.read_csv(file_path)

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')
    df = df.sort_index()
    
    logging.info("Data loaded successfully")
    logging.info(f"Shape: {df.shape}")
    logging.info(f"Columns: {list(df.columns)}")
    
    return df

# StandardScaler
def standardize_data(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
    
    df_scaled = pd.DataFrame(
        scaled,
        index=df.index,
        columns=df.columns
    )

    logging.info("StandardScaler applied to all variables")
    return df_scaled, scaler

# Split Train / Val / Test (time order, no shuffle)
def split_time(df, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    logging.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


# ADF test
def adf_test(series, name):
    p_value = adfuller(series)[1]
    logging.info(f"ADF p-value ({name}): {p_value:.4f}")
    return p_value

def adf_check_all(df, max_d=1, alpha=0.05):
    current = df.copy()
    for d in range(0, max_d + 1):
        logging.info(f"===== ADF TEST (d={d}) =====")
        pvals = {}
        for col in current.columns:
            pvals[col] = adf_test(current[col], f"{col}")
        non_stationary = [k for k, v in pvals.items() if v >= alpha]
        if not non_stationary:
            logging.info(f"All variables stationary at d={d}")
            return current, d
        if d < max_d:
            logging.info(f"Non-stationary at d={d}: {non_stationary}")
            logging.info("Applying first-order differencing")
            current = current.diff().dropna()
    return current, max_d

# Differencing
def diff_data(df):
    logging.info("Applying first-order differencing")
    return df.diff().dropna()

# Choose lag by AIC
def var_lag(train_diff, images_dir, max_lag=24):
    model = VAR(train_diff)
    aic_values = []

    for lag in range(1, max_lag + 1):
        result = model.fit(lag)
        aic_values.append(result.aic)
        logging.info(f"AIC (lag={lag}): {result.aic:.6f}")

    best_lag = np.argmin(aic_values) + 1

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, max_lag + 1), aic_values, marker='o')
    plt.title("VAR Lag Selection (AIC)")
    plt.savefig(os.path.join(images_dir, 'var_lag_selection.png'))
    plt.close()

    
    logging.info(f"Best lag selected by AIC: {best_lag}")
    return best_lag

# Train VAR model
def train(train_diff, lag):
    model = VAR(train_diff)
    result = model.fit(lag)
    logging.info("VAR model trained successfully")
    return result

# Walk-forward Forecast
def walk_forward_forecast(model, history_diff, test_diff):
    forecasts = []
    history = history_diff.copy()

    for idx in test_diff.index:
        yhat = model.forecast(history.values[-model.k_ar:], steps=1)[0]
        forecasts.append(yhat)
        # Append actual diff (walk-forward with observed data)
        history = pd.concat([history, test_diff.loc[[idx]]], axis=0)

    return pd.DataFrame(
        forecasts,
        index=test_diff.index,
        columns=history_diff.columns
    )

# Hoàn nguyên differencing
def invert_diff(train_original, forecast_diff):
    last_value = train_original['load'].iloc[-1]
    return last_value + forecast_diff['load'].cumsum()

def inverse_scale_load(series, scaler, load_index):
    return series * scaler.scale_[load_index] + scaler.mean_[load_index]

# visulize
def plot_prediction(actual, predicted, images_dir):
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label="Actual Load", color='blue', alpha=0.7)
    plt.plot(predicted.index, predicted, '--', label="VAR Prediction", color='red')
    plt.title("VAR Model – Electricity Load: Actual vs Predicted")
    plt.xlabel("Time")
    plt.ylabel("Load Value")
    plt.legend()
    plt.grid(True)
    
    # Lưu vào thư mục images (VARmodel/images/)
    save_path = os.path.join(images_dir, 'var_prediction.png')
    plt.savefig(save_path)
    logging.info(f"Prediction plot saved to {save_path}")
    plt.close() # Quan trọng: Giải phóng bộ nhớ và tránh đè ảnh

def plot_residuals(actual, predicted, images_dir):
    residuals = actual - predicted
    plt.figure(figsize=(12, 4))
    plt.plot(residuals, color='purple', linewidth=1)
    plt.axhline(y=0, color='black', linestyle='-')
    plt.title("VAR Model Residuals (Error)")
    plt.grid(True)
    
    # Lưu vào thư mục images
    save_path = os.path.join(images_dir, 'var_residuals.png')
    plt.savefig(save_path)
    logging.info(f"Residuals plot saved to {save_path}")
    plt.close()

def var_lag_plot(aic_values, max_lag, images_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_lag + 1), aic_values, marker='o', linestyle='--', color='green')
    plt.xlabel("Lag Order")
    plt.ylabel("AIC Value")
    plt.title("VAR Lag Selection - AIC Scores")
    plt.grid(True)
    
    save_path = os.path.join(images_dir, 'var_lag_selection.png')
    plt.savefig(save_path)
    plt.close()

# Metrics
def evaluate_var(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"MAE : {mae:.4f}")
    logging.info(f"MAPE: {mape:.2f}%")

# Main
def main():
    # 1. Load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    logs_dir = os.path.join(script_dir, 'logs')
    images_dir = os.path.join(script_dir, 'images')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    log_filename = f"var_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - VAR - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, log_filename)),
            logging.StreamHandler()
        ],
        force=True
    )

    data_path = os.path.abspath(os.path.join(script_dir, "..", "dataclean.csv"))
    df = load_data(data_path)

    # 2. Select variables for VAR
    var_cols = ['load', 'temp', 'humidity', 'wind_speed']
    df_var = df[var_cols].copy()

    # 3. Standardize
    df_var_scaled, scaler_var = standardize_data(df_var)

    # 4. Split Train / Val / Test
    train_var, val_var, test_var = split_time(df_var_scaled, train_ratio=0.7, val_ratio=0.15)

    # 5. ADF test with iterative differencing
    stationary_df, used_d = adf_check_all(train_var, max_d=1, alpha=0.05)

    # 6. Differencing (use full diff to keep continuity at split boundary)
    full_diff = diff_data(df_var_scaled) if used_d == 1 else df_var_scaled.copy()
    train_diff = full_diff.loc[train_var.index[1:]]
    val_diff = full_diff.loc[val_var.index]
    test_diff = full_diff.loc[test_var.index]

    # 7. Select lag by AIC
    best_lag = var_lag(train_diff, images_dir)

    # 8. Train VAR on Train + Val
    train_val_diff = pd.concat([train_diff, val_diff], axis=0)
    train_val_var = pd.concat([train_var, val_var], axis=0)
    var_model = train(train_val_diff, best_lag)

    # 9. Walk-forward Forecast on test set
    forecast_diff = walk_forward_forecast(
        var_model,
        train_val_diff,
        test_diff
    )

    # 10. Invert differencing (scaled space) and inverse StandardScaler to original
    load_pred_scaled = invert_diff(train_val_var, forecast_diff)
    load_actual_scaled = test_var['load']
    load_index = var_cols.index('load')
    load_pred = inverse_scale_load(load_pred_scaled, scaler_var, load_index)
    load_actual = inverse_scale_load(load_actual_scaled, scaler_var, load_index)

    # 11. Visualization
    plot_prediction(load_actual, load_pred, images_dir)
    plot_residuals(load_actual, load_pred, images_dir)

    # 12. Evaluation
    logging.info("===== VAR EVALUATION =====")
    evaluate_var(load_actual, load_pred)


if __name__ == "__main__":
    main()
