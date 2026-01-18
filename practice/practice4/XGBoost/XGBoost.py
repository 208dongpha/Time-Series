import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error, mean_absolute_percentage_error

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

# Standart scaler
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

# Feature engineering
def create_feature(df, lags=24):
    df_lag = df.copy()

    # Tạo đặc trưng trễ cho từng biến để lấy thông tin L giờ trước.
    for lag in range(1, lags + 1):
        for col in df.columns:
            df_lag[f"{col}_lag_{lag}"] = df_lag[col].shift(lag)
    
    logging.info(f"Lag features created (window={lags})")
    return df_lag

# Rolling features
def create_rolling(df):
    df_roll = df.copy()

    
    # Rolling chỉ dùng dữ liệu quá khứ để tránh leakage.
    df_roll['load_roll_mean_6'] = df['load'].rolling(6).mean().shift(1) # load.shift(1) tại thời điểm t sẽ lấy giá trị của t‑1.
    df_roll['load_roll_std_24'] = df['load'].rolling(24).std().shift(1)
    df_roll['load_trend_24'] = df['load'].shift(1) - df['load'].shift(25)

    logging.info("Rolling features created")
    return df_roll

# Combine features
def prepare_features(df):
    df_feat = create_feature(df, lags=24)
    df_feat = create_rolling(df_feat)
    df_feat["target"] = df_feat["load"].shift(-1) # Target tại thời điểm t là load ở thời điểm t+1.
    df_feat = df_feat.dropna()

    logging.info(f"Final feature shape: {df_feat.shape}")
    return df_feat

#split train
def split(df, train_ratio=0.7, val_ratio=0.15):
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    logging.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test

# X split
def split_xy(df):
    X = df.drop(columns=['target'])
    y = df['target']
    return X, y

# train
def train_model(X_train, y_train, X_val, y_val):
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        eval_metric="rmse",
        random_state=42
    )

    model.fit(
        X_train, y_train,
        eval_set = [(X_val, y_val)],
        verbose = False
    )

    logging.info("XGBoost model trained successfully")
    return model

# Inverse scale load
def inverse_scale_load(series, scaler, load_index):
    return series * scaler.scale_[load_index] + scaler.mean_[load_index]

# Visualization
def plot_prediction(actual, predicted, images_dir, suffix="test"):
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label="Actual Load")
    plt.plot(predicted.index, predicted, '--', label="XGBoost Prediction")
    plt.legend()
    plt.grid()

    path = os.path.join(images_dir, f"xgb_prediction_{suffix}.png")
    plt.savefig(path)
    plt.close()

    logging.info(f"Prediction plot saved to {path}")

def plot_residuals(actual, predicted, images_dir, suffix="test"):
    residuals = actual - predicted
    plt.figure(figsize=(12, 4))
    plt.plot(residuals, color="purple", linewidth=1)
    plt.axhline(y=0, color="black", linestyle="-")
    plt.title("XGBoost Residuals (Error)")
    plt.grid(True)
    path = os.path.join(images_dir, f"xgb_residuals_{suffix}.png")
    plt.savefig(path)
    plt.close()
    logging.info(f"Residuals plot saved to {path}")

# Metrics
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"MAE : {mae:.4f}")
    logging.info(f"MAPE: {mape:.2f}%")


def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(script_dir, "logs")
    images_dir = os.path.join(script_dir, "images")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Logging
    log_file = f"xgb_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - XGBOOST - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, log_file)),
            logging.StreamHandler()
        ],
        force=True
    )
    logging.info(f"Log file: {os.path.join(logs_dir, log_file)}")

    # Load data
    data_path = os.path.abspath(os.path.join(script_dir, "..", "dataclean.csv"))
    df = load_data(data_path)

    # Select columns
    cols = ['load', 'temp', 'humidity', 'wind_speed']
    df_scaled, scaler = standardize_data(df[cols])

    # Feature engineering
    df_feat = prepare_features(df_scaled)

    # Split
    train, val, test = split(df_feat)

    X_train, y_train = split_xy(train)
    X_val, y_val = split_xy(val)
    X_test, y_test = split_xy(test)

    # Train
    model = train_model(X_train, y_train, X_val, y_val)

    # Predict
    y_pred_train = pd.Series(model.predict(X_train), index=y_train.index)
    y_pred_test = pd.Series(model.predict(X_test), index=y_test.index)

    # Inverse scale
    load_idx = cols.index('load')
    y_pred_train = inverse_scale_load(y_pred_train, scaler, load_idx)
    y_pred_test = inverse_scale_load(y_pred_test, scaler, load_idx)
    y_test = inverse_scale_load(y_test, scaler, load_idx)
    y_train = inverse_scale_load(y_train, scaler, load_idx)

    # Plot & evaluate
    plot_prediction(y_train, y_pred_train, images_dir, suffix="train")
    plot_residuals(y_train, y_pred_train, images_dir, suffix="train")
    plot_prediction(y_test, y_pred_test, images_dir, suffix="test")
    plot_residuals(y_test, y_pred_test, images_dir, suffix="test")

    logging.info("===== XGBOOST TRAIN EVALUATION =====")
    evaluate(y_train, y_pred_train)
    logging.info("===== XGBOOST TEST EVALUATION =====")
    evaluate(y_test, y_pred_test)


if __name__ == "__main__":
    main()
