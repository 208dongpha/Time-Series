import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score


# load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.set_index('Datetime')
    df = df.sort_index()

    logging.info("Data loaded successfully")
    logging.info(f"Shape: {df.shape}")
    return df

def standardize_data(df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)

    df_scaled = pd.DataFrame(
        scaled,
        index=df.index,
        columns=df.columns
    )

    logging.info("StandardScaler applied")
    return df_scaled, scaler

# create sequence
def create_sequence(df, lookback=24, target_col="load"):
    X, y, idx = [], [], []
    values = df.values
    load_idx = df.columns.get_loc(target_col)

    for i in range(lookback, len(df)):
        X.append(values[i - lookback:i, :])
        y.append(values[i, load_idx])
        idx.append(df.index[i])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    logging.info(f"LSTM sequences created: X{X.shape}, y{y.shape}")

    return X, y, np.array(idx)

# split train
def split_timeseries(X, y, train_ratio=0.7, val_ratio=0.15):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return(
        X[:train_end], y[:train_end],
        X[train_end:val_end], y[train_end:val_end],
        X[val_end:], y[val_end:]
    )

# Dataset, Dataloader
def create_dataloader(X, y, batch_size=32, shuffle=False):
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden1=64, hidden2=32):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.fc = nn.Linear(hidden2, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        return self.fc(out).squeeze()

# train
def train(model, train_loader, val_loader, epochs=100, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, val_losses = [], []
    best_val = float("inf")
    best_epoch = -1
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_loss += criterion(model(X_batch), y_batch).item()
        
        
        train_epoch = train_loss / len(train_loader)
        val_epoch = val_loss / len(val_loader)
        train_losses.append(train_epoch)
        val_losses.append(val_epoch)

        if val_epoch < best_val:
            best_val = val_epoch
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        logging.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train MSE: {train_epoch:.6f} | "
            f"Val MSE: {val_epoch:.6f}"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
        logging.info(f"Best epoch selected: {best_epoch} (Val MSE: {best_val:.6f})")
    return train_losses, val_losses, best_epoch, best_val

# predict
def predict(model, X_test):
    model.eval()
    preds=[]

    with torch.no_grad():
        for i in range(len(X_test)):
            yhat = model(X_test[i:i+1])
            preds.append(yhat.item())
    
    return np.array(preds)

# inverse scaling 
def inverse_scale_load(series, scaler, load_index):
    return series * scaler.scale_[load_index] + scaler.mean_[load_index]

# Visualize
def plot_prediction(actual, predicted, save_path, title, index=None):
    plt.figure(figsize=(12,5))
    if index is None:
        plt.plot(actual, label="Actual")
        plt.plot(predicted, '--', label="LSTM Prediction")
    else:
        plt.plot(index, actual, label="Actual")
        plt.plot(index, predicted, '--', label="LSTM Prediction")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()

def plot_loss(train_losses, val_losses, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train MSE")
    plt.plot(val_losses, label="Val MSE")
    plt.title("LSTM Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


# Metrics
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    logging.info(f"RMSE: {rmse:.4f}")
    logging.info(f"MAE : {mae:.4f}")
    logging.info(f"MAPE: {mape:.2f}%")
    logging.info(f"R2  : {r2:.4f}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(base_dir, "logs")
    images_dir = os.path.join(base_dir, "images")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    log_file = f"lstm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - LSTM - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(logs_dir, log_file)),
            logging.StreamHandler()
        ],
        force=True
    )
    logging.info(f"Log file: {os.path.join(logs_dir, log_file)}")

    df = load_data(os.path.join(base_dir, "..", "dataclean.csv"))
    cols = ['load', 'temp', 'humidity', 'wind_speed']
    df_raw = df[cols].copy()

    # Split indices first, then fit scaler only on train
    n = len(df_raw)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    scaler = StandardScaler()
    scaler.fit(df_raw.iloc[:train_end])
    df_scaled = pd.DataFrame(
        scaler.transform(df_raw),
        index=df_raw.index,
        columns=df_raw.columns
    )

    X, y, idx = create_sequence(df_scaled, lookback=24, target_col="load")
    idx_pos = df_scaled.index.get_indexer(idx)
    train_idx = np.where(idx_pos < train_end)[0]
    val_idx = np.where((idx_pos >= train_end) & (idx_pos < val_end))[0]
    test_idx = np.where(idx_pos >= val_end)[0]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    test_index = idx[test_idx]

    train_loader = create_dataloader(X_train, y_train)
    val_loader = create_dataloader(X_val, y_val)

    model = LSTMModel()
    train_losses, val_losses, best_epoch, best_val = train(model, train_loader, val_loader)

    y_pred = predict(model, X_test)

    load_idx = cols.index("load")
    y_test_inv = inverse_scale_load(y_test.numpy(), scaler, load_idx)
    y_pred_inv = inverse_scale_load(y_pred, scaler, load_idx)

    plot_prediction(
        y_test_inv,
        y_pred_inv,
        os.path.join(images_dir, "lstm_prediction.png"),
        "LSTM  Actual vs Predicted (Test)",
        index=test_index
    )

    # === ZOOM PLOT (first 72 hours) ===
    plot_prediction(
        y_test_inv[:72],
        y_pred_inv[:72],
        os.path.join(images_dir, "lstm_zoom_72h.png"),
        "LSTM – Actual vs Predicted (First 72 Hours)",
        index=test_index[:72]
    )
    plot_loss(
        train_losses,
        val_losses,
        os.path.join(images_dir, "lstm_train_val_loss.png")
    )

    # === DAILY MEAN PLOT ===
    df_daily = pd.DataFrame({
        "actual": y_test_inv,
        "pred": y_pred_inv
    }, index=test_index)

    daily_mean = df_daily.resample("D").mean()

    plt.figure(figsize=(12,5))
    plt.plot(daily_mean.index, daily_mean["actual"], label="Actual")
    plt.plot(daily_mean.index, daily_mean["pred"], '--', label="LSTM Prediction")
    plt.title("LSTM – Daily Average Load (Test)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(images_dir, "lstm_daily_mean.png"))
    plt.close()


    evaluate(y_test_inv, y_pred_inv)


if __name__ == "__main__":
    main()



