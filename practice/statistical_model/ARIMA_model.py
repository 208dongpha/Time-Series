"""
ARIMA Time Series Analysis
ADF - Differencing - ACF/PACF - ARIMA - Forecast - Evaluation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from datetime import datetime

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf


# =====================================================
# PATH
# =====================================================
def get_base_dir() -> Path:
    return Path(__file__).resolve().parent


def get_data_path() -> Path:
    return get_base_dir() / "data" / "monthlyretailsales.csv"


def get_log_dir() -> Path:
    log_dir = get_base_dir() / "logs" / "ARIMA_model"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


# --- Hàm hỗ trợ ghi file ---
def save_report(content, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Successfully generated: {filename}")

# =====================================================
# LOGGER
# =====================================================
def create_logger(name: str, filename: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        return logger

    fh = logging.FileHandler(get_log_dir() / filename, encoding="utf-8")
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


main_logger = create_logger(
    "MAIN", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)
lag_logger = create_logger("LAGS", "suggested_lags.log")
adf_logger = create_logger("ADF", "adf.log")
ic_logger = create_logger("IC", "information_criteria.log")
eval_logger = create_logger("EVAL", "evaluation.log")
res_logger = create_logger("RES", "residuals.log")


# =====================================================
# LOAD DATA
# =====================================================
def load_data() -> pd.Series:
    path = get_data_path()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    main_logger.info(f"Loading data from {path}")
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    ts = df.iloc[:, 0]
    ts = ts.asfreq("MS")
    return ts



# =====================================================
# PLOT
# =====================================================
def save_plot(fig, filename: str):
    path = get_log_dir() / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    main_logger.info(f"Saved figure: {filename}")


def plot_series(ts: pd.Series, title: str, filename: str):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts)
    ax.set_title(title)
    ax.grid()
    save_plot(fig, filename)


def plot_acf_pacf(ts: pd.Series, lags: int = 24):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(ts, lags=lags, ax=ax[0])
    plot_pacf(ts, lags=lags, ax=ax[1])
    save_plot(fig, "03_acf_pacf.png")


def generate_analysis_report(ts: pd.Series):
    ts = ts.dropna()
    n = len(ts)

    # Z-scores
    z_90 = 1.645
    z_95 = 1.960
    z_99 = 2.576

    th_90 = z_90 / np.sqrt(n)
    th_95 = z_95 / np.sqrt(n)
    th_99 = z_99 / np.sqrt(n)

    acf_vals = acf(ts, nlags=10)
    pacf_vals = pacf(ts, nlags=10, method="ywm")

    report = []
    report.append("ACF and PACF Analysis for ARIMA Model Selection")
    report.append("=" * 60)
    report.append(f"Data length: {n}")
    report.append(f"Date range: {ts.index[0]} to {ts.index[-1]}")
    report.append(f"Mean: {ts.mean():.2f}")
    report.append(f"Std deviation: {ts.std():.2f}\n")

    report.append("=" * 60)
    report.append("THRESHOLD SELECTION EXPLANATION")
    report.append("=" * 60)
    report.append(f"Sample size (n): {n}\n")

    report.append("Confidence Interval Thresholds:")
    report.append(f"  90% CI threshold: ±{th_90:.4f}")
    report.append(f"  95% CI threshold: ±{th_95:.4f} (used for analysis)")
    report.append(f"  99% CI threshold: ±{th_99:.4f}\n")

    report.append(f"Threshold Used: {th_95:.4f}\n")

    report.append("Origin of Z-Score Values (1.65, 1.96, 2.58):")
    report.append("  These are critical values (z-scores) from the Standard Normal Distribution:")
    report.append("  - 1.65 ≈ z(0.95)  for 90% CI")
    report.append("  - 1.96 ≈ z(0.975) for 95% CI")
    report.append("  - 2.58 ≈ z(0.995) for 99% CI\n")

    report.append("  Statistical Basis:")
    report.append("  Under the null hypothesis (no autocorrelation),")
    report.append("  ACF/PACF values follow approximately N(0, 1/√n)")
    report.append("  CI = ±z_α/2 / √n\n")

    report.append("ARIMA Model Selection Guidelines:")
    report.append("  - AR order (p): Determined from PACF cutoff")
    report.append("  - MA order (q): Determined from ACF cutoff")
    report.append("  - Differencing order (d): Determined using ADF test")
    report.append("  - Values beyond ±threshold indicate significance\n")

    report.append("=" * 60)
    report.append(f"Significant PACF lags (|PACF| > {th_95:.4f}) - for AR order:")
    for i, v in enumerate(pacf_vals[1:], 1):
        if abs(v) > th_95:
            report.append(f"  Lag {i}: {v:.4f}")

    report.append("\n" + "=" * 60)
    report.append(f"Significant ACF lags (|ACF| > {th_95:.4f}) - for MA order:")
    for i, v in enumerate(acf_vals[1:], 1):
        if abs(v) > th_95:
            report.append(f"  Lag {i}: {v:.4f}")

    # Heuristic cutoff (lag đầu tiên vượt ngưỡng)
    p_cut = next((i for i, v in enumerate(pacf_vals[1:], 1) if abs(v) > th_95), 0)
    q_cut = next((i for i, v in enumerate(acf_vals[1:], 1) if abs(v) > th_95), 0)

    report.append("\n" + "=" * 60)
    report.append(f"Suggested ARIMA order: AR({p_cut}), I(d), MA({q_cut})")
    report.append("Note: Differencing order (d) is determined separately using ADF test\n")
    report.append(
        "Note: Visual inspection of ACF/PACF plots is recommended to confirm\n"
        "      the cutoff points and validate the model order selection."
    )

    return "\n".join(report)


# =====================================================
# ADF
# =====================================================
def adf_test(ts: pd.Series, d: int = 0):
    stat, pval = adfuller(ts.dropna())[0:2]

    adf_logger.info(f"--- ADF Test for d={d} ---")
    adf_logger.info(f"ADF Statistic: {stat}")
    adf_logger.info(f"p-value: {pval}")

    if pval < 0.05:
        adf_logger.info("Conclusion: Stationary")
    else:
        adf_logger.info("Conclusion: Non-stationary")

    adf_logger.info("-" * 40)
    return stat, pval


# 2. TẠO BÁO CÁO ADF TEST (Tính dừng)
def generate_stationarity_report(ts: pd.Series, alpha: float = 0.05):
    ts = ts.dropna()
    n = len(ts)

    def iterative_difference(ts, d):
        temp = ts.copy()
        for _ in range(d):
            temp = temp.diff().dropna()
        return temp

    report = []
    report.append("Stationarity Test Results (Augmented Dickey-Fuller Test)")
    report.append("=" * 60)
    report.append(f"Original data length: {n}")
    report.append(f"Date range: {ts.index[0]} to {ts.index[-1]}\n")

    report.append("=" * 60)
    report.append("TEST RESULTS FOR DIFFERENT DIFFERENCING ORDERS")
    report.append("=" * 60)

    best_d = None

    for d in [0, 1, 2]:
        ts_d = ts if d == 0 else iterative_difference(ts, d)

        stat, pval, _, _, crit, _ = adfuller(ts_d)

        report.append(
            f"\nDifferencing Order d = {d} " +
            ("(Original):" if d == 0 else f"(Differenced {d} time{'s' if d > 1 else ''}):")
        )
        report.append(f"  ADF Statistic: {stat:.4f}")
        report.append(f"  p-value: {pval:.4f}")
        report.append("  Critical Values:")
        for k, v in crit.items():
            report.append(f"    {k}: {v:.4f}")

        stationary = pval < alpha
        report.append(
            f"  Stationary: {'Yes' if stationary else 'No'} "
            f"(p-value {'<' if stationary else '>='} {alpha})"
        )
        report.append(f"  Number of observations: {len(ts_d)}")

        if stationary and best_d is None:
            best_d = d

    report.append("\n" + "=" * 60)
    report.append(f"Suggested Differencing Order (d): {best_d}\n")

    report.append("Interpretation:")
    report.append("  - H0 (null hypothesis): Series has a unit root (non-stationary)")
    report.append("  - H1 (alternative): Series is stationary")
    report.append(f"  - If p-value < {alpha}: Reject H0, series is stationary")
    report.append(f"  - If p-value >= {alpha}: Fail to reject H0, series is non-stationary")

    return "\n".join(report)


# =====================================================
# TRANSFORM
# =====================================================
def difference(ts: pd.Series, d: int = 1) -> pd.Series:
    main_logger.info(f"Differencing with order d={d}")
    temp_ts = ts.copy()
    for _ in range(d):
        temp_ts = temp_ts.diff().dropna()
    return temp_ts


def generate_z_table_report(n_sample: int):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # -------------------------------
    # Z-scores (standard, fixed)
    # -------------------------------
    z_values = [
        (80.0, 0.200, 1.2816),
        (85.0, 0.150, 1.4395),
        (90.0, 0.100, 1.6449),
        (95.0, 0.050, 1.9600),
        (99.0, 0.010, 2.5758),
        (99.9, 0.001, 3.2905),
    ]

    percentile_table = [
        (50.00, 0.0000, "50%", "Median"),
        (60.00, 0.2533, "60%", ""),
        (70.00, 0.5244, "70%", ""),
        (75.00, 0.6745, "75%", ""),
        (80.00, 0.8416, "80%", "Common threshold"),
        (85.00, 1.0364, "85%", ""),
        (90.00, 1.2816, "90%", "Common threshold"),
        (95.00, 1.6449, "95%", "Very common"),
        (97.50, 1.9600, "97.5%", "95% CI (two-tailed)"),
        (99.00, 2.3263, "99%", "Common threshold"),
        (99.50, 2.5758, "99.5%", "99% CI (two-tailed)"),
        (99.90, 3.0902, "99.9%", "High confidence"),
        (99.95, 3.2905, "99.95%", "Very high confidence"),
    ]

    # -------------------------------
    # Thresholds for THIS dataset
    # -------------------------------
    z_90 = 1.6449
    z_95 = 1.9600
    z_99 = 2.5758

    thr_90 = z_90 / np.sqrt(n_sample)
    thr_95 = z_95 / np.sqrt(n_sample)
    thr_99 = z_99 / np.sqrt(n_sample)

    # -------------------------------
    # Report
    # -------------------------------
    report = []

    report.append("=" * 80)
    report.append("STANDARD NORMAL DISTRIBUTION (Z-TABLE)")
    report.append("Critical Values for ACF / PACF Confidence Intervals")
    report.append("=" * 80 + "\n")

    report.append("This report explains the statistical basis used to determine")
    report.append("confidence interval thresholds in ACF and PACF analysis.\n")

    report.append("Statistical assumption:")
    report.append("Under the null hypothesis of no autocorrelation,")
    report.append("sample ACF and PACF values are approximately distributed as:")
    report.append("  N(0, 1 / √n)\n")

    report.append("Confidence interval formula:")
    report.append("  CI = ± z_(α/2) / √n\n")

    # -------------------------------
    # Confidence levels
    # -------------------------------
    report.append("=" * 80)
    report.append("COMMONLY USED CONFIDENCE LEVELS")
    report.append("=" * 80)
    report.append("Confidence Level      Two-Tailed α      Z-Score (zα/2)")
    report.append("-" * 80)

    for cl, alpha, z in z_values:
        report.append(f"{cl:>6.1f}%{'':14}{alpha:<16.3f}{z:>10.4f}")

    # -------------------------------
    # Percentile table
    # -------------------------------
    report.append("\n" + "=" * 80)
    report.append("DETAILED Z-TABLE (COMMON PERCENTILES)")
    report.append("=" * 80)
    report.append("Percentile      Z-Score      Confidence Level     Usage")
    report.append("-" * 80)

    for p, z, cl, usage in percentile_table:
        report.append(f"{p:>6.2f}%{'':8}{z:<12.4f}{cl:<20}{usage}")

    # -------------------------------
    # Application to dataset
    # -------------------------------
    report.append("\n" + "=" * 80)
    report.append("APPLICATION TO YOUR DATASET")
    report.append("=" * 80)
    report.append(f"Sample size (n): {n_sample}\n")

    report.append("Calculated confidence thresholds:")
    report.append(f"  90% CI  : ±{thr_90:.4f}   (z = 1.645)")
    report.append(f"  95% CI  : ±{thr_95:.4f}   (z = 1.96)  ← used for analysis")
    report.append(f"  99% CI  : ±{thr_99:.4f}   (z = 2.576)\n")

    report.append("Interpretation for ACF/PACF plots:")
    report.append(f"- Any ACF or PACF value exceeding ±{thr_95:.4f}")
    report.append("  is considered statistically significant.")
    report.append("- Significant PACF lags suggest AR order (p).")
    report.append("- Significant ACF lags suggest MA order (q).")
    report.append("- Differencing order (d) is determined separately using ADF test.\n")

    # -------------------------------
    # Notes
    # -------------------------------
    report.append("=" * 80)
    report.append("NOTES")
    report.append("=" * 80)
    report.append("- Z-scores are derived from the Standard Normal Distribution N(0,1).")
    report.append("- Two-tailed confidence intervals are used for ACF/PACF analysis.")
    report.append("- This approximation is valid for large samples (n > 30).")
    report.append("- For small samples, t-distribution may be more appropriate.\n")

    report.append(f"Generated: {timestamp}")
    report.append("=" * 80)

    return "\n".join(report)

# =====================================================
# SPLIT
# =====================================================
def split(ts: pd.Series, ratio: float = 0.8):
    n = int(len(ts) * ratio)
    train, test = ts.iloc[:n], ts.iloc[n:]
    main_logger.info(f"Train size: {len(train)}, Test size: {len(test)}")
    return train, test


# =====================================================
# ARIMA
# =====================================================
def fit_arima(train: pd.Series, order: tuple):
    main_logger.info(f"Fitting ARIMA{order}")
    model = ARIMA(train, order=order)
    result = model.fit()

    ic_logger.info(f"ARIMA{order}")
    ic_logger.info(f"AIC: {result.aic:.4f}")
    ic_logger.info(f"BIC: {result.bic:.4f}")
    ic_logger.info("-" * 50)

    return result

# =====================================================
# SARIMA
# =====================================================
def fit_sarima(train: pd.Series, order: tuple, s_order: tuple):
    main_logger.info(f"Fitting SARIMA{order}x{s_order}")
    # statsmodels.tsa.arima.model.ARIMA hỗ trợ cả SARIMA qua seasonal_order
    model = ARIMA(train, order=order, seasonal_order=s_order)
    result = model.fit()
    
    # Ghi lại AIC/BIC để so sánh
    ic_logger.info(f"SARIMA{order}x{s_order} | AIC: {result.aic:.2f} | BIC: {result.bic:.2f}")
    return result

# =====================================================
# RESIDUALS
# =====================================================
def residual_diagnostics(residuals: pd.Series, order: tuple):
    res_logger.info(f"Residual diagnostics ARIMA{order}")

    mean_res = residuals.mean()
    res_logger.info(f"Residual mean: {mean_res:.6f}")

    if abs(mean_res) < 0.01:
        res_logger.info("Residual mean approx 0 -> OK")
    else:
        res_logger.info("Residual mean != 0 -> CHECK")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(residuals)
    ax.set_title(f"Residuals ARIMA{order}")
    ax.grid()
    save_plot(fig, f"04_residuals_{order}.png")

    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(residuals.dropna(), lags=20, ax=ax)
    save_plot(fig, f"05_residuals_acf_{order}.png")


# =====================================================
# FORECAST & EVAL
# =====================================================
def forecast(model, steps: int) -> pd.Series:
    main_logger.info(f"Forecasting {steps} steps")
    return model.forecast(steps=steps)


def plot_forecast(train, test, pred):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train, label="Train")
    ax.plot(test, label="Test")
    ax.plot(test.index, pred, label="Forecast")
    ax.legend()
    ax.grid()
    save_plot(fig, "06_forecast.png")


def evaluation(test: pd.Series, pred: pd.Series, model_name: str):
    mae = np.mean(np.abs(test - pred))
    rmse = np.sqrt(np.mean((test - pred) ** 2))
    mape = np.mean(np.abs((test - pred) / test)) * 100

    eval_logger.info(f"=== Evaluation for: {model_name} ===")
    eval_logger.info(f"MAE: {mae}")
    eval_logger.info(f"RMSE: {rmse}")
    eval_logger.info(f"MAPE: {mape}")
    eval_logger.info("-" * 40)

# LAGS
def suggest_arima_lags(
    ts: pd.Series,
    max_lag: int = 20,
    alpha: float = 0.05,
    d_val: int = 1  # Thêm tham số này để nhận giá trị d từ hàm main
):
    """
    Suggest AR (p) and MA (q) orders based on PACF & ACF
    """
    n = len(ts.dropna())
    conf = 1.96 / np.sqrt(n)

    acf_vals = acf(ts.dropna(), nlags=max_lag)
    pacf_vals = pacf(ts.dropna(), nlags=max_lag)

    q_lags = [
        lag for lag, val in enumerate(acf_vals[1:], start=1)
        if abs(val) > conf
    ]

    p_lags = [
        lag for lag, val in enumerate(pacf_vals[1:], start=1)
        if abs(val) > conf
    ]

    lag_logger.info("=== SUGGESTED ARIMA LAGS ===")
    lag_logger.info(f"Sample size (N): {n}")
    lag_logger.info(f"Confidence threshold: ±{conf:.4f}")

    lag_logger.info(f"Suggested p (AR) lags from PACF: {p_lags}")
    lag_logger.info(f"Suggested q (MA) lags from ACF: {q_lags}")

    # Heuristic suggestion
    p_suggest = p_lags[:2]
    q_suggest = q_lags[:2]

    # SỬA TẠI ĐÂY: Thay số 1 cố định bằng biến d_val
    candidates = [(p, d_val, q) for p in p_suggest for q in q_suggest]

    lag_logger.info(f"Recommended ARIMA candidates: {candidates}")
    lag_logger.info("-" * 50)

    print(f"Suggested p (PACF): {p_lags}")
    print(f"Suggested q (ACF): {q_lags}")

    return p_lags, q_lags

def plot_comparison(train, test, pred_arima, pred_sarima):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train, label="Dữ liệu Train", color='blue', alpha=0.5)
    ax.plot(test, label="Dữ liệu Test (Thực tế)", color='orange')
    ax.plot(test.index, pred_arima, label="Dự báo ARIMA (Chỉ xu hướng)", color='red', linestyle='--')
    ax.plot(test.index, pred_sarima, label="Dự báo SARIMA (Xu hướng + Mùa vụ)", color='green')
    
    ax.set_title("So sánh mô hình ARIMA và SARIMA")
    ax.legend()
    ax.grid(True)
    save_plot(fig, "07_comparison_forecast.png")

# =====================================================
# MAIN
# =====================================================
def main():
    main_logger.info("=== START ARIMA & SARIMA PIPELINE ===")

    # 1. Load và EDA dữ liệu
    ts = load_data()
    plot_series(ts, "Original Time Series", "01_original_series.png")
    
    # 2. Kiểm định ADF trên chuỗi gốc (d=0)
    adf_test(ts, d=0)

    # 3. Xử lý sai phân để tìm d tối ưu
    ts_diff1 = difference(ts, d=1)
    stat1, p1 = adf_test(ts_diff1, d=1)

    # Nếu d=1 chưa dừng (p > 0.05), thử d=2 như bạn đã test thành công
    if p1 >= 0.05:
        main_logger.info("p-value > 0.05, trying d=2...")
        ts_diff2 = difference(ts, d=2)
        adf_test(ts_diff2, d=2)
        ts_final = ts_diff2
        d_final = 2
    else:
        ts_final = ts_diff1
        d_final = 1

    # 4. Vẽ ACF / PACF trên chuỗi đã đạt tính dừng
    plot_acf_pacf(ts_final)
    p_lags, q_lags = suggest_arima_lags(ts_final, max_lag=20, d_val=d_final)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_report(
        generate_z_table_report(len(ts)),
        f"logs/z_table_{timestamp}.txt"
    )

    save_report(
        generate_stationarity_report(ts),
        f"logs/stationarity_test_{timestamp}.txt"
    )

    save_report(
        generate_analysis_report(ts_final),
        f"logs/ACF_PACF_Analysis_{timestamp}.txt"
    )

    # 5. Chia tập Train/Test (80/20)
    train, test = split(ts)

    # 6. Huấn luyện mô hình ARIMA (Mô hình chỉ xử lý xu hướng)
    # Dựa trên test của bạn, ta chọn (2, 2, 1) là ứng viên mạnh nhất
    arima_order = (2, d_final, 1)
    best_arima = fit_arima(train, arima_order)
    
    # 7. Huấn luyện mô hình SARIMA (Mô hình xử lý cả xu hướng và mùa vụ)
    # Thêm thành phần seasonal_order (P, D, Q, s) với s=12 (tháng)
    seasonal_order = (1, 1, 1, 12)
    best_sarima = fit_sarima(train, (1, d_final, 1), seasonal_order)

    # 8. Chẩn đoán phần dư (Residuals)

    # Chẩn đoán cho ARIMA
    main_logger.info("Generating diagnostics for ARIMA...")
    residual_diagnostics(best_arima.resid, arima_order) 

    # Chẩn đoán cho SARIMA
    main_logger.info("Generating diagnostics for SARIMA...")
    residual_diagnostics(best_sarima.resid, (1, d_final, 1, 12))

    # 9. Dự báo trên tập Test
    pred_arima = forecast(best_arima, len(test))
    pred_sarima = forecast(best_sarima, len(test))

    # 10. Đánh giá và So sánh
    main_logger.info(f"--- SO SÁNH AIC ---")
    main_logger.info(f"ARIMA AIC: {best_arima.aic:.2f}")
    main_logger.info(f"SARIMA AIC: {best_sarima.aic:.2f}") 

    # Truyền thêm tên mô hình vào hàm evaluation để log hiển thị rõ ràng
    evaluation(test, pred_arima, "ARIMA(2, 2, 1)") 
    evaluation(test, pred_sarima, "SARIMA(1, 1, 1)x(1, 1, 1, 12)")

    # 11. Vẽ biểu đồ so sánh cuối cùng
    plot_comparison(train, test, pred_arima, pred_sarima)

    main_logger.info("=== END PIPELINE ===")


if __name__ == "__main__":
    main()
