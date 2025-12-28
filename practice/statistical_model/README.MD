# Phân Tích và Dự Báo Doanh Số Bán Lẻ bằng ARIMA và SARIMA

## 📋 Tổng Quan Dự Án

Dự án này thực hiện phân tích và dự báo chuỗi thời gian doanh số bán lẻ hàng tháng sử dụng các mô hình thống kê ARIMA (AutoRegressive Integrated Moving Average) và SARIMA (Seasonal ARIMA). Mục tiêu là so sánh hiệu quả giữa hai mô hình trong việc dự báo dữ liệu có xu hướng tăng trưởng mạnh và tính mùa vụ rõ rệt.

### Kết Quả Tóm Tắt

Dựa trên dữ liệu kiểm thử, mô hình **SARIMA** cho thấy sự vượt trội hoàn toàn:

- **Chỉ số AIC**: Giảm từ **597.03** (ARIMA) xuống **279.03** (SARIMA) - giảm 53%
- **Sai số MAPE**: Giảm từ **21.51%** xuống còn **11.55%** - giảm gần 50%
- **RMSE**: Giảm từ **4.30** xuống **2.47** - giảm 43%
- **Khả năng dự báo**: SARIMA tái hiện được các biến động hình răng cưa thực tế, trong khi ARIMA chỉ dự báo được xu hướng phẳng

---

## 🛠️ Cài Đặt và Yêu Cầu

### Yêu Cầu Hệ Thống

- Python 3.7+
- Các thư viện Python (xem `requirements.txt`)

### Cài Đặt

```bash
pip install -r requirements.txt
```

### Các Thư Viện Chính

- **statsmodels**: Xây dựng mô hình ARIMA/SARIMA, kiểm định ADF, vẽ ACF/PACF
- **pandas**: Xử lý và thao tác dữ liệu chuỗi thời gian
- **numpy**: Tính toán số học và thống kê
- **matplotlib**: Vẽ biểu đồ và trực quan hóa kết quả

---

## 📁 Cấu Trúc Dự Án

```
statistical_model/
├── ARIMA_model.py              # File chính chứa toàn bộ pipeline
├── data/
│   └── monthlyretailsales.csv  # Dữ liệu đầu vào
├── logs/
│   └── ARIMA_model/            # Thư mục chứa kết quả
│       ├── *.png               # Các biểu đồ
│       ├── *.log               # Các file log
│       └── *.txt               # Các báo cáo phân tích
└── README.md                   # File này
```

---

## 📖 Giải Thích Code Chi Tiết

### 1. Cấu Trúc Tổng Quan

File `ARIMA_model.py` được tổ chức thành các module chức năng rõ ràng:

```python
# 1. PATH MANAGEMENT - Quản lý đường dẫn
# 2. LOGGER - Hệ thống ghi log
# 3. LOAD DATA - Tải dữ liệu
# 4. PLOT - Vẽ biểu đồ
# 5. ADF - Kiểm định tính dừng
# 6. TRANSFORM - Biến đổi dữ liệu
# 7. SPLIT - Chia tập train/test
# 8. ARIMA - Xây dựng mô hình
# 9. RESIDUALS - Chẩn đoán phần dư
# 10. FORECAST & EVAL - Dự báo và đánh giá
# 11. MAIN - Hàm điều khiển chính
```

### 2. Quản Lý Đường Dẫn (PATH MANAGEMENT)

```python
def get_base_dir() -> Path:
    """Lấy đường dẫn thư mục gốc chứa file script hiện tại."""
    return Path(__file__).resolve().parent

def get_data_path() -> Path:
    """Lấy đường dẫn tuyệt đối đến file dữ liệu CSV."""
    return get_base_dir() / "data" / "monthlyretailsales.csv"

def get_log_dir() -> Path:
    """Tự động tạo thư mục logs nếu chưa tồn tại."""
    log_dir = get_base_dir() / "logs" / "ARIMA_model"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir
```

**Giải thích**: Sử dụng `pathlib.Path` để quản lý đường dẫn một cách an toàn và tự động, tránh lỗi khi chạy từ các thư mục khác nhau.

### 3. Hệ Thống Logger

Code sử dụng 6 logger riêng biệt để theo dõi từng khía cạnh của pipeline:

```python
main_logger   # Theo dõi luồng thực thi chính (mỗi lần chạy tạo file riêng)
lag_logger    # Ghi lại gợi ý về bậc p và q từ ACF/PACF
adf_logger    # Chi tiết kết quả kiểm định tính dừng
ic_logger     # Chỉ số AIC/BIC để so sánh mô hình
eval_logger   # Kết quả đánh giá (MAE, RMSE, MAPE)
res_logger    # Chẩn đoán phần dư (residuals)
```

**Lợi ích**: 
- Dễ dàng debug và theo dõi từng bước
- Lưu trữ lịch sử các lần chạy
- Phân tích kết quả một cách có hệ thống

### 4. Tải và Xử Lý Dữ Liệu

```python
def load_data() -> pd.Series:
    path = get_data_path()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    ts = df.iloc[:, 0]
    ts = ts.asfreq("MS")  # Monthly Start frequency
    return ts
```

**Giải thích**:
- Đọc CSV và chuyển cột `date` thành index
- `asfreq("MS")` đảm bảo tần suất dữ liệu là hàng tháng (Monthly Start)
- Trả về `pd.Series` với index là datetime

### 5. Kiểm Định Tính Dừng (ADF Test)

```python
def adf_test(ts: pd.Series, d: int = 0):
    """Thực hiện kiểm định Augmented Dickey-Fuller."""
    stat, pval = adfuller(ts.dropna())[0:2]
    
    if pval < 0.05:
        adf_logger.info("Conclusion: Stationary")
    else:
        adf_logger.info("Conclusion: Non-stationary")
    
    return stat, pval
```

**Giải thích**:
- **ADF Test** kiểm tra xem chuỗi có tính dừng (stationary) hay không
- **H0**: Chuỗi có unit root (không dừng)
- **H1**: Chuỗi dừng
- Nếu `p-value < 0.05`: Bác bỏ H0 → Chuỗi dừng

**Kết quả từ log**:
```
d=0: p-value = 1.0 > 0.05 → Không dừng
d=1: p-value = 0.117 > 0.05 → Không dừng  
d=2: p-value = 3.54e-18 < 0.05 → Dừng ✓
```

### 6. Sai Phân (Differencing)

```python
def difference(ts: pd.Series, d: int = 1) -> pd.Series:
    """Thực hiện lấy sai phân bậc d để loại bỏ xu hướng."""
    temp_ts = ts.copy()
    for _ in range(d):
        temp_ts = temp_ts.diff().dropna()
    return temp_ts
```

**Giải thích**:
- Sai phân bậc 1: `y_t = x_t - x_{t-1}`
- Sai phân bậc 2: Áp dụng sai phân 2 lần
- Mục đích: Loại bỏ xu hướng (trend) để đạt tính dừng

### 7. Phân Tích ACF/PACF

```python
def suggest_arima_lags(ts: pd.Series, max_lag: int = 20, d_val: int = 1):
    """Gợi ý bậc p (AR) và q (MA) dựa trên ACF/PACF."""
    n = len(ts.dropna())
    conf = 1.96 / np.sqrt(n)  # Ngưỡng tin cậy 95%
    
    acf_vals = acf(ts.dropna(), nlags=max_lag)
    pacf_vals = pacf(ts.dropna(), nlags=max_lag)
    
    # Tìm các lag vượt ngưỡng
    q_lags = [lag for lag, val in enumerate(acf_vals[1:], start=1) 
              if abs(val) > conf]
    p_lags = [lag for lag, val in enumerate(pacf_vals[1:], start=1) 
              if abs(val) > conf]
    
    return p_lags, q_lags
```

**Giải thích**:
- **ACF (Autocorrelation Function)**: Đo tương quan giữa giá trị hiện tại và các giá trị quá khứ
- **PACF (Partial Autocorrelation Function)**: Đo tương quan trực tiếp giữa `y_t` và `y_{t-k}` sau khi loại bỏ ảnh hưởng của các lag trung gian
- **Ngưỡng tin cậy**: `±1.96/√n` cho khoảng tin cậy 95%
- **Quy tắc**:
  - **PACF cắt cụt tại lag p** → Gợi ý bậc AR = p
  - **ACF cắt cụt tại lag q** → Gợi ý bậc MA = q

**Kết quả từ log**:
```
Suggested p (AR) lags from PACF: [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 16]
Suggested q (MA) lags from ACF: [1, 2, 10, 11, 12, 13]
Recommended ARIMA candidates: [(1, 2, 1), (1, 2, 2), (2, 2, 1), (2, 2, 2)]
```

### 8. Xây Dựng Mô Hình ARIMA

```python
def fit_arima(train: pd.Series, order: tuple):
    """Huấn luyện mô hình ARIMA với tham số (p, d, q)."""
    model = ARIMA(train, order=order)
    result = model.fit()
    
    ic_logger.info(f"ARIMA{order}")
    ic_logger.info(f"AIC: {result.aic:.4f}")
    ic_logger.info(f"BIC: {result.bic:.4f}")
    
    return result
```

**Giải thích**:
- **ARIMA(p, d, q)**:
  - `p`: Bậc tự hồi quy (AR)
  - `d`: Bậc sai phân (I)
  - `q`: Bậc trung bình trượt (MA)
- **AIC/BIC**: Chỉ số so sánh mô hình (càng nhỏ càng tốt)

### 9. Xây Dựng Mô Hình SARIMA

```python
def fit_sarima(train: pd.Series, order: tuple, s_order: tuple):
    """Huấn luyện mô hình SARIMA với thành phần mùa vụ."""
    model = ARIMA(train, order=order, seasonal_order=s_order)
    result = model.fit()
    return result
```

**Giải thích**:
- **SARIMA(p, d, q)(P, D, Q, s)**:
  - `(p, d, q)`: Tham số cho thành phần không mùa vụ
  - `(P, D, Q, s)`: Tham số cho thành phần mùa vụ
  - `s`: Chu kỳ mùa vụ (12 cho dữ liệu hàng tháng)
- **Ưu điểm**: Xử lý được cả xu hướng và tính mùa vụ

### 10. Chẩn Đoán Phần Dư (Residual Diagnostics)

```python
def residual_diagnostics(residuals: pd.Series, order: tuple):
    """Kiểm tra phần dư có phải là nhiễu trắng không."""
    mean_res = residuals.mean()
    
    if abs(mean_res) < 0.01:
        res_logger.info("Residual mean approx 0 -> OK")
    else:
        res_logger.info("Residual mean != 0 -> CHECK")
    
    # Vẽ biểu đồ phần dư và ACF của phần dư
    plot_residuals(residuals, order)
    plot_residuals_acf(residuals, order)
```

**Giải thích**:
- **Phần dư tốt** phải có:
  - Trung bình ≈ 0
  - Không có tương quan (ACF của phần dư nằm trong ngưỡng tin cậy)
  - Phân phối chuẩn
- Nếu phần dư không phải nhiễu trắng → Mô hình chưa tốt, cần điều chỉnh

**Kết quả từ log**:
```
ARIMA(2, 2, 1): Residual mean = 0.086 → CHECK (hơi cao)
SARIMA(1, 2, 1)x(1, 1, 1, 12): Residual mean = 0.010 → OK ✓
```

### 11. Dự Báo và Đánh Giá

```python
def forecast(model, steps: int) -> pd.Series:
    """Dự báo các giá trị tương lai."""
    return model.forecast(steps=steps)

def evaluation(test: pd.Series, pred: pd.Series, model_name: str):
    """Tính toán các chỉ số đánh giá."""
    mae = np.mean(np.abs(test - pred))
    rmse = np.sqrt(np.mean((test - pred) ** 2))
    mape = np.mean(np.abs((test - pred) / test)) * 100
```

**Giải thích các chỉ số**:
- **MAE (Mean Absolute Error)**: Sai số tuyệt đối trung bình
- **RMSE (Root Mean Squared Error)**: Căn bậc hai của sai số bình phương trung bình (nhạy cảm với outlier)
- **MAPE (Mean Absolute Percentage Error)**: Sai số phần trăm trung bình (dễ hiểu, không phụ thuộc đơn vị)

---

## 🔍 Phân Tích Kết Quả Đầu Ra

### 1. File Log: `run_YYYYMMDD_HHMMSS.log`

File này ghi lại toàn bộ luồng thực thi của pipeline:

```
2025-12-28 18:58:35 - === START ARIMA & SARIMA PIPELINE ===
2025-12-28 18:58:35 - Loading data from .../monthlyretailsales.csv
2025-12-28 18:58:36 - Saved figure: 01_original_series.png
2025-12-28 18:58:36 - Differencing with order d=1
2025-12-28 18:58:36 - p-value > 0.05, trying d=2...
2025-12-28 18:58:36 - Differencing with order d=2
2025-12-28 18:58:36 - Saved figure: 03_acf_pacf.png
2025-12-28 18:58:36 - Train size: 163, Test size: 41
2025-12-28 18:58:36 - Fitting ARIMA(2, 2, 1)
2025-12-28 18:58:37 - Fitting SARIMA(1, 2, 1)x(1, 1, 1, 12)
2025-12-28 18:58:38 - Generating diagnostics for ARIMA...
2025-12-28 18:58:39 - Forecasting 41 steps
2025-12-28 18:58:39 - ARIMA AIC: 597.03
2025-12-28 18:58:39 - SARIMA AIC: 279.03
2025-12-28 18:58:39 - === END PIPELINE ===
```

**Phân tích**: Pipeline chạy thành công, tự động phát hiện cần sai phân bậc 2, và chia dữ liệu 80/20 cho train/test.

### 2. File Log: `adf.log`

Kết quả kiểm định tính dừng:

```
--- ADF Test for d=0 ---
ADF Statistic: 3.1452
p-value: 1.0
Conclusion: Non-stationary

--- ADF Test for d=1 ---
ADF Statistic: -2.4952
p-value: 0.1167
Conclusion: Non-stationary

--- ADF Test for d=2 ---
ADF Statistic: -10.2923
p-value: 3.54e-18
Conclusion: Stationary ✓
```

**Phân tích**:
- Chuỗi gốc và sai phân bậc 1 đều **không dừng** (p-value > 0.05)
- Chỉ sau khi sai phân bậc 2, chuỗi mới đạt tính dừng (p-value cực nhỏ)
- **Quyết định**: Sử dụng `d = 2` cho mô hình ARIMA

### 3. File Log: `suggested_lags.log`

Gợi ý về bậc p và q:

```
=== SUGGESTED ARIMA LAGS ===
Sample size (N): 202
Confidence threshold: ±0.1379

Suggested p (AR) lags from PACF: [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 16]
Suggested q (MA) lags from ACF: [1, 2, 10, 11, 12, 13]

Recommended ARIMA candidates: [(1, 2, 1), (1, 2, 2), (2, 2, 1), (2, 2, 2)]
```

**Phân tích**:
- Nhiều lag vượt ngưỡng, nhưng thường chọn lag đầu tiên hoặc vài lag đầu
- Code chọn 2 lag đầu tiên: `p_suggest = [1, 2]`, `q_suggest = [1, 2]`
- Tạo ra 4 ứng viên: `(1,2,1)`, `(1,2,2)`, `(2,2,1)`, `(2,2,2)`
- **Lưu ý**: Lag 12 xuất hiện trong cả ACF và PACF → Gợi ý mạnh về tính mùa vụ (s=12)

### 4. File Log: `information_criteria.log`

So sánh AIC/BIC giữa các mô hình:

```
ARIMA(2, 2, 1)
AIC: 597.0252
BIC: 609.3508

SARIMA(1, 2, 1)x(1, 1, 1, 12)
AIC: 279.03
BIC: 294.05
```

**Phân tích**:
- **AIC của SARIMA thấp hơn 53%** so với ARIMA → SARIMA phù hợp hơn rất nhiều
- **BIC cũng thấp hơn đáng kể** → Xác nhận SARIMA là mô hình tốt hơn
- **Kết luận**: SARIMA giải thích được nhiều biến thiên hơn với cùng số tham số (hoặc ít tham số hơn)

### 5. File Log: `residuals.log`

Chẩn đoán phần dư:

```
Residual diagnostics ARIMA(2, 2, 1)
Residual mean: 0.086271
Residual mean != 0 -> CHECK

Residual diagnostics SARIMA(1, 2, 1, 12)
Residual mean: 0.009902
Residual mean approx 0 -> OK ✓
```

**Phân tích**:
- **ARIMA**: Trung bình phần dư = 0.086 (hơi cao) → Có thể còn bias
- **SARIMA**: Trung bình phần dư = 0.010 (rất gần 0) → Phần dư tốt hơn
- **Kết luận**: SARIMA có phần dư gần với nhiễu trắng hơn

### 6. File Log: `evaluation.log`

Kết quả đánh giá trên tập test:

```
=== Evaluation for: ARIMA(2, 2, 1) ===
MAE: 3.636
RMSE: 4.300
MAPE: 21.51%

=== Evaluation for: SARIMA(1, 1, 1)x(1, 1, 1, 12) ===
MAE: 1.995
RMSE: 2.466
MAPE: 11.55%
```

**Phân tích**:
- **MAE**: SARIMA giảm 45% (từ 3.64 → 1.99)
- **RMSE**: SARIMA giảm 43% (từ 4.30 → 2.47)
- **MAPE**: SARIMA giảm 46% (từ 21.51% → 11.55%)
- **Kết luận**: SARIMA dự báo chính xác hơn đáng kể trên mọi chỉ số

### 7. File Báo Cáo: `stationarity_test_YYYYMMDD_HHMMSS.txt`

Báo cáo chi tiết về kiểm định tính dừng:

```
Stationarity Test Results (Augmented Dickey-Fuller Test)
============================================================
Original data length: 204
Date range: 1991-07-01 to 2008-06-01

Differencing Order d = 0 (Original):
  ADF Statistic: 3.1452
  p-value: 1.0000
  Stationary: No (p-value >= 0.05)

Differencing Order d = 1:
  ADF Statistic: -2.4952
  p-value: 0.1167
  Stationary: No (p-value >= 0.05)

Differencing Order d = 2:
  ADF Statistic: -10.2923
  p-value: 0.0000
  Stationary: Yes (p-value < 0.05) ✓

Suggested Differencing Order (d): 2
```

**Phân tích**: Báo cáo này cung cấp đầy đủ thông tin về quá trình kiểm định, giúp người đọc hiểu rõ lý do chọn `d=2`.

### 8. File Báo Cáo: `ACF_PACF_Analysis_YYYYMMDD_HHMMSS.txt`

Báo cáo phân tích ACF/PACF:

```
ACF and PACF Analysis for ARIMA Model Selection
============================================================
Data length: 202
Confidence Interval Thresholds:
  90% CI threshold: ±0.1157
  95% CI threshold: ±0.1379 (used for analysis)
  99% CI threshold: ±0.1812

Significant PACF lags (|PACF| > 0.1379) - for AR order:
  Lag 1: -0.6346
  Lag 2: -0.3673
  Lag 3: -0.3032
  ...

Significant ACF lags (|ACF| > 0.1379) - for MA order:
  Lag 1: -0.6346
  Lag 2: -0.3673
  ...

Suggested ARIMA order: AR(2), I(2), MA(1)
```

**Phân tích**: 
- Giải thích rõ cách tính ngưỡng tin cậy dựa trên Z-score
- Liệt kê tất cả các lag có ý nghĩa thống kê
- Đưa ra gợi ý cụ thể về bậc mô hình

### 9. File Báo Cáo: `z_table_YYYYMMDD_HHMMSS.txt`

Bảng tra cứu Z-score và giải thích cơ sở toán học:

```
STANDARD NORMAL DISTRIBUTION (Z-TABLE)
Critical Values for ACF / PACF Confidence Intervals
================================================================================

Statistical assumption:
Under the null hypothesis of no autocorrelation,
sample ACF and PACF values are approximately distributed as:
  N(0, 1 / √n)

Confidence interval formula:
  CI = ± z_(α/2) / √n

COMMONLY USED CONFIDENCE LEVELS
Confidence Level      Two-Tailed α      Z-Score (zα/2)
--------------------------------------------------------------------------------
  90.0%                0.100            1.6449
  95.0%                0.050            1.9600
  99.0%                0.010            2.5758

APPLICATION TO YOUR DATASET
Sample size (n): 204

Calculated confidence thresholds:
  90% CI  : ±0.1157   (z = 1.645)
  95% CI  : ±0.1379   (z = 1.96)  ← used for analysis
  99% CI  : ±0.1812   (z = 2.576)
```

**Phân tích**: File này giải thích cơ sở toán học đằng sau việc chọn ngưỡng tin cậy, giúp người đọc hiểu tại sao sử dụng giá trị `±0.1379` cho phân tích.

---

## 📊 Phân Tích Hình Ảnh Đầu Ra

### 1. `01_original_series.png`

**Mô tả**: Biểu đồ chuỗi thời gian gốc

**Phân tích**:
- Cho thấy xu hướng tăng trưởng mạnh theo thời gian
- Có tính mùa vụ rõ rệt (biến động lên xuống theo chu kỳ)
- Dữ liệu không dừng (cần sai phân)

### 2. `03_acf_pacf.png`

**Mô tả**: Biểu đồ ACF và PACF của chuỗi sau khi sai phân bậc 2

**Phân tích**:
- **ACF**: 
  - Lag 1 có giá trị cao (≈ -0.63)
  - Có đỉnh tại lag 12 → Gợi ý mạnh về tính mùa vụ
- **PACF**:
  - Lag 1 và 2 vượt ngưỡng → Gợi ý AR(2)
  - Cũng có đỉnh tại lag 12
- **Kết luận**: Cần mô hình có thành phần mùa vụ (SARIMA)

### 3. `04_residuals_(2, 2, 1).png` và `04_residuals_(1, 2, 1, 12).png`

**Mô tả**: Biểu đồ phần dư của ARIMA và SARIMA

**Phân tích**:
- **ARIMA**: Phần dư có vẻ còn một số pattern (không hoàn toàn ngẫu nhiên)
- **SARIMA**: Phần dư trông ngẫu nhiên hơn, phân tán đều quanh 0
- **Kết luận**: SARIMA có phần dư tốt hơn

### 4. `05_residuals_acf_(2, 2, 1).png` và `05_residuals_acf_(1, 2, 1, 12).png`

**Mô tả**: ACF của phần dư

**Phân tích**:
- **Lý tưởng**: Tất cả các lag nằm trong ngưỡng tin cậy (vùng xanh)
- **ARIMA**: Có thể còn một số lag vượt ngưỡng
- **SARIMA**: Hầu hết lag nằm trong ngưỡng → Phần dư gần với nhiễu trắng

### 5. `07_comparison_forecast.png`

**Mô tả**: So sánh dự báo giữa ARIMA và SARIMA với dữ liệu thực tế

**Phân tích**:
- **ARIMA (đường đỏ nét đứt)**: 
  - Chỉ dự báo được xu hướng phẳng
  - Không bắt được các biến động mùa vụ
  - Sai số lớn tại các điểm có biến động mạnh
  
- **SARIMA (đường xanh lá)**:
  - Tái hiện được các biến động hình răng cưa
  - Bắt được cả xu hướng và tính mùa vụ
  - Sát với dữ liệu thực tế (đường cam) hơn nhiều

- **Kết luận**: SARIMA vượt trội hoàn toàn trong việc dự báo dữ liệu có tính mùa vụ

---

## 🚀 Hướng Dẫn Sử Dụng

### Chạy Pipeline

```bash
python ARIMA_model.py
```

### Kết Quả Đầu Ra

Sau khi chạy, bạn sẽ có:

1. **Các file log** trong `logs/ARIMA_model/`:
   - `run_YYYYMMDD_HHMMSS.log`: Log tổng quan
   - `adf.log`: Kết quả kiểm định tính dừng
   - `suggested_lags.log`: Gợi ý bậc p, q
   - `information_criteria.log`: AIC/BIC
   - `evaluation.log`: MAE, RMSE, MAPE
   - `residuals.log`: Chẩn đoán phần dư

2. **Các file báo cáo** (`.txt`):
   - `stationarity_test_*.txt`: Báo cáo chi tiết về tính dừng
   - `ACF_PACF_Analysis_*.txt`: Phân tích ACF/PACF
   - `z_table_*.txt`: Bảng tra cứu Z-score

3. **Các biểu đồ** (`.png`):
   - `01_original_series.png`: Chuỗi gốc
   - `03_acf_pacf.png`: ACF/PACF
   - `04_residuals_*.png`: Phần dư
   - `05_residuals_acf_*.png`: ACF của phần dư
   - `07_comparison_forecast.png`: So sánh dự báo

### Tùy Chỉnh Tham Số

Bạn có thể chỉnh sửa các tham số trong hàm `main()`:

```python
# Thay đổi tỷ lệ train/test
train, test = split(ts, ratio=0.8)  # Mặc định 80/20

# Thay đổi tham số ARIMA
arima_order = (2, d_final, 1)  # (p, d, q)

# Thay đổi tham số SARIMA
seasonal_order = (1, 1, 1, 12)  # (P, D, Q, s)
```

---

## 📝 Kết Luận

### So Sánh ARIMA vs SARIMA

| Chỉ Số | ARIMA(2,2,1) | SARIMA(1,2,1)x(1,1,1,12) | Cải Thiện |
|--------|--------------|--------------------------|-----------|
| **AIC** | 597.03 | 279.03 | ↓ 53% |
| **BIC** | 609.35 | 294.05 | ↓ 52% |
| **MAE** | 3.64 | 1.99 | ↓ 45% |
| **RMSE** | 4.30 | 2.47 | ↓ 43% |
| **MAPE** | 21.51% | 11.55% | ↓ 46% |
| **Residual Mean** | 0.086 | 0.010 | ↓ 88% |

### Kết Luận Chính

1. **SARIMA vượt trội hoàn toàn** trong việc dự báo dữ liệu có tính mùa vụ
2. **Tính mùa vụ là yếu tố quan trọng** - ARIMA không thể bắt được pattern này
3. **Phần dư của SARIMA tốt hơn** - gần với nhiễu trắng hơn
4. **Mô hình SARIMA phù hợp hơn** cho dữ liệu doanh số bán lẻ hàng tháng

### Khuyến Nghị

- **Sử dụng SARIMA** cho dữ liệu có tính mùa vụ rõ rệt
- **Kiểm tra phần dư** để đảm bảo mô hình phù hợp
- **So sánh AIC/BIC** giữa các mô hình để lựa chọn tốt nhất
- **Đánh giá trên tập test** để kiểm tra khả năng dự báo thực tế
