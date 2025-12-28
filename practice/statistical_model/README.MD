# Phân Tích và Dự Báo Doanh Số Bán Lẻ bằng ARIMA và SARIMA

## Tổng Quan Dự Án

Dự án này thực hiện phân tích và dự báo chuỗi thời gian doanh số bán lẻ hàng tháng sử dụng các mô hình thống kê ARIMA (AutoRegressive Integrated Moving Average) và SARIMA (Seasonal ARIMA). Mục tiêu là so sánh hiệu quả giữa hai mô hình trong việc dự báo dữ liệu có xu hướng tăng trưởng mạnh và tính mùa vụ rõ rệt.

### Kết Quả Tóm Tắt

Dựa trên dữ liệu kiểm thử, mô hình **SARIMA** cho thấy sự vượt trội hoàn toàn:

- **Chỉ số AIC**: Giảm từ **597.03** (ARIMA) xuống **279.03** (SARIMA) - giảm 53%
- **Sai số MAPE**: Giảm từ **21.51%** xuống còn **11.55%** - giảm gần 50%
- **RMSE**: Giảm từ **4.30** xuống **2.47** - giảm 43%
- **Khả năng dự báo**: SARIMA tái hiện được các biến động hình răng cưa thực tế, trong khi ARIMA chỉ dự báo được xu hướng phẳng

---

## Cài Đặt và Yêu Cầu

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

## Cấu Trúc Dự Án

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

## Giải Thích Code Chi Tiết

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

## Phân Tích Chi Tiết 3 Báo Cáo Chính

Phần này tập trung phân tích chi tiết 3 file báo cáo quan trọng nhất trong quá trình xây dựng mô hình ARIMA/SARIMA:

---

## 1. Báo Cáo Kiểm Định Tính Dừng: `stationarity_test_YYYYMMDD_HHMMSS.txt`

### Nội Dung Báo Cáo

```
Stationarity Test Results (Augmented Dickey-Fuller Test)
============================================================
Original data length: 204
Date range: 1991-07-01 00:00:00 to 2008-06-01 00:00:00

============================================================
TEST RESULTS FOR DIFFERENT DIFFERENCING ORDERS
============================================================

Differencing Order d = 0 (Original):
  ADF Statistic: 3.1452
  p-value: 1.0000
  Critical Values:
    1%: -3.4656
    5%: -2.8770
    10%: -2.5750
  Stationary: No (p-value >= 0.05)
  Number of observations: 204

Differencing Order d = 1 (Differenced 1 time):
  ADF Statistic: -2.4952
  p-value: 0.1167
  Critical Values:
    1%: -3.4658
    5%: -2.8771
    10%: -2.5751
  Stationary: No (p-value >= 0.05)
  Number of observations: 203

Differencing Order d = 2 (Differenced 2 times):
  ADF Statistic: -10.2923
  p-value: 0.0000
  Critical Values:
    1%: -3.4658
    5%: -2.8771
    10%: -2.5751
  Stationary: Yes (p-value < 0.05) ✓
  Number of observations: 202

============================================================
Suggested Differencing Order (d): 2
```

### Phân Tích Chi Tiết

#### **Bước 1: Kiểm định chuỗi gốc (d = 0)**

| Chỉ Số | Giá Trị | Ý Nghĩa |
|--------|---------|---------|
| **ADF Statistic** | 3.1452 | Giá trị dương, rất xa các giá trị tới hạn âm |
| **p-value** | 1.0000 | Cực kỳ cao → Không thể bác bỏ H0 |
| **Critical Value (5%)** | -2.8770 | ADF Statistic (3.1452) > -2.8770 |
| **Kết luận** | **Không dừng** | Chuỗi có unit root, cần sai phân |

**Giải thích**: 
- ADF Statistic dương và lớn cho thấy chuỗi có xu hướng tăng mạnh
- p-value = 1.0 nghĩa là xác suất chuỗi dừng gần như bằng 0
- **Quyết định**: Cần thực hiện sai phân

#### **Bước 2: Kiểm định sau sai phân bậc 1 (d = 1)**

| Chỉ Số | Giá Trị | Ý Nghĩa |
|--------|---------|---------|
| **ADF Statistic** | -2.4952 | Đã chuyển sang âm nhưng chưa đủ nhỏ |
| **p-value** | 0.1167 | Vẫn > 0.05 → Chưa đạt ngưỡng ý nghĩa |
| **Critical Value (5%)** | -2.8771 | ADF Statistic (-2.4952) > -2.8771 |
| **Kết luận** | **Vẫn không dừng** | Cần sai phân thêm 1 lần nữa |

**Giải thích**:
- ADF Statistic đã âm nhưng chưa vượt qua ngưỡng -2.8771
- p-value = 0.117 > 0.05 → Chưa đủ bằng chứng để kết luận chuỗi dừng
- **Quyết định**: Cần sai phân bậc 2

#### **Bước 3: Kiểm định sau sai phân bậc 2 (d = 2)**

| Chỉ Số | Giá Trị | Ý Nghĩa |
|--------|---------|---------|
| **ADF Statistic** | -10.2923 | Rất âm, vượt xa tất cả giá trị tới hạn |
| **p-value** | 0.0000 | Cực kỳ nhỏ (< 0.0001) |
| **Critical Value (1%)** | -3.4658 | ADF Statistic (-10.29) << -3.47 |
| **Critical Value (5%)** | -2.8771 | ADF Statistic (-10.29) << -2.88 |
| **Kết luận** | **Dừng ✓** | Chuỗi đã đạt tính dừng |

**Giải thích**:
- ADF Statistic = -10.29 rất nhỏ, vượt xa cả ngưỡng 1% (-3.47)
- p-value ≈ 0 → Bằng chứng mạnh mẽ để bác bỏ H0
- **Quyết định cuối cùng**: Sử dụng **d = 2** cho mô hình ARIMA

### Bảng Tổng Hợp So Sánh

| Bậc Sai Phân | ADF Statistic | p-value | So với CV 5% | Kết Luận | Số Quan Sát |
|--------------|---------------|---------|--------------|----------|-------------|
| **d = 0** | 3.1452 | 1.0000 | > -2.8770 | Không dừng | 204 |
| **d = 1** | -2.4952 | 0.1167 | > -2.8771 | Không dừng | 203 |
| **d = 2** | **-10.2923** | **0.0000** | **<< -2.8771** | **Dừng** | 202 |

### Kết Luận

1. **Chuỗi gốc có xu hướng mạnh**: ADF Statistic dương (3.15) cho thấy xu hướng tăng rõ rệt
2. **Sai phân bậc 1 chưa đủ**: Vẫn còn một phần xu hướng chưa được loại bỏ
3. **Sai phân bậc 2 đạt tính dừng**: ADF Statistic = -10.29 cho thấy chuỗi đã hoàn toàn dừng
4. **Mất 2 quan sát**: Mỗi lần sai phân làm mất 1 quan sát (204 → 203 → 202)

---

## 📈 2. Báo Cáo Phân Tích ACF/PACF: `ACF_PACF_Analysis_YYYYMMDD_HHMMSS.txt`

### Nội Dung Báo Cáo

```
ACF and PACF Analysis for ARIMA Model Selection
============================================================
Data length: 202
Date range: 1991-09-01 00:00:00 to 2008-06-01 00:00:00
Mean: -0.02
Std deviation: 3.71

============================================================
THRESHOLD SELECTION EXPLANATION
============================================================
Sample size (n): 202

Confidence Interval Thresholds:
  90% CI threshold: ±0.1157
  95% CI threshold: ±0.1379 (used for analysis)
  99% CI threshold: ±0.1812

Threshold Used: 0.1379

Origin of Z-Score Values (1.65, 1.96, 2.58):
  These are critical values (z-scores) from the Standard Normal Distribution:
  - 1.65 ≈ z(0.95)  for 90% CI
  - 1.96 ≈ z(0.975) for 95% CI
  - 2.58 ≈ z(0.995) for 99% CI

  Statistical Basis:
  Under the null hypothesis (no autocorrelation),
  ACF/PACF values follow approximately N(0, 1/√n)
  CI = ±z_α/2 / √n

============================================================
Significant PACF lags (|PACF| > 0.1379) - for AR order:
  Lag 1: -0.6346
  Lag 2: -0.3673
  Lag 3: -0.3032
  Lag 4: -0.3189
  Lag 5: -0.2062
  Lag 6: -0.1817
  Lag 7: -0.1527
  Lag 10: 0.1872

============================================================
Significant ACF lags (|ACF| > 0.1379) - for MA order:
  Lag 1: -0.6346
  Lag 2: 0.1833
  Lag 10: 0.1771

============================================================
Suggested ARIMA order: AR(1), I(d), MA(1)
Note: Differencing order (d) is determined separately using ADF test
```

### Phân Tích Chi Tiết

#### **A. Thông Tin Cơ Bản về Dữ Liệu**

| Thông Số | Giá Trị | Ý Nghĩa |
|----------|---------|---------|
| **Độ dài chuỗi** | 202 | Sau khi sai phân bậc 2 (từ 204) |
| **Khoảng thời gian** | 1991-09 đến 2008-06 | ~17 năm dữ liệu |
| **Trung bình** | -0.02 | Gần 0 → Chuỗi đã được chuẩn hóa tốt |
| **Độ lệch chuẩn** | 3.71 | Độ biến thiên của chuỗi sau sai phân |

#### **B. Ngưỡng Tin Cậy (Confidence Thresholds)**

**Công thức tính**: `CI = ±z_(α/2) / √n`

Với n = 202:
- **90% CI**: ±1.645 / √202 = **±0.1157**
- **95% CI**: ±1.960 / √202 = **±0.1379** ← **Sử dụng cho phân tích**
- **99% CI**: ±2.576 / √202 = **±0.1812**

**Giải thích**: 
- Ngưỡng 95% được chọn vì cân bằng giữa độ chính xác và độ nhạy
- Giá trị ACF/PACF vượt quá ±0.1379 được coi là có ý nghĩa thống kê

#### **C. Phân Tích PACF (Partial Autocorrelation Function) - Xác định bậc AR (p)**

**Các lag có ý nghĩa thống kê** (|PACF| > 0.1379):

| Lag | PACF Value | |PACF| | Vượt Ngưỡng | Ý Nghĩa |
|-----|------------|-------|-------------|----------|
| **1** | -0.6346 | 0.6346 | Rất lớn | Tương quan trực tiếp mạnh với lag 1 |
| **2** | -0.3673 | 0.3673 | Lớn | Tương quan trực tiếp với lag 2 |
| **3** | -0.3032 | 0.3032 | Lớn | Tương quan trực tiếp với lag 3 |
| **4** | -0.3189 | 0.3189 | Lớn | Tương quan trực tiếp với lag 4 |
| **5** | -0.2062 | 0.2062 | Vừa | Tương quan trực tiếp với lag 5 |
| **6** | -0.1817 | 0.1817 | Vừa | Tương quan trực tiếp với lag 6 |
| **7** | -0.1527 | 0.1527 | Nhỏ | Tương quan trực tiếp với lag 7 |
| **10** | 0.1872 | 0.1872 | Vừa | Tương quan trực tiếp với lag 10 |

**Phân tích**:
- **Lag 1 có giá trị tuyệt đối lớn nhất** (-0.63) → Gợi ý mạnh về AR(1)
- **Lag 2 cũng rất lớn** (-0.37) → Có thể cần AR(2)
- **Nhiều lag vượt ngưỡng** → Có thể cần mô hình AR phức tạp hơn
- **Lag 10 có giá trị dương** (0.19) → Có thể liên quan đến tính mùa vụ

**Gợi ý bậc AR (p)**: 
- **p = 1**: Dựa trên lag 1 có giá trị lớn nhất
- **p = 2**: Nếu muốn bắt được tương quan với lag 2

#### **D. Phân Tích ACF (Autocorrelation Function) - Xác định bậc MA (q)**

**Các lag có ý nghĩa thống kê** (|ACF| > 0.1379):

| Lag | ACF Value | |ACF| | Vượt Ngưỡng | Ý Nghĩa |
|-----|----------|-------|-------------|----------|
| **1** | -0.6346 | 0.6346 | Rất lớn | Tương quan tổng thể mạnh với lag 1 |
| **2** | 0.1833 | 0.1833 | Vừa | Tương quan tổng thể với lag 2 |
| **10** | 0.1771 | 0.1771 | Vừa | Tương quan tổng thể với lag 10 |

**Phân tích**:
- **Lag 1 có giá trị tuyệt đối lớn nhất** (-0.63) → Gợi ý mạnh về MA(1)
- **Lag 2 và 10 có giá trị vừa** → Có thể cần MA(2) hoặc có tính mùa vụ
- **Ít lag vượt ngưỡng hơn PACF** → Mô hình MA đơn giản hơn

**Gợi ý bậc MA (q)**:
- **q = 1**: Dựa trên lag 1 có giá trị lớn nhất
- **q = 2**: Nếu muốn bắt được tương quan với lag 2

#### **E. Dấu Hiệu Tính Mùa Vụ**

**Quan sát quan trọng**:
- **Lag 10 và 12** (không hiển thị đầy đủ trong báo cáo nhưng có trong suggested_lags.log) xuất hiện trong cả ACF và PACF
- Điều này gợi ý mạnh về **tính mùa vụ với chu kỳ 12 tháng**
- **Kết luận**: Nên sử dụng **SARIMA** thay vì ARIMA đơn thuần

### Bảng Tổng Hợp Gợi ý Mô Hình

| Tham Số | Gợi ý từ PACF | Gợi ý từ ACF | Giá Trị Chọn | Lý Do |
|---------|---------------|--------------|--------------|-------|
| **p (AR)** | Lag 1, 2, 3, 4... | - | **1 hoặc 2** | Lag 1 và 2 có giá trị lớn nhất |
| **d (I)** | - | - | **2** | Từ ADF test |
| **q (MA)** | - | Lag 1, 2, 10 | **1 hoặc 2** | Lag 1 có giá trị lớn nhất |
| **s (Seasonal)** | Lag 10, 12 | Lag 10, 12 | **12** | Dấu hiệu mùa vụ rõ rệt |

**Mô hình đề xuất**:
- **ARIMA(1, 2, 1)** hoặc **ARIMA(2, 2, 1)** - Mô hình cơ bản
- **SARIMA(1, 2, 1)(1, 1, 1, 12)** - Mô hình có tính mùa vụ (khuyến nghị)

---

## 📐 3. Bảng Tra Cứu Z-Score: `z_table_YYYYMMDD_HHMMSS.txt`

### Nội Dung Báo Cáo

```
================================================================================
STANDARD NORMAL DISTRIBUTION (Z-TABLE)
Critical Values for ACF / PACF Confidence Intervals
================================================================================

Statistical assumption:
Under the null hypothesis of no autocorrelation,
sample ACF and PACF values are approximately distributed as:
  N(0, 1 / √n)

Confidence interval formula:
  CI = ± z_(α/2) / √n

================================================================================
COMMONLY USED CONFIDENCE LEVELS
================================================================================
Confidence Level      Two-Tailed α      Z-Score (zα/2)
--------------------------------------------------------------------------------
  80.0%              0.200               1.2816
  85.0%              0.150               1.4395
  90.0%              0.100               1.6449
  95.0%              0.050               1.9600
  99.0%              0.010               2.5758
  99.9%              0.001               3.2905

================================================================================
DETAILED Z-TABLE (COMMON PERCENTILES)
================================================================================
Percentile      Z-Score      Confidence Level     Usage
--------------------------------------------------------------------------------
 50.00%        0.0000      50%                 Median
 60.00%        0.2533      60%                 
 70.00%        0.5244      70%                 
 75.00%        0.6745      75%                 
 80.00%        0.8416      80%                 Common threshold
 85.00%        1.0364      85%                 
 90.00%        1.2816      90%                 Common threshold
 95.00%        1.6449      95%                 Very common
 97.50%        1.9600      97.5%               95% CI (two-tailed)
 99.00%        2.3263      99%                 Common threshold
 99.50%        2.5758      99.5%               99% CI (two-tailed)
 99.90%        3.0902      99.9%               High confidence
 99.95%        3.2905      99.95%              Very high confidence

================================================================================
APPLICATION TO YOUR DATASET
================================================================================
Sample size (n): 204

Calculated confidence thresholds:
  90% CI  : ±0.1152   (z = 1.645)
  95% CI  : ±0.1372   (z = 1.96)  ← used for analysis
  99% CI  : ±0.1803   (z = 2.576)

Interpretation for ACF/PACF plots:
- Any ACF or PACF value exceeding ±0.1372
  is considered statistically significant.
- Significant PACF lags suggest AR order (p).
- Significant ACF lags suggest MA order (q).
- Differencing order (d) is determined separately using ADF test.
```

### Phân Tích Chi Tiết

#### **A. Cơ Sở Toán Học**

**Giả định thống kê**:
- Dưới giả thuyết H0 (không có tự tương quan), các giá trị ACF/PACF mẫu tuân theo phân phối chuẩn:
  ```
  ACF/PACF ~ N(0, 1/√n)
  ```

**Công thức khoảng tin cậy**:
```
CI = ± z_(α/2) / √n
```

Trong đó:
- `z_(α/2)`: Z-score từ phân phối chuẩn chuẩn hóa
- `n`: Kích thước mẫu
- `α`: Mức ý nghĩa (significance level)

#### **B. Bảng Z-Score Phổ Biến**

| Mức Tin Cậy | Two-Tailed α | Z-Score | Ứng Dụng |
|-------------|--------------|---------|----------|
| **80%** | 0.200 | 1.2816 | Ngưỡng thấp, ít sử dụng |
| **85%** | 0.150 | 1.4395 | Ngưỡng trung bình |
| **90%** | 0.100 | 1.6449 | Ngưỡng phổ biến |
| **95%** | 0.050 | **1.9600** | **Rất phổ biến, tiêu chuẩn** |
| **99%** | 0.010 | 2.5758 | Ngưỡng cao, nghiêm ngặt |
| **99.9%** | 0.001 | 3.2905 | Ngưỡng rất cao |

**Giải thích**:
- **95% CI (z = 1.96)** là tiêu chuẩn vàng trong thống kê
- Cân bằng giữa độ chính xác và khả năng phát hiện tương quan
- Được sử dụng trong hầu hết các nghiên cứu khoa học

#### **C. Bảng Chi Tiết Các Phân Vị**

| Phân Vị | Z-Score | Mức Tin Cậy | Cách Sử Dụng |
|---------|---------|-------------|--------------|
| 50% | 0.0000 | 50% | Trung vị (median) |
| 75% | 0.6745 | 75% | Phân vị thứ 3 |
| 80% | 0.8416 | 80% | Ngưỡng thông thường |
| 90% | 1.2816 | 90% | Ngưỡng thông thường |
| **95%** | **1.6449** | **95%** | **Rất phổ biến** |
| **97.5%** | **1.9600** | **97.5%** | **95% CI (two-tailed)** ← **Dùng cho ACF/PACF** |
| 99% | 2.3263 | 99% | Ngưỡng thông thường |
| 99.5% | 2.5758 | 99.5% | 99% CI (two-tailed) |
| 99.9% | 3.0902 | 99.9% | Độ tin cậy cao |
| 99.95% | 3.2905 | 99.95% | Độ tin cậy rất cao |

**Lưu ý quan trọng**:
- **97.5% percentile (z = 1.96)** được dùng cho **95% CI two-tailed**
- Vì two-tailed test chia α thành 2 phần: α/2 ở mỗi đuôi
- 95% CI two-tailed = 2.5% ở mỗi đuôi = 97.5% percentile

#### **D. Áp Dụng Cho Dataset Cụ Thể**

**Thông tin dataset**:
- **Kích thước mẫu (n)**: 204 quan sát
- **Công thức**: CI = ± z / √204 = ± z / 14.28

**Bảng Ngưỡng Tin Cậy Tính Toán**:

| Mức Tin Cậy | Z-Score | Ngưỡng Tính Toán | Ngưỡng Làm Tròn |
|-------------|---------|------------------|-----------------|
| 90% | 1.6449 | ±0.1152 | **±0.1157** |
| **95%** | **1.9600** | **±0.1372** | **±0.1379** ← **Sử dụng** |
| 99% | 2.5758 | ±0.1803 | **±0.1812** |

**Giải thích**:
- Ngưỡng **±0.1379** được sử dụng để xác định các lag có ý nghĩa thống kê
- Bất kỳ giá trị ACF hoặc PACF nào vượt quá ngưỡng này đều được coi là có tương quan đáng kể
- Ngưỡng này đảm bảo 95% độ tin cậy trong việc phát hiện tự tương quan

#### **E. Ý Nghĩa Thực Tiễn**

**Quy tắc sử dụng**:

1. **Xác định bậc AR (p)**:
   - Xem biểu đồ PACF
   - Tìm lag đầu tiên mà giá trị PACF vượt quá ±0.1379
   - Lag đó chính là bậc AR đề xuất

2. **Xác định bậc MA (q)**:
   - Xem biểu đồ ACF
   - Tìm lag đầu tiên mà giá trị ACF vượt quá ±0.1379
   - Lag đó chính là bậc MA đề xuất

3. **Xác định tính mùa vụ**:
   - Nếu có lag 12, 24, 36... vượt ngưỡng → Có tính mùa vụ
   - Chu kỳ mùa vụ (s) = lag có giá trị cao nhất trong các lag mùa vụ

**Ví dụ từ dataset**:
- **PACF lag 1 = -0.6346** >> 0.1379 → Gợi ý AR(1) hoặc AR(2)
- **ACF lag 1 = -0.6346** >> 0.1379 → Gợi ý MA(1)
- **Lag 12 xuất hiện** → Gợi ý tính mùa vụ với s = 12

### Kết Luận về Z-Table

1. **Cơ sở toán học vững chắc**: Dựa trên phân phối chuẩn chuẩn hóa N(0,1)
2. **Ngưỡng 95% CI là tiêu chuẩn**: Cân bằng giữa độ chính xác và độ nhạy
3. **Áp dụng trực tiếp**: Ngưỡng ±0.1379 được tính toán cụ thể cho dataset này
4. **Hướng dẫn rõ ràng**: Giúp xác định bậc mô hình một cách khoa học

---

## Tổng Kết 3 Báo Cáo

### Quy Trình Xây Dựng Mô Hình Dựa Trên 3 Báo Cáo

```
1. stationarity_test.txt
   ↓
   Xác định d = 2 (sai phân bậc 2)

2. z_table.txt
   ↓
   Tính ngưỡng tin cậy ±0.1379 (95% CI)

3. ACF_PACF_Analysis.txt
   ↓
   Xác định p và q dựa trên ngưỡng
   ↓
   Kết quả: ARIMA(1,2,1) hoặc ARIMA(2,2,1)
   + Phát hiện tính mùa vụ → SARIMA(1,2,1)(1,1,1,12)
```

### Bảng Tổng Hợp Quyết Định

| Tham Số | Nguồn Xác Định | Giá Trị | Báo Cáo Tham Khảo |
|---------|----------------|---------|-------------------|
| **d** | ADF Test | **2** | `stationarity_test.txt` |
| **Ngưỡng CI** | Z-Table | **±0.1379** | `z_table.txt` |
| **p** | PACF Analysis | **1 hoặc 2** | `ACF_PACF_Analysis.txt` |
| **q** | ACF Analysis | **1** | `ACF_PACF_Analysis.txt` |
| **s** | ACF/PACF lag 12 | **12** | `ACF_PACF_Analysis.txt` |

### Mô Hình Cuối Cùng

Dựa trên 3 báo cáo, mô hình được đề xuất:

1. **ARIMA(2, 2, 1)**: Mô hình cơ bản, bỏ qua tính mùa vụ
2. **SARIMA(1, 2, 1)(1, 1, 1, 12)**: Mô hình đầy đủ, bao gồm tính mùa vụ **Khuyến nghị**

---

## Phân Tích Hình Ảnh Đầu Ra

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

## Hướng Dẫn Sử Dụng

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

## Kết Luận

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
