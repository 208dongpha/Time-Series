# XGBoost – Practice 4

README này giải thích cách chạy `XGBoost.py`, luồng xử lý, các chỉ số đánh giá, và nhận xét nhanh về các ảnh trong `images/`.

## 1. Mục tiêu

- Dự báo `load` bằng XGBoost với cửa sổ trễ 24 giờ.
- Thêm rolling features cho `load` (mean, std, trend).
- Đánh giá bằng RMSE, MAE, MAPE trên tập train/test.
- Lưu log và ảnh minh hoạ.

## 2. Cấu trúc thư mục

- `XGBoost.py`: mã nguồn chạy mô hình.
- `logs/`: log theo từng lần chạy (tên file có timestamp).
- `images/`: hình prediction và residuals.
- Dữ liệu `dataclean.csv` nằm ở `practice/practice4/dataclean.csv`.

## 3. Cách chạy

Từ thư mục `timeseries`:

```bash
python practice/practice4/XGBoost/XGBoost.py
```

## 4. Tóm tắt luồng xử lý trong code

1. **Load dữ liệu**  
   Đọc `dataclean.csv`, parse `Datetime`, sắp theo thời gian.

2. **Chuẩn hoá dữ liệu**  
   Dùng `StandardScaler` cho 4 biến `load`, `temp`, `humidity`, `wind_speed`.

3. **Feature Engineering**  
   - Lag 1..24 cho tất cả biến.  
   - Rolling features (chỉ dùng quá khứ):
     - `load_roll_mean_6`
     - `load_roll_std_24`
     - `load_trend_24`
   - Target: `load_{t+1}` (shift -1).

4. **Chia tập theo thời gian (70/15/15)**  
   `train/val/test` không shuffle.

5. **Huấn luyện XGBoost**  
   `XGBRegressor` với các tham số mặc định trong code.

6. **Dự báo & đánh giá**  
   Dự báo trên train/test, inverse scale về đơn vị gốc, tính RMSE/MAE/MAPE.

## 5. Kết quả đánh giá (log mẫu)

Từ `xgb_log_20260118_121206.txt`:
- **Train**: RMSE 232.5479, MAE 173.8041, MAPE 0.55%
- **Test**: RMSE 894.1787, MAE 664.3313, MAPE 2.36%

**Nhận xét nhanh**:
- Train thấp, test cao hơn → có dấu hiệu overfitting nhẹ.
- MAPE test ~2.36% là khá tốt so với VAR baseline.

## 6. Nhận xét ảnh trong `images/`

**`xgb_prediction_train.png`**
- Dự báo bám sát chuỗi thật trên train, sai số nhỏ.

**`xgb_prediction_test.png`**
- Dự báo bám theo xu hướng và chu kỳ ngày/tuần.
- Có lệch ở một số đỉnh/đáy lớn nhưng nhìn chung sát hơn VAR.

**`xgb_residuals_train.png`**
- Residuals nhỏ, tập trung gần 0 → mô hình học tốt trên train.

**`xgb_residuals_test.png`**
- Residuals phân bố quanh 0, biên độ lớn hơn train.
- Có vài spike âm/dương → các biến động bất thường chưa được học hết.