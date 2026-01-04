## Bài tập: Dự báo chuỗi giờ bằng XGBoost (Hourly Energy Consumption)

### Mục tiêu
- Chuyển chuỗi thời gian → supervised bằng lag features
- Huấn luyện XGBoost Regressor
- Đánh giá dự báo, kiểm tra residuals, phân tích feature importance

### Quy trình đã thực hiện (1-step ahead)
1) Chuẩn bị dữ liệu  
   - Đọc `AEP_hourly.csv`, parse `Datetime`, sort.  
   - Có thể đặt tần suất giờ và nội suy thiếu: `series = df['AEP_MW'].asfreq('H').interpolate('time')`.
2) Tạo lag features  
   - `p = 24`; tạo `lag_1 ... lag_24` bằng `shift(i)`; `dropna` sau đó.
3) Chia X – y  
   - `X`: các cột lag; `y`: `AEP_MW`.
4) Chia train / test theo thời gian  
   - 80% đầu train, 20% cuối test; không shuffle.
5) Huấn luyện XGBoost  
   - Tham số dùng: `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`, `objective='reg:squarederror'`.
6) Dự đoán & đánh giá  
   - Kết quả XGBoost (test): MAE ≈ 169.82, RMSE ≈ 225.80, MAPE ≈ 1.18%.  
   - Đã vẽ thực tế vs dự báo, residuals.
7) Kiểm tra residuals  
   - ACF/PACF residuals được vẽ để xem còn cấu trúc hay không.
8) Phân tích feature importance  
   - Dùng `feature_importances_` và bar chart; xem lag nào quan trọng nhất.
9) So sánh Linear Regression (baseline)  
   - MAE ≈ 147.85, RMSE ≈ 203.31, MAPE ≈ 1.02% (tốt hơn nhẹ trên cấu hình này).

### Giải thích code chính
- Tạo lag: `shift(i)` sinh cột quá khứ, tránh leakage nhờ `dropna` rồi mới split.  
- Chia tập: `iloc[:split]` và `iloc[split:]` giữ thứ tự thời gian.  
- Huấn luyện XGBoost: `XGBRegressor(...).fit(X_train, y_train)`.  
- Đánh giá: MAE/RMSE/MAPE, vẽ đường thực tế vs dự báo, residuals và ACF/PACF residuals.  
- Feature importance: lấy trọng số theo gain để biết lag nào tác động mạnh.

### Nâng cao (gợi ý)
1) Thêm rolling mean/std (24h, 168h) trên chuỗi đã `shift(1)` để tránh leakage.  
2) Dự báo multi-step: recursive hoặc mô hình direct nhiều bước.  
3) Tune hyperparameter với `TimeSeriesSplit` + Grid/Random search.  
4) Thử Random Forest và so sánh với XGBoost/LR.

### Câu hỏi thảo luận
1) XGBoost vs Linear Regression: XGBoost bắt phi tuyến/tương tác lag, thường tốt hơn LR tuyến tính; LR là baseline nhanh.  
2) Tăng số lag `p`: thêm thông tin dài hạn nhưng dễ quá khớp và tốn tài nguyên; chọn `p` qua validation.  
3) Khi nào XGBoost phù hợp hơn ARIMA: dữ liệu phi tuyến, cần thêm đặc trưng thủ công (rolling, holiday, weather), dữ liệu lớn/phức tạp; ARIMA hợp cấu trúc tuyến tính + mùa vụ rõ và ít đặc trưng phụ.

### Dataset
- `AEP_hourly.csv`: tiêu thụ điện năng giờ của PJM (hơn 10 năm), đơn vị MW.

### Trích dẫn code & biểu đồ (trong `practice/practice3/test.ipynb`)
- Sinh lag, chia tập, huấn luyện XGBoost & tính MAE/RMSE/MAPE: các ô sau phần tạo lag (X_train/X_test) và ô “Dự đoán và đánh giá”.
- Biểu đồ:  
  - Thực tế vs dự báo, residuals (ô “Thực tế vs dự báo (test) và residuals”).  
  - ACF/PACF của residuals (ô “ACF và PACF của residuals”).  
  - Feature importance (ô bar chart cuối + `xgb.plot_importance`).  
- So sánh Linear Regression: ô LR ngay sau phần XGBoost, in MAE/RMSE/MAPE.  
- Chỉ số hiện có: XGBoost test — MAE ≈ 169.82, RMSE ≈ 225.80, MAPE ≈ 1.18%; Linear Regression — MAE ≈ 147.85, RMSE ≈ 203.31, MAPE ≈ 1.02%.
