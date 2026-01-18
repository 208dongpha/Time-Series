# VAR Model – Practice 4

README này giải thích cách chạy `VARmodel.py`, luồng xử lý, các chỉ số đánh giá, và nhận xét nhanh về các ảnh kết quả trong thư mục `images/`.

## 1. Mục tiêu

- Xây dựng mô hình VAR cho chuỗi thời gian đa biến (load + thời tiết).
- Chọn bậc trễ tối ưu theo AIC.
- Dự báo `load` và đánh giá bằng RMSE, MAE, MAPE.
- Lưu log và ảnh minh hoạ.

## 2. Cấu trúc thư mục

- `VARmodel.py`: mã nguồn chạy mô hình.
- `logs/`: log theo từng lần chạy (tên file có timestamp).
- `images/`: hình kết quả (AIC, dự báo, residuals).
- Dữ liệu `dataclean.csv` nằm ở `practice/practice4/dataclean.csv`.

## 3. Cách chạy

Từ thư mục `timeseries`:

```bash
python practice/practice4/VARmodel/VARmodel.py
```

Mỗi lần chạy sẽ tạo log mới theo timestamp trong `logs/`.

## 4. Tóm tắt luồng xử lý trong code

1. **Load dữ liệu**  
   Đọc `dataclean.csv`, parse `Datetime`, sắp theo thời gian.

2. **Chuẩn hoá dữ liệu**  
   Dùng `StandardScaler` cho tất cả biến để ổn định việc huấn luyện.

3. **Chia tập theo thời gian (70/15/15)**  
   `train/val/test` không shuffle.

4. **Kiểm định ADF (d=0 → d=1)**  
   In p‑value cho từng biến ở d=0. Nếu còn biến chưa dừng thì sai phân bậc 1 và kiểm tra lại.

5. **Differencing có điều kiện**  
   Nếu d=1 được dùng, sai phân bậc 1; nếu d=0 đã dừng, giữ nguyên dữ liệu.

6. **Chọn lag theo AIC**  
   Tính AIC cho lag 1..24, chọn lag nhỏ nhất.

7. **Huấn luyện VAR**  
   Train trên `train + val`.

8. **Walk-forward forecast**  
   Dự báo từng bước trên test, mỗi bước cập nhật history bằng giá trị thật.

9. **Hoàn nguyên & đánh giá**  
   Hoàn nguyên differencing, sau đó inverse StandardScaler để đưa về đơn vị gốc.
   Tính RMSE, MAE, MAPE cho `load` trên tập test.

## 5. Kết quả đánh giá (log mẫu)

Trong log gần nhất, các chỉ số được ghi ở cuối file trong `logs/`.

**Nhận xét nhanh**:
- MAPE thường thấp hơn nhiều so với khi tính trên dữ liệu đã chuẩn hoá.
- RMSE/MAE phản ánh sai số tuyệt đối; cần so sánh với biên độ `load` để đánh giá.
- Kết quả có thể cải thiện nếu tối ưu thêm lag, thử BIC, hoặc tăng preprocessing.

## 6. Nhận xét các ảnh trong `images/`

### `var_lag_selection.png`
- Đường AIC giảm dần và đạt thấp nhất tại **lag = 24**.
- Điều này hợp lý với dữ liệu theo giờ vì 24 thể hiện chu kỳ 1 ngày.

### `var_prediction.png`
- Đường dự báo bám xu hướng tổng thể của `load`.
- Vẫn có độ trễ nhẹ và biên độ dao động chưa theo sát các đỉnh/đáy lớn.
- Đây là đặc trưng thường gặp ở VAR baseline.

### `var_residuals.png`
- Residuals phân bố quanh 0, không bị lệch mạnh.
- Vẫn còn các cụm sai số lớn → mô hình chưa bắt được các biến động bất thường.
