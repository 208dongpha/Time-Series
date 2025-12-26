# Phân Tích và Dự Báo Doanh Số Bán Lẻ (ARIMA vs SARIMA)
Dự án này tập trung vào việc xây dựng mô hình thống kê để dự báo doanh số bán lẻ hàng tháng. Trọng tâm của nghiên cứu là quy trình xử lý chuỗi thời gian không dừng, có xu hướng tăng trưởng mạnh và tính mùa vụ rõ rệt.
## Công Nghệ Sử Dụng
- Ngôn ngữ: Python 

- Thư viện chính: statsmodels (ARIMA/SARIMA), pandas, numpy, matplotlib.
## Tóm tắt kết quả thực nghiệm
Dựa trên dữ liệu kiểm thử, mô hình SARIMA cho thấy sự vượt trội hoàn toàn:
- Chỉ số AIC: Giảm từ 597.02 (ARIMA) xuống 245.41 (SARIMA).
- Sai số MAPE: Giảm từ 21.51% xuống còn 9.06%.
- Khả năng dự báo: SARIMA tái hiện được các biến động hình răng cưa thực tế, trong khi ARIMA chỉ dự báo được xu hướng phẳng.
## Quy Trình Thực Hiện Chi Tiết
1. Phân Tích Tính Dừng (Stationarity Analysis)
    Việc kiểm định tính dừng là bắt buộc để đảm bảo các tham số thống kê của chuỗi không thay đổi theo thời gian.
    - Kết quả kiểm định ADF:
            
        ![alt text]({31BCC5ED-DC96-4A6E-A794-9CE21BA7BBBC}.png)
        

        - Bậc 0 ($d=0$): $p$-value = 1.0 > 0.05 (Không dừng).
        - Bậc 1 ($d=1$): $p$-value = 0.116 > 0.05 (Không dừng).
        - Bậc 2 ($d=2$): $p$-value = $3.54 \times 10^{-18}$ < 0.05 (Dừng hoàn toàn).
    - Quyết định: Áp dụng sai phân bậc 2 ($d=2$) để triệt tiêu xu hướng.
2. Xác Định Tham Số ($p, q$) qua ACF/PACF

![alt text]({8CE48665-F8FA-457B-A6A6-7945792DCD23}.png)

Dựa trên biểu đồ ACF/PACF của chuỗi sau khi sai phân bậc 2:

![alt text](image.png)

- Tham số $p=2$ (Bậc tự hồi quy - AR):
    - Lý do: Biểu đồ PACF cho thấy hiện tượng "cắt cụt" (cut-off) sau lag 2, với hai cột đầu tiên vượt xa ngưỡng tin cậy 0.1379.
- Tham số $q=1$ (Bậc trung bình trượt - MA):
    - Lý do: Biểu đồ ACF cho thấy lag 1 là thành phần tương quan ngắn hạn rõ rệt nhất trước khi chuỗi đi vào vùng ổn định.
- Tham số mùa vụ: Xuất hiện tương quan cực cao tại lag 12 trên cả hai biểu đồ, gợi ý sử dụng chu kỳ mùa vụ $s=12$.