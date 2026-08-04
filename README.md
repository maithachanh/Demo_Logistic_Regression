# Demo Logistic Regression

File này mô tả và minh họa cách hoạt động của thuật toán Logistic Regression thông qua một ví dụ đơn giản với các biến đầu vào là "Giờ học" và "Làm bài tập", còn nhãn đầu ra là "Kết quả".

## Mục đích

Script [Demo_Logistic_Regression.py](Demo_Logistic_Regression.py) thực hiện một demo về ba kiểu Logistic Regression:

- Binary Logistic Regression
- Multinomial Logistic Regression
- Ordinal Logistic Regression

Mỗi phiên bản sẽ dùng cùng một tập dữ liệu mẫu nhưng có cách mã hóa nhãn khác nhau để phù hợp với từng loại bài toán.

## Flow hoạt động

### 1. Chọn kiểu mô hình

Khi chạy script, chương trình sẽ hỏi người dùng lựa chọn kiểu Logistic Regression:

- binary
- multinomial
- ordinal

Nếu người dùng nhập không hợp lệ, chương trình sẽ tự động chuyển về kiểu binary.

### 2. Tạo dữ liệu mẫu

Script xây dựng một bảng dữ liệu gồm các cột:

- Sinh viên
- Giờ học
- Làm bài tập
- Kết quả

Dữ liệu này được dùng để minh họa mối quan hệ giữa các đặc trưng đầu vào và nhãn đầu ra.

### 3. Chuyển dữ liệu sang dạng số

Các thuộc tính dạng chữ được chuyển thành số để mô hình có thể học được:

- Giờ học: Thấp = 0, Trung bình = 1, Cao = 2
- Làm bài tập: Không đầy đủ = 0, Đầy đủ = 1
- Kết quả: được mã hóa thành các giá trị số tương ứng tùy theo kiểu mô hình

Quá trình này được gọi là encoding.

### 4. Xây dựng ma trận đặc trưng

Script tạo:

- Ma trận X chứa các đặc trưng đầu vào
- Vector y chứa nhãn mục tiêu

Để mô hình có thể học được hệ số chặn, chương trình thêm một cột bias vào đầu ma trận X.

### 5. Khởi tạo tham số

Tùy theo kiểu mô hình, chương trình sẽ khởi tạo các tham số khác nhau:

- Binary: vector trọng số w
- Multinomial: ma trận trọng số W
- Ordinal: vector beta và các thresholds

Các giá trị này được khởi tạo ngẫu nhiên với seed cố định để đảm bảo kết quả có thể lặp lại.

### 6. Huấn luyện bằng Gradient Descent

Script thực hiện huấn luyện qua nhiều epoch (25 epoch mặc định).

Trong mỗi epoch, chương trình sẽ:

- tính giá trị tuyến tính z hoặc eta
- tính xác suất dự đoán bằng sigmoid hoặc softmax
- tính loss
- tính gradient
- cập nhật trọng số

Quá trình này lặp lại để tối ưu hóa mô hình.

### 7. Dự đoán và đánh giá

Sau khi huấn luyện, script sẽ dùng trọng số đã học để dự đoán lại trên tập dữ liệu ban đầu.

Kết quả hiển thị gồm:

- xác suất dự đoán
- lớp dự đoán
- nhãn thực tế
- độ chính xác của mô hình trên tập dữ liệu hiện tại

### 8. Vẽ biểu đồ

Cuối cùng, chương trình vẽ biểu đồ để trực quan hóa:

- các điểm dữ liệu ban đầu
- vùng phân lớp dự đoán
- đường hoặc vùng xác suất tương ứng với từng mô hình

Nếu chạy ở môi trường không tương tác, biểu đồ sẽ được lưu thành file ảnh PNG; nếu chạy tương tác, biểu đồ sẽ được hiển thị trực tiếp.

## Cách chạy

Chạy lệnh sau:

```bash
python Demo_Logistic_Regression.py
```

## Output

Khi chạy xong, bạn sẽ thấy:

- bảng dữ liệu ban đầu và dữ liệu đã mã hóa
- các bước huấn luyện epoch-by-epoch
- kết quả dự đoán và độ chính xác
- biểu đồ minh họa

## Ghi chú

Đây là một demo giáo dục, nên dữ liệu khá nhỏ và đơn giản. Mục tiêu chính là giúp người học hiểu được luồng làm việc của Logistic Regression, không phải để xây dựng một mô hình production hoàn chỉnh.