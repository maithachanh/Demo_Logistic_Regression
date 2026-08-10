# Demo Logistic Regression

Dự án này minh họa chi tiết và trực quan hóa cách hoạt động của thuật toán **Logistic Regression** trong Machine Learning với 3 biến thể chính:
1. **Binary Logistic Regression** (Phân loại Nhị phân: Trượt / Đậu)
2. **Multinomial Logistic Regression / Softmax Regression** (Phân loại Đa lớp độc lập: Trượt / Đậu / Xuất sắc)
3. **Ordinal Logistic Regression / Proportional Odds Model** (Phân loại Đa lớp có thứ tự: Trượt < Đậu < Xuất sắc)

---

## 🌟 Các Cải Tiến & Nâng Cấp Trong Phiên Bản Mới

1. **Cấu Trúc Hướng Đối Tượng (OOP)**:
   - Tách biệt rõ ràng thành các lớp `BinaryLogisticRegression`, `MultinomialLogisticRegression`, `OrdinalLogisticRegression` kế thừa từ `BaseLogisticRegression`.
   - Giúp mã nguồn sạch đẹp, đóng gói logic huấn luyện (`fit`), dự đoán (`predict`, `predict_proba`), và đánh giá (`evaluate`).

2. **Khắc Phục Lỗi Đa Cộng Tuyến Trong Ordinal Logistic Regression**:
   - Trong mô hình Cumulative Logit (Proportional Odds), các điểm ngưỡng (thresholds/cutpoints $\theta_k$) đã đóng vai trò làm hệ số chặn phân tách các lớp.
   - Việc đưa thêm cột bias ($1$) vào ma trận $X$ dẫn tới hiện tượng đa cộng tuyến hoàn hảo giữa bias $w_0$ và $\theta_k$. Mô hình mới đã loại bỏ cột bias dư thừa khi huấn luyện Ordinal Logistic Regression để tham số hội tụ chính xác.

3. **Hỗ Trợ Giao Diện Dòng Lệnh (CLI Arguments)**:
   - Thêm bộ xử lý `argparse` cho phép chạy tham số qua terminal linh hoạt mà không cần nhập thủ công.
   - Vẫn giữ nguyên chế độ tương tác hỏi đáp (`input()`) nếu không truyền cờ tham số.

4. **Trực Quan Hóa Đôi (Dual Subplots Visualization)**:
   - **Subplot 1**: Trực quan hóa đường ranh giới quyết định (Decision Boundary line $w_0 + w_1 x_1 + w_2 x_2 = 0$) và vùng xác suất phân lớp.
   - **Subplot 2**: Đồ thị theo dõi lịch sử giảm hàm mất mát (Loss Curve) qua từng Epoch giúp người học hiểu rõ sự hội tụ của thuật toán Gradient Descent.

---

## 📐 Cơ Sở Toán Học

### 1. Binary Logistic Regression
- **Hàm kích hoạt**: Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$, với $z = X \cdot w$.
- **Hàm mất mát**: Binary Cross-Entropy Loss:
  $$L(w) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]$$
- **Gradient**:
  $$\nabla_w L = \frac{1}{N} X^T (\hat{y} - y)$$

### 2. Multinomial Logistic Regression (Softmax)
- **Hàm kích hoạt**: Softmax cho $K$ lớp:
  $$P(Y_i = k) = \frac{e^{z_{i, k}}}{\sum_{j=1}^K e^{z_{i, j}}}, \quad Z = X \cdot W^T$$
- **Hàm mất mát**: Categorical Cross-Entropy Loss:
  $$L(W) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K y_{i, k} \ln(\hat{y}_{i, k})$$
- **Gradient**:
  $$\nabla_W L = \frac{1}{N} (\hat{Y} - Y_{\text{onehot}})^T X$$

### 3. Ordinal Logistic Regression (Proportional Odds)
- **Xác suất tích lũy**:
  $$\gamma_{i, k} = P(Y_i \le k) = \sigma(\theta_k - X_{\text{raw}, i} \cdot \beta), \quad k = 0, \dots, K-2$$
- **Xác suất từng lớp**:
  $$P(Y_i = 0) = \gamma_{i, 0}, \quad P(Y_i = k) = \gamma_{i, k} - \gamma_{i, k-1}, \quad P(Y_i = K-1) = 1 - \gamma_{i, K-2}$$

---

## 🚀 Hướng Dẫn Sử Dụng

### Cách 1: Chạy Tương Tác (Interactive Mode)
Chạy lệnh bên dưới và chọn kiểu mô hình khi được hỏi:
```bash
python Demo_Logistic_Regression.py
```
Nhập một trong các lựa chọn: `binary`, `multinomial`, `ordinal`, hoặc `all`.

### Cách 2: Chạy Qua Dòng Lệnh (CLI Flags)

1. **Chạy Binary Logistic Regression với 100 Epochs**:
   ```bash
   python Demo_Logistic_Regression.py --model binary --epochs 100 --lr 0.1
   ```

2. **Chạy tất cả mô hình cùng lúc**:
   ```bash
   python Demo_Logistic_Regression.py --model all --epochs 50
   ```

3. **Chạy chế độ ản log Epochs chi tiết (`--quiet`) và không hiển thị cửa sổ plot (`--no-plot`)**:
   ```bash
   python Demo_Logistic_Regression.py --model ordinal --quiet --no-plot
   ```

### Danh sách Tham số CLI:
- `--model`: Kiểu mô hình (`binary`, `multinomial`, `ordinal`, `all`).
- `--epochs`: Số lượng Epoch huấn luyện (Mặc định: `50`).
- `--lr`: Tốc độ học Learning Rate (Mặc định: `0.1`).
- `--seed`: Seed khởi tạo ngẫu nhiên (Mặc định: `42`).
- `--no-plot`: Không hiển thị/vẽ biểu đồ.
- `--quiet`: Ẩn chi tiết từng Epoch trong quá trình huấn luyện.

---

## 📊 File Đầu Ra (Output)

Khi chạy xong, chương trình sẽ xuất ra các file ảnh biểu đồ PNG tại thư mục dự án:
- `logistic_regression_plot_binary.png`
- `logistic_regression_plot_multinomial.png`
- `logistic_regression_plot_ordinal.png`

Mỗi file ảnh chứa 2 biểu đồ trực quan hóa:
1. **Bên trái**: Phân vùng dự đoán & Ranh giới phân lớp (Decision Boundary).
2. **Bên phải**: Lịch sử hàm mất mát (Loss Curve) giảm dần qua các Epochs.