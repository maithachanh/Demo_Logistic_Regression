import sys
import io
import numpy as np
import pandas as pd
import matplotlib

# Force UTF-8 output to handle Vietnamese characters reliably on Windows
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

interactive_display = sys.stdout.isatty() or sys.stderr.isatty()
if not interactive_display:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

def print_boxed_df(df, title):
    cols = df.columns.tolist()
    col_widths = [max(len(str(col)), *(len(str(x)) for x in df[col])) for col in cols]
    border = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    title_border = "+" + "-" * (len(title) + 2) + "+"
    print(title_border)
    print(f"| {title} |")
    print(title_border)
    print(border)
    header = "|" + "|".join(f" {col.center(w)} " for col, w in zip(cols, col_widths)) + "|"
    print(header)
    print(border)
    for _, row in df.iterrows():
        row_text = "|" + "|".join(f" {str(row[col]).ljust(w)} " for col, w in zip(cols, col_widths)) + "|"
        print(row_text)
    print(border)

# 0. Chọn kiểu Logistic Regression
try:
    model_choice = input("Chọn kiểu Logistic Regression (binary/multinomial/ordinal) [binary]: ").strip().lower()
except Exception:
    model_choice = "binary"
if model_choice not in {"binary", "multinomial", "ordinal"}:
    print(f"Lựa chọn không hợp lệ: {model_choice}. Chuyển sang binary.")
    model_choice = "binary"

model_name = {
    "binary": "Binary Logistic Regression",
    "multinomial": "Multinomial Logistic Regression",
    "ordinal": "Ordinal Logistic Regression",
}.get(model_choice, "Binary Logistic Regression")

print(f"\n>>> Kiểu giải: {model_name}")

if model_choice == "binary":
    raw_data = [
        {"Sinh viên": "A", "Giờ học": "Thấp", "Làm bài tập": "Không đầy đủ", "Kết quả": "Trượt"},
        {"Sinh viên": "B", "Giờ học": "Thấp", "Làm bài tập": "Đầy đủ", "Kết quả": "Trượt"},
        {"Sinh viên": "C", "Giờ học": "Trung bình", "Làm bài tập": "Không đầy đủ", "Kết quả": "Trượt"},
        {"Sinh viên": "D", "Giờ học": "Trung bình", "Làm bài tập": "Đầy đủ", "Kết quả": "Đậu"},
        {"Sinh viên": "E", "Giờ học": "Cao", "Làm bài tập": "Không đầy đủ", "Kết quả": "Đậu"},
        {"Sinh viên": "F", "Giờ học": "Cao", "Làm bài tập": "Đầy đủ", "Kết quả": "Đậu"},
        {"Sinh viên": "G", "Giờ học": "Trung bình", "Làm bài tập": "Đầy đủ", "Kết quả": "Đậu"},
        {"Sinh viên": "H", "Giờ học": "Thấp", "Làm bài tập": "Đầy đủ", "Kết quả": "Trượt"},
    ]
    result_mapping = {"Trượt": 0, "Đậu": 1}
    result_label_text = "Kết quả: Trượt=0, Đậu=1"

elif model_choice == "multinomial":
    raw_data = [
        {"Sinh viên": "A", "Giờ học": "Thấp", "Làm bài tập": "Không đầy đủ", "Kết quả": "Trượt"},
        {"Sinh viên": "B", "Giờ học": "Thấp", "Làm bài tập": "Đầy đủ", "Kết quả": "Đậu"},
        {"Sinh viên": "C", "Giờ học": "Trung bình", "Làm bài tập": "Không đầy đủ", "Kết quả": "Đậu"},
        {"Sinh viên": "D", "Giờ học": "Trung bình", "Làm bài tập": "Đầy đủ", "Kết quả": "Xuất sắc"},
        {"Sinh viên": "E", "Giờ học": "Cao", "Làm bài tập": "Không đầy đủ", "Kết quả": "Đậu"},
        {"Sinh viên": "F", "Giờ học": "Cao", "Làm bài tập": "Đầy đủ", "Kết quả": "Xuất sắc"},
        {"Sinh viên": "G", "Giờ học": "Trung bình", "Làm bài tập": "Đầy đủ", "Kết quả": "Xuất sắc"},
        {"Sinh viên": "H", "Giờ học": "Thấp", "Làm bài tập": "Đầy đủ", "Kết quả": "Đậu"},
    ]
    result_mapping = {"Trượt": 0, "Đậu": 1, "Xuất sắc": 2}
    result_label_text = "Kết quả: Trượt=0, Đậu=1, Xuất sắc=2"

else:
    raw_data = [
        {"Sinh viên": "A", "Giờ học": "Thấp", "Làm bài tập": "Không đầy đủ", "Kết quả": "Trượt"},
        {"Sinh viên": "B", "Giờ học": "Thấp", "Làm bài tập": "Đầy đủ", "Kết quả": "Đậu"},
        {"Sinh viên": "C", "Giờ học": "Trung bình", "Làm bài tập": "Không đầy đủ", "Kết quả": "Đậu"},
        {"Sinh viên": "D", "Giờ học": "Trung bình", "Làm bài tập": "Đầy đủ", "Kết quả": "Xuất sắc"},
        {"Sinh viên": "E", "Giờ học": "Cao", "Làm bài tập": "Không đầy đủ", "Kết quả": "Đậu"},
        {"Sinh viên": "F", "Giờ học": "Cao", "Làm bài tập": "Đầy đủ", "Kết quả": "Xuất sắc"},
        {"Sinh viên": "G", "Giờ học": "Trung bình", "Làm bài tập": "Đầy đủ", "Kết quả": "Xuất sắc"},
        {"Sinh viên": "H", "Giờ học": "Thấp", "Làm bài tập": "Đầy đủ", "Kết quả": "Đậu"},
    ]
    result_mapping = {"Trượt": 0, "Đậu": 1, "Xuất sắc": 2}
    result_label_text = "Kết quả: Trượt=0, Đậu=1, Xuất sắc=2 (thứ tự Ordinal)"

print()
df = pd.DataFrame(raw_data)
print_boxed_df(df, "1) Dữ liệu ban đầu")

# 2. Encoding dữ liệu sang số
hour_mapping = {"Thấp": 0, "Trung bình": 1, "Cao": 2}
homework_mapping = {"Không đầy đủ": 0, "Đầy đủ": 1}

if model_choice == "binary":
    result_mapping = {"Trượt": 0, "Đậu": 1}
    result_label_text = "Kết quả: Trượt=0, Đậu=1"
elif model_choice == "multinomial":
    result_mapping = {"Trượt": 0, "Đậu": 1, "Xuất sắc": 2}
    result_label_text = "Kết quả: Trượt=0, Đậu=1, Xuất sắc=2"
else:
    result_mapping = {"Trượt": 0, "Đậu": 1, "Xuất sắc": 2}
    result_label_text = "Kết quả: Trượt=0, Đậu=1, Xuất sắc=2 (thứ tự Ordinal)"

print("\n2) Chuyển dữ liệu sang dạng số (Encoding):")
print("Giờ học: Thấp=0, Trung bình=1, Cao=2")
print("Làm bài tập: Không đầy đủ=0, Đầy đủ=1")
print(result_label_text)

X_num = np.array([
    [hour_mapping[row["Giờ học"]], homework_mapping[row["Làm bài tập"]]]
    for _, row in df.iterrows()
], dtype=float)
y = np.array([result_mapping[row["Kết quả"]] for _, row in df.iterrows()], dtype=int)

encoded_df = df.copy()
encoded_df["Giờ học"] = encoded_df["Giờ học"].map(hour_mapping)
encoded_df["Làm bài tập"] = encoded_df["Làm bài tập"].map(homework_mapping)
encoded_df["Kết quả"] = encoded_df["Kết quả"].map(result_mapping)
print()
print_boxed_df(encoded_df, "2) Chuyển dữ liệu sang dạng số (Encoding)")

# 3. Xây dựng ma trận đặc trưng X và nhãn y
print("\n3) Ma trận đặc trưng X và nhãn y:")
print("X = [[Giờ học, Làm bài tập], ...]")
print(X_num)
print("y = [Kết quả ...]")
print(y)

# Gắn thêm bias bằng cột 1 để mô hình có hệ số chặn
X = np.hstack([np.ones((X_num.shape[0], 1)), X_num])
print("\nThêm cột bias vào X để có dạng [1, x1, x2]:")
print(X)

# 4. Khởi tạo trọng số
np.random.seed(42)

# Các hàm hỗ trợ

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    z_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return z_exp / np.sum(z_exp, axis=1, keepdims=True)

def categorical_cross_entropy(y_true_onehot, y_pred):
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(np.sum(y_true_onehot * np.log(y_pred), axis=1))

def ordinal_loss_and_grad(X, y, beta, thresholds):
    n = len(y)
    K = len(thresholds) + 1
    eta = X.dot(beta)
    scores = thresholds - eta[:, None]
    s = sigmoid(scores)
    y_cum = (y[:, None] <= np.arange(K - 1)).astype(float)
    epsilon = 1e-9
    s = np.clip(s, epsilon, 1 - epsilon)
    loss = -np.mean(np.sum(y_cum * np.log(s) + (1 - y_cum) * np.log(1 - s), axis=1))
    diff = s - y_cum
    grad_beta = -X.T.dot(np.sum(diff, axis=1)) / n
    grad_thresholds = np.mean(diff, axis=0)
    return loss, grad_beta, grad_thresholds

if model_choice == "binary":
    weights = np.random.randn(X.shape[1]) * 0.1
    print("\n4) Khởi tạo trọng số (weights) ban đầu:")
    print(weights)
elif model_choice == "multinomial":
    num_classes = len(result_mapping)
    weights = np.random.randn(num_classes, X.shape[1]) * 0.1
    print("\n4) Khởi tạo trọng số (weight matrix) ban đầu:")
    print(weights)
else:
    num_classes = len(result_mapping)
    beta = np.random.randn(X.shape[1]) * 0.1
    thresholds = np.linspace(-0.5, num_classes - 1.5, num_classes - 1)
    print("\n4) Khởi tạo beta và thresholds ban đầu:")
    print("beta =", beta)
    print("thresholds =", thresholds)

# 5. Huấn luyện
learning_rate = 0.1
epochs = 25
print("\n5-10) Huấn luyện Logistic Regression bằng Gradient Descent:")
for epoch in range(1, epochs + 1):
    if model_choice == "binary":
        linear_combination = X.dot(weights)
        y_pred = sigmoid(linear_combination)
        epsilon = 1e-9
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
        gradient = X.T.dot(y_pred - y) / len(y)
        weights -= learning_rate * gradient
        print(f"\nEpoch {epoch}")
        print("Linear combination z = X * w:")
        for i, z in enumerate(linear_combination):
            print(f"  mẫu {i + 1}: z = {X[i].tolist()} · {weights.tolist()} = {z:.6f}")
        print("Sigmoid y_hat = 1 / (1 + exp(-z)):")
        for i, p in enumerate(y_pred):
            print(f"  mẫu {i + 1}: y_hat = {p:.6f}")
        print(f"Loss (Binary Cross Entropy): {loss:.6f}")
        print("Gradient:")
        print(gradient)
        print("Weights sau cập nhật:")
        print(weights)

    elif model_choice == "multinomial":
        linear_combination = X.dot(weights.T)
        y_pred = softmax(linear_combination)
        y_onehot = np.eye(num_classes)[y]
        loss = categorical_cross_entropy(y_onehot, y_pred)
        gradient = (y_pred - y_onehot).T.dot(X) / len(y)
        weights -= learning_rate * gradient
        print(f"\nEpoch {epoch}")
        print("Linear combination z = X * W^T (n_samples x n_classes):")
        print(linear_combination)
        print("Softmax y_hat:")
        print(y_pred)
        print("Nhận dạng one-hot cho nhãn thực tế:")
        print(y_onehot)
        print(f"Loss (Categorical Cross Entropy): {loss:.6f}")
        print("Gradient ma trận W:")
        print(gradient)
        print("Weights sau cập nhật:")
        print(weights)

    else:
        loss, grad_beta, grad_thresholds = ordinal_loss_and_grad(X, y, beta, thresholds)
        beta -= learning_rate * grad_beta
        thresholds -= learning_rate * grad_thresholds
        thresholds = np.sort(thresholds)
        print(f"\nEpoch {epoch}")
        print("Eta = X · beta:")
        print(X.dot(beta))
        print("Thresholds hiện tại:")
        print(thresholds)
        print("Scores = thresholds - eta")
        print((thresholds - X.dot(beta)[:, None]))
        print("Xác suất tích lũy s = sigmoid(scores):")
        print(sigmoid(thresholds - X.dot(beta)[:, None]))
        print(f"Loss (Ordinal proportional odds): {loss:.6f}")
        print("Gradient beta:")
        print(grad_beta)
        print("Gradient thresholds:")
        print(grad_thresholds)

# 11. Dự đoán
print("\n11) Dự đoán với trọng số đã học:")
if model_choice == "binary":
    linear_combination_final = X.dot(weights)
    y_pred_final = sigmoid(linear_combination_final)
    predictions = (y_pred_final >= 0.5).astype(int)
    for i in range(len(y)):
        percent = round(y_pred_final[i] * 100, 3)
        print(
            f"Mẫu {i + 1}: X = {X_num[i].tolist()}, z = {linear_combination_final[i]:.6f}, "
            f"y_hat = {y_pred_final[i]:.6f} ({percent}%), dự đoán = {predictions[i]}, thực tế = {int(y[i])}"
        )
    accuracy = np.mean(predictions == y)
    print(f"\nĐộ chính xác trên tập dữ liệu: {accuracy * 100:.2f}%")

elif model_choice == "multinomial":
    linear_combination_final = X.dot(weights.T)
    y_pred_final = softmax(linear_combination_final)
    predictions = np.argmax(y_pred_final, axis=1)
    for i in range(len(y)):
        probs = ", ".join([f"{p:.3f}" for p in y_pred_final[i]])
        print(
            f"Mẫu {i + 1}: X = {X_num[i].tolist()}, y_hat = [{probs}], dự đoán = {predictions[i]}, thực tế = {int(y[i])}"
        )
    accuracy = np.mean(predictions == y)
    print(f"\nĐộ chính xác trên tập dữ liệu: {accuracy * 100:.2f}%")

else:
    eta_final = X.dot(beta)
    scores_final = thresholds - eta_final[:, None]
    s_final = sigmoid(scores_final)
    prob_final = np.zeros((len(y), num_classes))
    prob_final[:, 0] = s_final[:, 0]
    for k in range(1, num_classes - 1):
        prob_final[:, k] = s_final[:, k] - s_final[:, k - 1]
    prob_final[:, -1] = 1 - s_final[:, -1]
    predictions = np.argmax(prob_final, axis=1)
    for i in range(len(y)):
        probs = ", ".join([f"{p:.3f}" for p in prob_final[i]])
        print(
            f"Mẫu {i + 1}: X = {X_num[i].tolist()}, y_hat = [{probs}], dự đoán = {predictions[i]}, thực tế = {int(y[i])}"
        )
    accuracy = np.mean(predictions == y)
    print(f"\nĐộ chính xác trên tập dữ liệu: {accuracy * 100:.2f}%")

# Hiển thị biểu đồ
print("\n12) Vẽ biểu đồ Logistic Regression:")
plt.figure(figsize=(10, 6))
plt.scatter(X_num[:, 0], X_num[:, 1], c=y, cmap="bwr", edgecolors="k", s=120)
for i in range(len(X_num)):
    plt.text(X_num[i, 0] + 0.03, X_num[i, 1] + 0.03, df.loc[i, "Sinh viên"], fontsize=10)

# Tạo lưới để hiển thị xác suất dự đoán hoặc phân vùng lớp
xx1 = np.linspace(0, 2, 100)
xx2 = np.linspace(0, 1, 100)
xx1_grid, xx2_grid = np.meshgrid(xx1, xx2)
X_grid = np.vstack([np.ones(xx1_grid.ravel().shape), xx1_grid.ravel(), xx2_grid.ravel()]).T

if model_choice == "binary":
    prob_grid = sigmoid(X_grid.dot(weights)).reshape(xx1_grid.shape)
    contour = plt.contourf(xx1_grid, xx2_grid, prob_grid, levels=20, alpha=0.4, cmap="coolwarm")
    plt.colorbar(contour, label="Xác suất Đậu")

elif model_choice == "multinomial":
    logits = X_grid.dot(weights.T)
    prob_grid = softmax(logits)
    labels = np.argmax(prob_grid, axis=1).reshape(xx1_grid.shape)
    contour = plt.contourf(xx1_grid, xx2_grid, labels, levels=np.arange(num_classes + 1) - 0.5, alpha=0.4, cmap="tab10")
    cbar = plt.colorbar(contour, ticks=np.arange(num_classes))
    cbar.set_label("Lớp dự đoán")

else:
    eta_grid = X_grid.dot(beta)
    scores_grid = thresholds - eta_grid[:, None]
    s_grid = sigmoid(scores_grid)
    prob_grid = np.zeros((X_grid.shape[0], num_classes))
    prob_grid[:, 0] = s_grid[:, 0]
    for k in range(1, num_classes - 1):
        prob_grid[:, k] = s_grid[:, k] - s_grid[:, k - 1]
    prob_grid[:, -1] = 1 - s_grid[:, -1]
    labels = np.argmax(prob_grid, axis=1).reshape(xx1_grid.shape)
    contour = plt.contourf(xx1_grid, xx2_grid, labels, levels=np.arange(num_classes + 1) - 0.5, alpha=0.4, cmap="tab10")
    cbar = plt.colorbar(contour, ticks=np.arange(num_classes))
    cbar.set_label("Lớp Ordinal dự đoán")

if model_choice == "binary":
    for x2_value, label in [(0, "Không đầy đủ"), (1, "Đầy đủ")]:
        x1_line = np.linspace(0, 2, 200)
        X_line = np.vstack([np.ones_like(x1_line), x1_line, np.full_like(x1_line, x2_value)]).T
        y_line = sigmoid(X_line.dot(weights))
        plt.plot(x1_line, y_line, label=f"Sigmoid khi Làm bài tập={label}")
    plt.ylabel("Xác suất Đậu")
    plt.title(f"{model_name}: Xác suất dự đoán \nGiờ học và Làm bài tập", pad=20)
    plt.legend(title="Đường Sigmoid", loc="upper left", frameon=True)
elif model_choice == "multinomial":
    plt.ylabel("Lớp dự đoán")
    plt.title(f"{model_name}: Phân vùng lớp dự đoán \nGiờ học và Làm bài tập", pad=20)
else:
    plt.ylabel("Lớp Ordinal dự đoán")
    plt.title(f"{model_name}: Phân vùng Ordinal dự đoán \nGiờ học và Làm bài tập", pad=20)

plt.xlabel("Giờ học (0=Thấp, 1=Trung bình, 2=Cao)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

if interactive_display:
    plt.show()
else:
    output_path = f"logistic_regression_plot_{model_choice}.png"
    plt.savefig(output_path, dpi=150)
    print(f"Biểu đồ đã được lưu vào: {output_path}")
plt.close()