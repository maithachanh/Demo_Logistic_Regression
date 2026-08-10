import sys
import io
import argparse
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

# ==========================================
# 1. Các hàm tiện ích & Hiển thị dữ liệu
# ==========================================

def print_boxed_df(df, title):
    """In DataFrame dưới dạng khung bảng ASCII đẹp mắt."""
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

def sigmoid(z):
    """Hàm Sigmoid: 1 / (1 + exp(-z)). Clip để tránh bùng nổ số học."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def softmax(z):
    """Hàm Softmax tính xác suất đa lớp ổn định số học (subtract max z)."""
    z_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return z_exp / np.sum(z_exp, axis=1, keepdims=True)

# ==========================================
# 2. Cấu trúc Mô hình Logistic Regression (OOP)
# ==========================================

class BaseLogisticRegression:
    """Lớp cơ sở cho các mô hình Logistic Regression."""
    def __init__(self, learning_rate=0.1, epochs=50, seed=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed
        self.loss_history = []

    def fit(self, X, y, verbose=True):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        probs = self.predict_proba(X)
        if probs.ndim == 1:
            return (probs >= 0.5).astype(int)
        return np.argmax(probs, axis=1)

    def evaluate(self, X, y):
        preds = self.predict(X)
        acc = np.mean(preds == y)
        return acc, preds

class BinaryLogisticRegression(BaseLogisticRegression):
    """
    Binary Logistic Regression:
    Dự đoán xác suất nhãn nhị phân y in {0, 1} bằng hàm Sigmoid.
    Tối ưu hóa bằng Gradient Descent trên Binary Cross-Entropy Loss.
    """
    def __init__(self, learning_rate=0.1, epochs=50, seed=42):
        super().__init__(learning_rate, epochs, seed)
        self.weights = None

    def fit(self, X, y, verbose=True):
        np.random.seed(self.seed)
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features) * 0.1
        self.loss_history = []

        if verbose:
            print("\n4) Khởi tạo trọng số (weights) ban đầu [bias, w1, w2]:")
            print(f"  w0 = {self.weights.tolist()}")

        print(f"\n5) Huấn luyện Binary Logistic Regression ({self.epochs} Epochs, lr={self.learning_rate}):")
        epsilon = 1e-9
        for epoch in range(1, self.epochs + 1):
            linear_combination = X.dot(self.weights)
            y_pred = sigmoid(linear_combination)
            
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(y_pred_clipped) + (1 - y) * np.log(1 - y_pred_clipped))
            self.loss_history.append(loss)

            gradient = X.T.dot(y_pred - y) / n_samples
            self.weights -= self.learning_rate * gradient

            if verbose:
                print(f"\n--- Epoch {epoch} ---")
                print("Linear combination z = X · w:")
                for i, z in enumerate(linear_combination):
                    print(f"  Mẫu {i + 1}: z = {X[i].tolist()} · {self.weights.tolist()} = {z:.6f}")
                print("Sigmoid y_hat = 1 / (1 + exp(-z)):")
                for i, p in enumerate(y_pred):
                    print(f"  Mẫu {i + 1}: y_hat = {p:.6f}")
                print(f"Loss (Binary Cross Entropy): {loss:.6f}")
                print(f"Gradient: {gradient.tolist()}")
                print(f"Weights sau cập nhật: {self.weights.tolist()}")

    def predict_proba(self, X):
        return sigmoid(X.dot(self.weights))

class MultinomialLogisticRegression(BaseLogisticRegression):
    """
    Multinomial Logistic Regression (Softmax Regression):
    Dự đoán phân bố xác suất cho K lớp (K >= 3).
    Tối ưu hóa bằng Gradient Descent trên Categorical Cross-Entropy Loss.
    """
    def __init__(self, num_classes=3, learning_rate=0.1, epochs=50, seed=42):
        super().__init__(learning_rate, epochs, seed)
        self.num_classes = num_classes
        self.weights = None

    def fit(self, X, y, verbose=True):
        np.random.seed(self.seed)
        n_samples, n_features = X.shape
        self.weights = np.random.randn(self.num_classes, n_features) * 0.1
        self.loss_history = []

        if verbose:
            print("\n4) Khởi tạo ma trận trọng số W ban đầu (num_classes x n_features):")
            print(self.weights)

        print(f"\n5) Huấn luyện Multinomial Logistic Regression ({self.epochs} Epochs, lr={self.learning_rate}):")
        epsilon = 1e-9
        y_onehot = np.eye(self.num_classes)[y]

        for epoch in range(1, self.epochs + 1):
            linear_combination = X.dot(self.weights.T)
            y_pred = softmax(linear_combination)
            
            y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
            loss = -np.mean(np.sum(y_onehot * np.log(y_pred_clipped), axis=1))
            self.loss_history.append(loss)

            gradient = (y_pred - y_onehot).T.dot(X) / n_samples
            self.weights -= self.learning_rate * gradient

            if verbose:
                print(f"\n--- Epoch {epoch} ---")
                print("Linear combination Z = X · W^T:")
                print(linear_combination)
                print("Softmax y_hat:")
                print(y_pred)
                print(f"Loss (Categorical Cross Entropy): {loss:.6f}")
                print("Gradient ma trận W:")
                print(gradient)
                print("Weights sau cập nhật:")
                print(self.weights)

    def predict_proba(self, X):
        return softmax(X.dot(self.weights.T))

class OrdinalLogisticRegression(BaseLogisticRegression):
    """
    Ordinal Logistic Regression (Cumulative Logit / Proportional Odds Model):
    Xử lý các nhãn có thứ tự (Ordinal: Trượt < Đậu < Xuất sắc).
    
    Giải quyết vấn đề đa cộng tuyến bằng cách KHÔNG đưa cột bias vào ma trận X
    (dùng X_raw không bias), vì các điểm ngưỡng (thresholds / cutpoints theta_k)
    đã đóng vai trò làm hệ số chặn phân tách ranh giới giữa các lớp thứ tự.
    """
    def __init__(self, num_classes=3, learning_rate=0.1, epochs=50, seed=42):
        super().__init__(learning_rate, epochs, seed)
        self.num_classes = num_classes
        self.beta = None
        self.thresholds = None

    def fit(self, X_raw, y, verbose=True):
        np.random.seed(self.seed)
        n_samples, n_features = X_raw.shape
        K = self.num_classes

        # Khởi tạo beta cho đặc trưng thực và thresholds cho K-1 ranh giới
        self.beta = np.random.randn(n_features) * 0.1
        self.thresholds = np.linspace(-0.5, 0.5, K - 1)
        self.loss_history = []

        if verbose:
            print("\n4) Khởi tạo beta (không chứa bias dư thừa) và thresholds ban đầu:")
            print("  beta =", self.beta)
            print("  thresholds =", self.thresholds)

        print(f"\n5) Huấn luyện Ordinal Logistic Regression ({self.epochs} Epochs, lr={self.learning_rate}):")
        epsilon = 1e-9

        for epoch in range(1, self.epochs + 1):
            eta = X_raw.dot(self.beta)
            scores = self.thresholds - eta[:, None]  # Ma trận (n_samples x K-1)
            s = sigmoid(scores)
            s_clipped = np.clip(s, epsilon, 1 - epsilon)

            # y_cum[i, k] = 1 nếu y_i <= k, ngược lại = 0
            y_cum = (y[:, None] <= np.arange(K - 1)).astype(float)
            
            # Loss Proportional Odds (Binary Cumulative Loss)
            loss = -np.mean(np.sum(y_cum * np.log(s_clipped) + (1 - y_cum) * np.log(1 - s_clipped), axis=1))
            self.loss_history.append(loss)

            # Gradients
            diff = s - y_cum  # (n_samples x K-1)
            grad_beta = -X_raw.T.dot(np.sum(diff, axis=1)) / n_samples
            grad_thresholds = np.mean(diff, axis=0)

            # Cập nhật tham số
            self.beta -= self.learning_rate * grad_beta
            self.thresholds -= self.learning_rate * grad_thresholds

            # Đảm bảo tính đơn điệu của các điểm ngưỡng (monotonic cutpoints)
            for k in range(len(self.thresholds) - 1):
                if self.thresholds[k] >= self.thresholds[k + 1]:
                    self.thresholds[k + 1] = self.thresholds[k] + 1e-3

            if verbose:
                print(f"\n--- Epoch {epoch} ---")
                print(f"Eta = X_raw · beta: {eta}")
                print(f"Thresholds hiện tại: {self.thresholds}")
                print(f"Xác suất tích lũy s = sigmoid(thresholds - eta):")
                print(s)
                print(f"Loss (Ordinal Cumulative Loss): {loss:.6f}")
                print(f"Gradient beta: {grad_beta}")
                print(f"Gradient thresholds: {grad_thresholds}")

    def predict_proba(self, X_raw):
        eta = X_raw.dot(self.beta)
        scores = self.thresholds - eta[:, None]
        s = sigmoid(scores)
        n_samples = len(X_raw)
        K = self.num_classes

        prob = np.zeros((n_samples, K))
        prob[:, 0] = s[:, 0]
        for k in range(1, K - 1):
            prob[:, k] = np.maximum(0, s[:, k] - s[:, k - 1])
        prob[:, -1] = np.maximum(0, 1.0 - s[:, -1])

        # Chuẩn hóa lại tổng xác suất mỗi hàng = 1
        row_sums = np.sum(prob, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return prob / row_sums

# ==========================================
# 3. Hàm chính & Xử lý chạy chương trình
# ==========================================

def run_demo(model_choice="binary", epochs=50, learning_rate=0.1, seed=42, show_plot=True, verbose=True):
    model_name = {
        "binary": "Binary Logistic Regression",
        "multinomial": "Multinomial Logistic Regression",
        "ordinal": "Ordinal Logistic Regression",
    }.get(model_choice, "Binary Logistic Regression")

    print(f"\n=======================================================")
    print(f"   DEMO MACHINE LEARNING: {model_name.upper()}")
    print(f"=======================================================")

    # 1. Khởi tạo dữ liệu mẫu
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
        result_label_text = "Kết quả: Trượt=0, Đậu=1, Xuất sắc=2"

    df = pd.DataFrame(raw_data)
    print()
    print_boxed_df(df, "1) Dữ liệu ban đầu")

    # 2. Mã hóa dữ liệu (Encoding)
    hour_mapping = {"Thấp": 0, "Trung bình": 1, "Cao": 2}
    homework_mapping = {"Không đầy đủ": 0, "Đầy đủ": 1}

    print("\n2) Chuyển dữ liệu sang dạng số (Encoding):")
    print("  Giờ học: Thấp=0, Trung bình=1, Cao=2")
    print("  Làm bài tập: Không đầy đủ=0, Đầy đủ=1")
    print(f"  {result_label_text}")

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
    print_boxed_df(encoded_df, "2) Dữ liệu sau khi mã hóa số (Encoding)")

    # 3. Tạo ma trận đặc trưng X
    print("\n3) Ma trận đặc trưng X và nhãn y:")
    print("  X_num = [[Giờ học, Làm bài tập], ...]")
    print(X_num)
    print("  y = [Kết quả ...]")
    print(y)

    X_bias = np.hstack([np.ones((X_num.shape[0], 1)), X_num])

    # 4 & 5. Huấn luyện mô hình
    if model_choice == "binary":
        model = BinaryLogisticRegression(learning_rate=learning_rate, epochs=epochs, seed=seed)
        model.fit(X_bias, y, verbose=verbose)
        X_eval = X_bias
    elif model_choice == "multinomial":
        model = MultinomialLogisticRegression(num_classes=len(result_mapping), learning_rate=learning_rate, epochs=epochs, seed=seed)
        model.fit(X_bias, y, verbose=verbose)
        X_eval = X_bias
    else:
        # Ordinal model dùng X_num không chứa cột bias dư thừa
        model = OrdinalLogisticRegression(num_classes=len(result_mapping), learning_rate=learning_rate, epochs=epochs, seed=seed)
        model.fit(X_num, y, verbose=verbose)
        X_eval = X_num

    # 6. Đánh giá & Dự đoán
    acc, predictions = model.evaluate(X_eval, y)
    probs = model.predict_proba(X_eval)

    print("\n11) Dự đoán với trọng số đã học:")
    for i in range(len(y)):
        if model_choice == "binary":
            prob_percent = round(probs[i] * 100, 2)
            z_val = X_eval[i].dot(model.weights)
            print(f"  Mẫu {i + 1}: X = {X_num[i].tolist()}, z = {z_val:.6f}, "
                  f"y_hat = {probs[i]:.6f} ({prob_percent}%), dự đoán = {predictions[i]}, thực tế = {y[i]}")
        else:
            prob_str = ", ".join([f"{p:.3f}" for p in probs[i]])
            print(f"  Mẫu {i + 1}: X = {X_num[i].tolist()}, y_hat = [{prob_str}], "
                  f"dự đoán = {predictions[i]}, thực tế = {y[i]}")

    print(f"\n>>> Độ chính xác trên tập dữ liệu: {acc * 100:.2f}%")

    # 7. Trực quan hóa Biểu đồ (Dual Subplots)
    print("\n12) Vẽ biểu đồ trực quan hóa (Decision Boundaries & Loss History):")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Subplot 1: Ranh giới quyết định & Phân vùng lớp ---
    ax1.scatter(X_num[:, 0], X_num[:, 1], c=y, cmap="bwr" if model_choice == "binary" else "tab10",
                edgecolors="k", s=140, zorder=5)
    for i in range(len(X_num)):
        ax1.text(X_num[i, 0] + 0.04, X_num[i, 1] + 0.04, df.loc[i, "Sinh viên"], fontsize=11, fontweight='bold')

    xx1 = np.linspace(-0.2, 2.2, 200)
    xx2 = np.linspace(-0.2, 1.2, 200)
    xx1_grid, xx2_grid = np.meshgrid(xx1, xx2)

    if model_choice == "binary":
        X_grid_bias = np.vstack([np.ones(xx1_grid.ravel().shape), xx1_grid.ravel(), xx2_grid.ravel()]).T
        prob_grid = model.predict_proba(X_grid_bias).reshape(xx1_grid.shape)
        contour = ax1.contourf(xx1_grid, xx2_grid, prob_grid, levels=20, alpha=0.4, cmap="coolwarm")
        fig.colorbar(contour, ax=ax1, label="Xác suất Đậu")

        # Vẽ đường ranh giới quyết định Decision Boundary line (z = 0 -> w0 + w1*x1 + w2*x2 = 0)
        w0, w1, w2 = model.weights
        if abs(w2) > 1e-6:
            x2_boundary = -(w0 + w1 * xx1) / w2
            valid = (x2_boundary >= -0.2) & (x2_boundary <= 1.2)
            ax1.plot(xx1[valid], x2_boundary[valid], "r--", linewidth=2.5, label="Ranh giới quyết định (P=0.5)")
            ax1.legend(loc="upper left")

        ax1.set_ylabel("Làm bài tập (0=Không đầy đủ, 1=Đầy đủ)")
        ax1.set_title(f"{model_name}\nPhân vùng xác suất dự đoán", fontsize=12)

    elif model_choice == "multinomial":
        X_grid_bias = np.vstack([np.ones(xx1_grid.ravel().shape), xx1_grid.ravel(), xx2_grid.ravel()]).T
        labels_grid = model.predict(X_grid_bias).reshape(xx1_grid.shape)
        num_classes = len(result_mapping)
        contour = ax1.contourf(xx1_grid, xx2_grid, labels_grid, levels=np.arange(num_classes + 1) - 0.5,
                               alpha=0.4, cmap="tab10")
        cbar = fig.colorbar(contour, ax=ax1, ticks=np.arange(num_classes))
        cbar.set_label("Lớp dự đoán (0=Trượt, 1=Đậu, 2=Xuất sắc)")
        ax1.set_ylabel("Làm bài tập (0=Không đầy đủ, 1=Đầy đủ)")
        ax1.set_title(f"{model_name}\nPhân vùng phân loại Đa lớp", fontsize=12)

    else:
        X_grid_raw = np.vstack([xx1_grid.ravel(), xx2_grid.ravel()]).T
        labels_grid = model.predict(X_grid_raw).reshape(xx1_grid.shape)
        num_classes = len(result_mapping)
        contour = ax1.contourf(xx1_grid, xx2_grid, labels_grid, levels=np.arange(num_classes + 1) - 0.5,
                               alpha=0.4, cmap="tab10")
        cbar = fig.colorbar(contour, ax=ax1, ticks=np.arange(num_classes))
        cbar.set_label("Lớp dự đoán (0=Trượt, 1=Đậu, 2=Xuất sắc)")
        ax1.set_ylabel("Làm bài tập (0=Không đầy đủ, 1=Đầy đủ)")
        ax1.set_title(f"{model_name}\nPhân vùng phân loại Thứ tự (Ordinal)", fontsize=12)

    ax1.set_xlabel("Giờ học (0=Thấp, 1=Trung bình, 2=Cao)")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- Subplot 2: Lịch sử hàm mất mát (Loss Curve) ---
    ax2.plot(range(1, len(model.loss_history) + 1), model.loss_history, "b-o", linewidth=2, markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mất mát (Loss)")
    ax2.set_title(f"Lịch sử hội tụ hàm Loss qua {epochs} Epochs", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    min_loss = min(model.loss_history)
    min_epoch = model.loss_history.index(min_loss) + 1
    ax2.annotate(f"Min Loss: {min_loss:.4f}", xy=(min_epoch, min_loss),
                 xytext=(min_epoch + epochs * 0.1, min_loss + 0.05),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6),
                 fontsize=10, fontweight='bold', color='darkred')

    plt.tight_layout()

    output_path = f"logistic_regression_plot_{model_choice}.png"
    plt.savefig(output_path, dpi=150)
    print(f"  -> Biểu đồ đã được lưu vào file: {output_path}")

    if show_plot and interactive_display:
        plt.show()
    plt.close()

# ==========================================
# 4. Entrypoint Dòng lệnh (CLI)
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Demo giải thuật Logistic Regression trong Machine Learning")
    parser.add_argument("--model", type=str, choices=["binary", "multinomial", "ordinal", "all"], default=None,
                        help="Kiểu mô hình Logistic Regression (binary, multinomial, ordinal, all)")
    parser.add_argument("--epochs", type=int, default=50, help="Số lượng epoch huấn luyện (Mặc định: 50)")
    parser.add_argument("--lr", type=float, default=0.1, help="Tốc độ học (Learning Rate, Mặc định: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed ngẫu nhiên (Mặc định: 42)")
    parser.add_argument("--no-plot", action="store_true", help="Không hiển thị hoặc vẽ biểu đồ")
    parser.add_argument("--quiet", action="store_true", help="Ẩn các log epoch chi tiết")

    args = parser.parse_args()

    model_choice = args.model
    if model_choice is None:
        try:
            user_input = input("Chọn kiểu Logistic Regression (binary/multinomial/ordinal/all) [binary]: ").strip().lower()
            if user_input in {"binary", "multinomial", "ordinal", "all"}:
                model_choice = user_input
            else:
                model_choice = "binary"
        except Exception:
            model_choice = "binary"

    show_plot = not args.no_plot
    verbose = not args.quiet

    if model_choice == "all":
        for mode in ["binary", "multinomial", "ordinal"]:
            run_demo(model_choice=mode, epochs=args.epochs, learning_rate=args.lr, seed=args.seed,
                     show_plot=show_plot, verbose=verbose)
    else:
        run_demo(model_choice=model_choice, epochs=args.epochs, learning_rate=args.lr, seed=args.seed,
                 show_plot=show_plot, verbose=verbose)

if __name__ == "__main__":
    main()