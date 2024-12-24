import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.datasets import mnist

# 加載 MNIST 資料集
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 資料標準化
X_train = X_train / 255.0  # 將像素值縮放到 [0, 1] 範圍
X_test = X_test / 255.0

# 動態生成 TensorBoard 日誌目錄
log_dir = os.path.join("tb_logs_dense_nn", datetime.now().strftime("%Y%m%d-%H%M%S"))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
print(f"TensorBoard 日誌目錄: {log_dir}")

# 定義 Dense NN 模型
model = Sequential([
    Flatten(input_shape=(28, 28)),  # 展平成 784 維向量
    Dense(256, activation='relu'),  # 第一個全連接層
    Dense(128, activation='relu'),  # 第二個全連接層
    Dense(10, activation='softmax')  # 輸出層 (10 個類別)
])

# 編譯模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 初始化 TensorBoard 回調
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# 訓練模型
model.fit(
    X_train, y_train,
    epochs=10,  # 訓練 10 個 Epoch
    batch_size=32,  # 每次使用 32 筆資料進行訓練
    validation_split=0.2,  # 使用 20% 的資料作為驗證集
    callbacks=[tensorboard_callback]  # 加入 TensorBoard 回調
)

# 評估模型
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"測試損失: {test_loss:.4f}, 測試準確率: {test_acc:.4f}")

# 提示啟動 TensorBoard
print(f"啟動 TensorBoard，執行以下命令：")
print(f"tensorboard --logdir {log_dir}")
