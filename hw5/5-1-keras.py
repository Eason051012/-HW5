# 匯入必要的套件
import os
import shutil
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime

# 清除舊的日誌目錄
log_dir = "tb_logs"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)  # 刪除舊的日誌目錄
    print(f"已清除日誌目錄: {log_dir}")
os.makedirs(log_dir, exist_ok=True)  # 確保重建日誌目錄
print(f"已建立日誌目錄: {log_dir}")

# 加載 Iris 資料集
data = load_iris()
X, y = data.data, data.target

# 資料標準化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 分割訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 將目標轉換為 one-hot 編碼
y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

# 建立模型
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dropout(0.5),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')  # 輸出層
])

# 編譯模型
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 設置 TensorBoard 回調
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# 訓練模型
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    callbacks=[tensorboard_callback]
)

# 測試模型
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"測試損失: {test_loss:.4f}, 測試準確率: {test_acc:.4f}")

# 提示啟動 TensorBoard
print(f"啟動 TensorBoard，執行以下命令：")
print(f"tensorboard --logdir {log_dir}")