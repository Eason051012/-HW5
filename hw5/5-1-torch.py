# 匯入必要的套件
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# 動態生成日誌名稱
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"tb_logs_pytorch/{current_time}"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)  # 刪除舊的日誌目錄
os.makedirs(log_dir, exist_ok=True)  # 重建日誌目錄
print(f"已建立日誌目錄: {log_dir}")

# 加載 Iris 資料集
data = load_iris()
X, y = data.data, data.target

# 資料標準化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 分割訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 轉換為 TensorDataset
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.long)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.long)
)

# 建立 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定義模型
class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

# 初始化模型
model = IrisClassifier()

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 初始化 TensorBoard 記錄器
writer = SummaryWriter(log_dir)

# 訓練模型
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in train_loader:
        # 前向傳播
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # 反向傳播與優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 計算損失和準確率
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = correct / total
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    # 記錄到 TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)

# 測試模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

test_acc = correct / total
print(f"測試準確率: {test_acc:.4f}")
writer.add_scalar('Accuracy/test', test_acc)

# 關閉 TensorBoard 記錄器
writer.close()

# 提示啟動 TensorBoard
print(f"啟動 TensorBoard，執行以下命令：")
print(f"tensorboard --logdir {log_dir}")
