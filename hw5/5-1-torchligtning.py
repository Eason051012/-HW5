# 匯入必要的套件
import os
import shutil
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime

# 動態生成日誌名稱
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f"tb_logs_pytorch_lightning/{current_time}"
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
class IrisClassifier(LightningModule):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(4, 16)
        self.fc2 = torch.nn.Linear(16, 8)
        self.fc3 = torch.nn.Linear(8, 3)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        outputs = self(X_batch)
        loss = self.criterion(outputs, y_batch)
        acc = (outputs.argmax(dim=1) == y_batch).float().mean()
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        outputs = self(X_batch)
        loss = self.criterion(outputs, y_batch)
        acc = (outputs.argmax(dim=1) == y_batch).float().mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        outputs = self(X_batch)
        acc = (outputs.argmax(dim=1) == y_batch).float().mean()
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# 初始化模型
model = IrisClassifier()

# 初始化 TensorBoard 記錄器
logger = TensorBoardLogger("tb_logs_pytorch_lightning", name=current_time)

# 設置回調函數
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=log_dir,
    filename="best-checkpoint",
    save_top_k=1,
    mode="min"
)
early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

# 訓練模型
trainer = Trainer(
    max_epochs=20,
    logger=logger,
    callbacks=[checkpoint_callback, early_stopping_callback]
)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

# 測試模型
trainer.test(model, dataloaders=test_loader)
print(f"測試完成！日誌儲存在: {log_dir}")
