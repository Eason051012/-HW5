import os
from datetime import datetime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.models import vgg16, vgg19
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

def main():
    # 動態生成 TensorBoard 日誌目錄
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"tb_logs_cifar/{current_time}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"TensorBoard 日誌目錄: {log_dir}")

    # 定義模型
    class VGGClassifier(pl.LightningModule):
        def __init__(self, use_vgg16=True, num_classes=10, lr=0.001):
            super(VGGClassifier, self).__init__()
            self.save_hyperparameters()  # 保存超參數
            self.lr = lr

            # 選擇 VGG16 或 VGG19 預訓練模型
            self.feature_extractor = vgg16(pretrained=True) if use_vgg16 else vgg19(pretrained=True)
            for param in self.feature_extractor.parameters():
                param.requires_grad = False  # 凍結預訓練權重

            # 修改分類器部分
            in_features = self.feature_extractor.classifier[0].in_features
            self.feature_extractor.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )

            self.criterion = nn.CrossEntropyLoss()

        def forward(self, x):
            return self.feature_extractor(x)

        def training_step(self, batch, batch_idx):
            images, labels = batch
            outputs = self(images)
            loss = self.criterion(outputs, labels)
            acc = (outputs.argmax(dim=1) == labels).float().mean()
            self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            images, labels = batch
            outputs = self(images)
            loss = self.criterion(outputs, labels)
            acc = (outputs.argmax(dim=1) == labels).float().mean()
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            return loss

        def test_step(self, batch, batch_idx):
            images, labels = batch
            outputs = self(images)
            acc = (outputs.argmax(dim=1) == labels).float().mean()
            self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            return acc

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.lr)

    # 資料預處理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 加載 CIFAR-10 資料集
    train_dataset = CIFAR10(root="data", train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root="data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 初始化模型
    model = VGGClassifier(use_vgg16=True)  # 設為 False 使用 VGG19

    # 初始化 TensorBoard 記錄器
    logger = TensorBoardLogger("tb_logs_cifar", name=current_time)

    # 設置回調函數
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=log_dir,
        filename="best-checkpoint",
        save_top_k=1,
        mode="min"
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    # 訓練器
    trainer = pl.Trainer(
        max_epochs=10,
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1
    )

    # 訓練模型
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 測試模型
    trainer.test(model, dataloaders=val_loader)
    print(f"測試完成！日誌儲存在: {log_dir}")

# 確保程式在主模組中執行
if __name__ == '__main__':
    main()
