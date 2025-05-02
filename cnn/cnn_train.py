#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cnn_def import SimpleCNN
from collections import Counter

def main():
    # —— 确定运行设备 —— #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print("GPU name:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())
        torch.backends.cudnn.benchmark = True  # 提升性能（仅固定输入尺寸时）

    # —— 路径设置 —— #
    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_ROOT    = os.path.join(PROJECT_ROOT, 'chest_xray_lung')
    TRAIN_DIR    = os.path.join(DATA_ROOT, 'train')
    VAL_DIR      = os.path.join(DATA_ROOT, 'val')

    # —— 数据加载 —— #
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)

    # 类别信息
    print("Classes:", train_ds.classes)
    print("Class→Index:", train_ds.class_to_idx)
    print("Train size:", len(train_ds), "Val size:", len(val_ds))
    print("Train class dist:", {train_ds.classes[k]: v for k,v in Counter([label for _, label in train_ds.imgs]).items()})
    print("Val class dist:  ", {val_ds.classes[k]: v for k,v in Counter([label for _, label in val_ds.imgs]).items()})

    # —— 模型与优化器 —— #
    model     = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # —— 训练循环 —— #
    num_epochs = 5
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch}] Train loss: {total_loss/len(train_loader):.4f}")

        # —— 验证 —— #
        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
        acc = correct / len(val_ds)
        print(f"[Epoch {epoch}] Val loss: {val_loss/len(val_loader):.4f}, Val acc: {acc:.3f}")

if __name__ == '__main__':
    print("hello world")
    main()
