#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cnn_def import SimpleCNN
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print("GPU name:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
        print("cuDNN version:", torch.backends.cudnn.version())
        torch.backends.cudnn.benchmark = True

    # —— 路径设置 —— #
    SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    DATA_ROOT    = os.path.join(PROJECT_ROOT, 'chest_xray_lung')
    TRAIN_DIR    = os.path.join(DATA_ROOT, 'train')
    VAL_DIR      = os.path.join(DATA_ROOT, 'val')

    # —— 数据增强 —— #
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_ds   = datasets.ImageFolder(VAL_DIR,   transform=transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=4)

    print("Classes:", train_ds.classes)
    print("Train size:", len(train_ds), "Val size:", len(val_ds))

    # —— 类别权重处理 —— #
    class_weights = compute_class_weight('balanced', classes=np.unique(train_ds.targets), y=train_ds.targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # —— 模型与训练组件 —— #
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    # —— 训练 —— #
    num_epochs = 10
    for epoch in range(1, num_epochs+1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        train_acc = correct / total
        print(f"[Epoch {epoch}] Train loss: {total_loss/len(train_loader):.4f}, Train acc: {train_acc:.3f}")

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
        val_acc = correct / len(val_ds)
        print(f"[Epoch {epoch}] Val loss: {val_loss/len(val_loader):.4f}, Val acc: {val_acc:.3f}")

        scheduler.step()

if __name__ == '__main__':
    print("hello world")
    main()
