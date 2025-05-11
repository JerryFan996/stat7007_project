import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import ResNet34_Weights
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# path to the dataset
data_dir = 'chest_xray_lung'
batch_size = 256
num_epochs = 20
learning_rate = 0.00005

# data augmentation and normalization for training
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(translate=(0.05, 0.05), degrees=0),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}


def load_data():
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val', 'test']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
        for x in ['train', 'val', 'test']
    }

    return image_datasets, dataloaders


def compute_class_weights(dataset):
    labels = [label for _, label in dataset]
    labels_tensor = torch.tensor(labels)
    class_counts = torch.bincount(labels_tensor)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_counts)  # 归一化，保持数量感知

    print("\n📊 Class counts:", class_counts.tolist())
    print("⚖️  Class weights:", class_weights.tolist())
    return class_weights.to(device)


def train_model(model, dataloaders, dataset_sizes, class_weights):
    lrs = []
    grads = []

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        current_lr = optimizer.param_groups[0]['lr']
                        lrs.append(current_lr)

                        total_norm = 0
                        for p in model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        grads.append(total_norm)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    return lrs, grads


def evaluate_model(model, dataloader, class_names):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())

    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # visualize confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def replace_relu_with(model, new_activation):
    for name, module in model.named_children():
        if isinstance(module, nn.ReLU):
            setattr(model, name, new_activation)
        else:
            replace_relu_with(module, new_activation)

def plot_lr_and_gradients(lrs, grads):
    print("\n📈 Plotting Learning Rate and Gradient Norms...")
    steps = np.arange(len(lrs))
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Learning Rate', color='tab:blue')
    ax1.plot(steps, lrs, color='tab:blue', label='Learning Rate')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Gradient Norm', color='tab:red')
    ax2.plot(steps, grads, color='tab:red', label='Gradient Norm')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Learning Rate and Gradient Norm During Training')
    fig.tight_layout()
    plt.show()


def main():
    image_datasets, dataloaders = load_data()
    class_names = image_datasets['train'].classes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # compute class weights
    class_weights = compute_class_weights(image_datasets['train'])

    model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3)

    replace_relu_with(model, nn.LeakyReLU(inplace=True))

    model = model.to(device)

    lrs, grads = train_model(model, dataloaders, dataset_sizes, class_weights)

    evaluate_model(model, dataloaders['test'], class_names)

    plot_lr_and_gradients(lrs, grads)

if __name__ == "__main__":
    main()
