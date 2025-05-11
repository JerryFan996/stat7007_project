import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define directories (use absolute paths if needed)
train_dir = "chest_xray_lung_low_res/train"
test_dir = "chest_xray_lung_low_res/test"

# Define image size and number of samples per class
image_size = (224, 224)
samples_per_class = 1000

# Function to load images and labels
def load_data(directory, classes, samples_per_class):
    data = []
    labels = []
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        if not os.path.exists(class_dir):
            print(f"Directory not found: {class_dir}")
            continue
        images = os.listdir(class_dir)[:samples_per_class]
        for img_name in tqdm(images, desc=f"Loading {class_name}"):
            img_path = os.path.join(class_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img = img.resize(image_size)
                    data.append(np.array(img).flatten())
                    labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(data), np.array(labels)

# Define class names (match folder names exactly)
classes = ["NORMAL", "Cancer", "PNEUMONIA"]

# Load and normalize data
print("Loading training data...")
X_train, y_train = load_data(train_dir, classes, samples_per_class)
print("Loading testing data...")
X_test, y_test = load_data(test_dir, classes, samples_per_class)
X_train, X_test = X_train / 255.0, X_test / 255.0

# Convert to torch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# Define Logistic Regression model
class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

# Instantiate model
input_dim = X_train.shape[1]
num_classes = len(classes)
model = LogisticRegressionTorch(input_dim, num_classes).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
print("Training model...")
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    predictions = torch.argmax(model(X_test), dim=1)
    acc = (predictions == y_test).float().mean().item()
    print(f"Test Accuracy: {acc:.2f}")
