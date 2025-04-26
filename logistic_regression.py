import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

# Define directories
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
        images = os.listdir(class_dir)[:samples_per_class]  # Sample images
        for img_name in tqdm(images, desc=f"Loading {class_name}"):
            img_path = os.path.join(class_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    # Ensure all images are in RGB mode
                    img = img.convert("RGB")
                    img = img.resize(image_size)
                    data.append(np.array(img).flatten())  # Flatten the image
                    labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    return np.array(data), np.array(labels)

# Define class names
classes = ["normal", "cancer", "pneumonia"]

# Load training data
print("Loading training data...")
X_train, y_train = load_data(train_dir, classes, samples_per_class)

# Load testing data
print("Loading testing data...")
X_test, y_test = load_data(test_dir, classes, samples_per_class)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Train logistic regression model
print("Training logistic regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")