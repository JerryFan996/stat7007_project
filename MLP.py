import os
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

# Label mapping
label_map = {
    'NORMAL': 0,
    'PNEUMONIA': 1,
    'Cancer': 2
}
label_names = {v: k for k, v in label_map.items()}

# Image loading function
def load_images_from_folder(base_folder, label_map):
    images = []
    labels = []
    for class_name, label in label_map.items():
        class_folder = os.path.join(base_folder, class_name)
        
        if not os.path.exists(class_folder):
            print(f"no {class_folder}")
            continue

        for filename in os.listdir(class_folder):
            if filename.startswith('.'):
                continue
            img_path = os.path.join(class_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_flat = img.flatten()
                images.append(img_flat)
                labels.append(label)
            else:
                print(f"fail，skip：{img_path}")
    return np.array(images), np.array(labels)


train_folder = os.path.join("chest_xray_lung_low_res", "train")
val_folder = r"D:\stat7007_project\chest_xray_lung\val"  # ✅ 你指定的新验证集路径


if not os.path.exists(train_folder):
    raise FileNotFoundError(f"❌ Training set path does not exist: {train_folder}")
if not os.path.exists(val_folder):
    raise FileNotFoundError(f"❌ Valiation set path does not exist: {val_folder}")

# ----------------- main -----------------

print("Load training data...")
X_train, y_train = load_images_from_folder(train_folder, label_map)

print("Load validation data...")
X_val, y_val = load_images_from_folder(val_folder, label_map)

# standardisation
print(" Standardised data...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Training the MLP model
print(" Training the MLP model...")
mlp_model = MLPClassifier(hidden_layer_sizes=(512, 256), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)

# Validation set prediction
print("Validation set prediction...")
y_pred = mlp_model.predict(X_val)

# print
print("\n Classified reports:")
print(classification_report(y_val, y_pred, target_names=label_map.keys()))

print("\n confusion matrix:")
print(confusion_matrix(y_val, y_pred))

# save
dump(mlp_model, 'mlp_lung_model.pkl')
dump(scaler, 'scaler.pkl')

print("\n already save as 'mlp_lung_model.pkl' 和 'scaler.pkl'")
