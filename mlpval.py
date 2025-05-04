import os
import cv2
import numpy as np
from joblib import load
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ============ 标签映射 ============
label_map = {
    'NORMAL': 0,
    'PNEUMONIA': 1,
    'Cancer': 2
}
label_names = {v: k for k, v in label_map.items()}

# ============ 图像加载函数（含resize） ============
def load_images_from_folder(base_folder, label_map, image_size=(224, 224)):
    images = []
    labels = []
    for class_name, label in label_map.items():
        class_folder = os.path.join(base_folder, class_name)
        if not os.path.exists(class_folder):
            print(f"⚠️ 类别文件夹不存在，跳过 {class_folder}")
            continue
        for filename in os.listdir(class_folder):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(class_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
                images.append(img.flatten())
                labels.append(label)
            else:
                print(f"❌ 无法读取图像：{img_path}")
    return np.array(images), np.array(labels)

# ============ 设置验证集路径 ============
val_folder = r"D:\stat7007_project\chest_xray_lung\val"

# ============ 加载模型与标准化器 ============
print("📦 加载模型和标准化器...")
model = load("mlp_lung_model.pkl")
scaler = load("scaler.pkl")

# ============ 加载验证集 ============
print("📥 加载验证数据...")
X_val, y_val = load_images_from_folder(val_folder, label_map)

# ============ 标准化 ============
print("🔄 标准化验证集...")
X_val_scaled = scaler.transform(X_val)

# ============ 预测 ============
print("🔍 进行预测...")
y_pred = model.predict(X_val_scaled)

# ============ 输出结果 ============
print("\n📊 分类报告:")
report = classification_report(y_val, y_pred, target_names=label_map.keys())
print(report)

print("\n📊 混淆矩阵:")
cm = confusion_matrix(y_val, y_pred)
print(cm)

# ============ 保存报告 ============
with open("val_classification_report.txt", "w", encoding="utf-8") as f:
    f.write("📊 分类报告:\n")
    f.write(report + "\n\n")
    f.write("📊 混淆矩阵:\n")
    f.write(np.array2string(cm))

print("\n✅ 报告已保存为 val_classification_report.txt")
