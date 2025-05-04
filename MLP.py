import os
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

# 标签映射（注意与文件夹一致）
label_map = {
    'NORMAL': 0,
    'PNEUMONIA': 1,
    'Cancer': 2
}
label_names = {v: k for k, v in label_map.items()}

# 图像加载函数（不再 resize！）
def load_images_from_folder(base_folder, label_map):
    images = []
    labels = []
    for class_name, label in label_map.items():
        class_folder = os.path.join(base_folder, class_name)
        
        if not os.path.exists(class_folder):
            print(f"⚠️ 类别文件夹不存在，跳过 {class_folder}")
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
                print(f"❌ 图片读取失败，跳过：{img_path}")
    return np.array(images), np.array(labels)

# 使用低分辨率训练集路径 + 原始验证集路径
train_folder = os.path.join("chest_xray_lung_low_res", "train")
val_folder = r"D:\stat7007_project\chest_xray_lung\val"  # ✅ 你指定的新验证集路径

# 路径检查
if not os.path.exists(train_folder):
    raise FileNotFoundError(f"❌ 训练集路径不存在: {train_folder}")
if not os.path.exists(val_folder):
    raise FileNotFoundError(f"❌ 验证集路径不存在: {val_folder}")

# ----------------- 主程序 -----------------

print("📥 加载训练数据...")
X_train, y_train = load_images_from_folder(train_folder, label_map)

print("📥 加载验证数据...")
X_val, y_val = load_images_from_folder(val_folder, label_map)

# 标准化
print("🔄 标准化数据...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 建立MLP模型
print("🧠 训练MLP模型...")
mlp_model = MLPClassifier(hidden_layer_sizes=(512, 256), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)

# 验证集预测
print("🔎 验证集预测...")
y_pred = mlp_model.predict(X_val)

# 打印评估结果
print("\n📊 分类报告:")
print(classification_report(y_val, y_pred, target_names=label_map.keys()))

print("\n📊 混淆矩阵:")
print(confusion_matrix(y_val, y_pred))

# 保存模型和标准化器
dump(mlp_model, 'mlp_lung_model.pkl')
dump(scaler, 'scaler.pkl')

print("\n✅ 模型和标准化器已保存为 'mlp_lung_model.pkl' 和 'scaler.pkl'")
