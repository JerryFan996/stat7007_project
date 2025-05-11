import os
import numpy as np
from PIL import Image
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# set random seed for reproducibility
def load_data(folder, image_size=(224, 224)):
    X, y = [], []
    for label in os.listdir(folder):
        class_path = os.path.join(folder, label)
        if not os.path.isdir(class_path):
            continue
        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)
            try:
                img = Image.open(img_path).convert('L')
                img = img.resize(image_size)
                img_array = np.array(img).flatten()
                X.append(img_array)
                y.append(label)
            except:
                continue
    return np.array(X), np.array(y)


train_dir = 'chest_xray_lung_low_res/train'
val_dir = 'chest_xray_lung_low_res/val'

X_train, y_train = load_data(train_dir)
X_val, y_val = load_data(val_dir)

# Data Preprocessing
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)

clf = GaussianNB()
clf.fit(X_train, y_train_enc)

y_pred = clf.predict(X_val)
print("NB Classification:\n")
print(classification_report(y_val_enc, y_pred, target_names=le.classes_))
