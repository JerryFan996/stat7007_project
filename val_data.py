import os
import random
import shutil

def move_images(train_dir, val_dir, split_ratio=0.2):
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    images = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
    num_to_move = int(len(images) * split_ratio)
    images_to_move = random.sample(images, num_to_move)

    for img in images_to_move:
        src_path = os.path.join(train_dir, img)
        dst_path = os.path.join(val_dir, img)
        shutil.move(src_path, dst_path)

    print(f"Moved {num_to_move} images from {train_dir} to {val_dir}.")


train_dir = 'chest_xray_lung/train/PNEUMONIA'
val_dir = 'chest_xray_lung/val/PNEUMONIA'
move_images(train_dir, val_dir, split_ratio=0.2)
