import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

DATA_FOLDER = "data/handwritten-english-characters-and-digits/combined_folder/train/"

class_to_data_map = {
    "NUMERIC": list('0123456789'),
    "ALPHANUMERIC": None
}

def load_images_from_folder(folder: str, img_size=(50, 50), select_class: str = 'NUMERIC'):
    images = []
    labels = []

    if select_class not in class_to_data_map:
        raise ValueError(f"Invalid data class '{select_class}'. Valid options are: {list(class_to_data_map.keys())}")

    class_directories = os.listdir(folder)

    if select_class == "NUMERIC":
        valid_labels = class_to_data_map["NUMERIC"]
        filtered_class_directories = [d for d in class_directories if d in valid_labels]
    else:
        filtered_class_directories = class_directories

    class_label_map = {class_name: i for i, class_name in enumerate(filtered_class_directories)}

    for class_label in filtered_class_directories:
        class_folder = os.path.join(folder, class_label)
        entries = os.listdir(class_folder)

        # DEBUG
        print(f'{class_folder}: {len(entries)} images found')

        for filename in entries:
            if filename.endswith('.png'):
                img_path = os.path.join(class_folder, filename)
                img = Image.open(img_path).convert('L')
                img = img.resize(img_size)
                img = np.array(img) / 255.0
                images.append(img)
                labels.append(class_label)

    return np.array(images), np.array(labels)

def one_hot(labels, classes, dtype=np.float32):
    label_to_index = {label: idx for idx, label in enumerate(classes)}
    indices = np.array([label_to_index[label] for label in labels])
    return np.eye(len(classes), dtype=dtype)[indices]

def get_dataset(train_ratio=0.8, img_size=(50, 50), select_class="NUMERIC"):
    x, y = load_images_from_folder(folder=DATA_FOLDER, img_size=img_size, select_class=select_class)
    x = x.reshape(x.shape[0], -1)

    unique_classes = list(np.unique(y))
    y_new = one_hot(y, unique_classes)

    total_size = x.shape[0]
    train_size = int(train_ratio * total_size)

    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y_new[:train_size], y_new[train_size:]

    shuffle_index = np.random.permutation(train_size)

    x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

    return x_train, x_test, y_train, y_test

def show_images(images, num_row=2, num_col=5):
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col, 2*num_row))
    for i in range(num_row * num_col):
        ax = axes[i // num_col, i % num_col]
        ax.imshow(images[i], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
