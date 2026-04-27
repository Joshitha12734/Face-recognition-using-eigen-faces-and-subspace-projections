from PIL import Image
import numpy as np
import os

def load_yale_dataset(path):
    images = []
    labels = []

    for file in os.listdir(path):
        img_path = os.path.join(path, file)

        # Skip folders
        if not os.path.isfile(img_path):
            continue

        if not file.lower().endswith(".gif"):
            continue

        
        if not file.startswith("subject"):
            continue

        label = file.split(".")[0]  # subject01

        try:
            img = Image.open(img_path).convert('L')
        except:
            print("Skipping file:", file)
            continue

        img = np.array(img)


        if np.std(img) != 0:
            img = (img - np.mean(img)) / np.std(img)

        images.append(img.flatten())
        labels.append(label)

    return np.array(images), np.array(labels)


if __name__ == "__main__":

    path = r"C:\Users\MURALI KRISHNA\Downloads\archive"

    X, y = load_yale_dataset(path)

    print("Data shape:", X.shape)
    print("Unique people:", len(set(y)))
