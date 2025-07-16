import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

DATASET_PATH = 'Dataset/UTKFace/crop_part1'
IMAGE_SIZE = 200

def load_data(limit=10000):
    images = []
    ages = []
    genders = []

    count = 0
    for filename in os.listdir(DATASET_PATH):
        if not filename.endswith('.jpg'):
            continue

        try:
            parts = filename.split('_')
            age = int(parts[0])
            gender = int(parts[1])
        except:
            continue

        img_path = os.path.join(DATASET_PATH, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img / 255.0

        images.append(img)
        ages.append(age)
        genders.append(gender)

        count += 1
        if count >= limit:
            break

    return np.array(images), np.array(ages), np.array(genders)
# Corrected entry point
if __name__ == '__main__':
    X, y_age, y_gender = load_data(limit=5000)  # Load 5K images
    print("Images shape:", X.shape)
    print("Ages shape:", y_age.shape)
    print("Genders shape:", y_gender.shape)

    # Show a few images
    for i in range(5):
        plt.figure()
        plt.imshow(cv2.cvtColor((X[i] * 255).astype(np.uint8), cv2.COLOR_BGR2RGB))
        plt.title(f"Age: {y_age[i]}, Gender: {'Male' if y_gender[i] == 0 else 'Female'}")
        plt.axis('off')
        plt.show()
        plt.show()
