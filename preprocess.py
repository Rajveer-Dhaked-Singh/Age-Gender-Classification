
import os
import cv2
import numpy as np

DATASET_PATH = 'Dataset/UTKFace/crop_part1'
IMAGE_SIZE = 200

# Load images, ages, and genders
def load_data(split=None, limit=None):
    images = []
    ages = []
    genders = []
    races = []

    filenames = sorted([f for f in os.listdir(DATASET_PATH) if f.endswith('.jpg')])

    # Optional train/test split
    if split == 'train':
        filenames = filenames[:int(0.8 * len(filenames))]
    elif split == 'test':
        filenames = filenames[int(0.8 * len(filenames)):]

    count = 0
    for filename in filenames:
        try:
            age, gender, race = map(int, filename.split('_')[:3])
        except:
            continue

        img_path = os.path.join(DATASET_PATH, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0

        images.append(img)
        ages.append(age)
        genders.append(gender)
        races.append(race)

        count += 1
        if limit and count >= limit:
            break

    return np.array(images), np.array(ages), np.array(genders), np.array(races)


# Only read age values from filenames for mean/std
def get_all_ages():
    filenames = sorted([f for f in os.listdir(DATASET_PATH) if f.endswith('.jpg')])
    ages = []
    for filename in filenames:
        try:
            age = int(filename.split('_')[0])
            ages.append(age)
        except:
            continue
    return np.array(ages)

# Compute mean and std only once
_ages_for_stats = get_all_ages()
mean_age = np.mean(_ages_for_stats)
std_age = np.std(_ages_for_stats)

print(f"[preprocess.py] Mean age: {mean_age:.2f}, Std age: {std_age:.2f}")
