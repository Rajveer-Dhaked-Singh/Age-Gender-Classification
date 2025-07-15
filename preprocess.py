import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH='Dataset/UTKFace/'
IMAGE_SIZE = 200

def load_data(limit=10000):
    images = []
    ages = []
    genders = []
    
    count=0
    for filename in os.listdir(DATASET_PATH):
        if  not filename.endswith('.jpg'):
            continue
        
        try:
            parts=filename.split('_')
            age = int(parts[0])
            gender = int(parts[1])
            
        except:
            continue
        
        
        img_path = os.path.join(DATASET_PATH, filename)
        img = cv2.imread(img_path)
        if img is not None:
            continue
        

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img =img/255.0

        images.append(img)
        ages.append(age)
        genders.append(gender)

        count += 1
        if count >= limit:
            break

    return np.array(images), np.array(ages), np.array(genders)

if __name__ == "__main__":
    X, y_age, y_gender = load_data(limit=5000)   
    print(f"Loaded {len(X)} images, ages, and genders.")   
    print(f"Sample image shape: {X[0].shape}")
    print(f"Sample age: {y_age[0]}, gender: {y_gender[0]}")         
    plt.imshow(X[0])
    plt.title(f"Age: {y_age[0]}, Gender: {y_gender[0]}")
    plt.axis("off")
    plt.show()
      