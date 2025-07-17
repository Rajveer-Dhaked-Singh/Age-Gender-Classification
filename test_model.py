import numpy as np
import cv2
from tensorflow.keras.models import load_model

model=load_model('gender_model.h5')

img = cv2.imread('download (1).jpg')
img = cv2.resize(img, (200, 200))
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=0)
#img = img.reshape(1,-1)

predictions = model.predict(img)
class_idx= np.argmax(predictions, axis=1)

labels = ['Male', 'Female']
print("predicted gender:", labels[class_idx[0]])