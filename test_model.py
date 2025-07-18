import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# --- Constants ---
IMAGE_SIZE = 200
mean_age = 29.42
std_age = 24.78
gender_labels = ['Male', 'Female']
race_labels = ['White', 'Black', 'Asian', 'Indian', 'Others']

# --- Load model and face detector ---
print("[INFO] Loading model and Haar cascade...")
model = load_model('model_output_keras.keras')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# --- Webcam setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not access webcam.")
    exit()

print("[INFO] Webcam started. Press 'q' to quit.")
prev_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame not read correctly.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (IMAGE_SIZE, IMAGE_SIZE))
            face_input = np.expand_dims(face_img / 255.0, axis=0)

            # Predict age, gender, race
            try:
                pred_age, pred_gender, pred_race = model.predict(face_input, verbose=0)
                age = pred_age[0][0] * std_age + mean_age
                gender = gender_labels[int(pred_gender[0][0] > 0.5)]
                race = race_labels[np.argmax(pred_race[0])]

                # Display predictions
                label = f"{gender}, {race}, {age:.1f} yrs"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            except Exception as e:
                print("[ERROR] Prediction failed:", e)
                cv2.putText(frame, "Prediction error", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show webcam frame
        cv2.imshow("Webcam - Age, Gender, Race", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user.")

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("[INFO] Webcam closed.")
