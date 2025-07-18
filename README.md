
# 👤 Gender, Age & Race Prediction using CNN (UTKFace)

This project builds a deep learning model using TensorFlow and Keras to predict **Age**, **Gender**, and **Race** from facial images. It is trained on the UTKFace dataset.

---

## 🗂️ Project Structure

```

Gender and Age Classification/
├── UTKFace/                 # Dataset folder (downloaded separately)
├── preprocess.py            # Preprocessing and data loader
├── model.py                 # CNN model definition
├── train.py                 # Training script
├── predict.py               # Real-time webcam predictions
├── requirements.txt         # Python dependencies
└── README.md                # Project description (this file)

````

---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/yourusername/age-gender-race-predictor.git
cd age-gender-race-predictor

# Create virtual environment (optional)
python -m venv venv
venv\Scripts\activate  # On Windows

# Install requirements
pip install -r requirements.txt
````

Make sure:

* You have TensorFlow with GPU support installed
* CUDA and cuDNN are properly set up
* Your GPU is recognized by TensorFlow

---

## 🧠 Model Overview

* **Input**: 200x200 RGB face images
* **Outputs**:

  * Age (Regression)
  * Gender (Binary Classification: Male/Female)
  * Race (5-Class Classification: White, Black, Asian, Indian, Others)

---

## 🚀 Training

```bash
python train.py
```

This:

* Loads UTKFace dataset
* Trains a multi-output CNN
* Saves model to `age_model.keras`

Metrics displayed:

* Age MAE (normalized and actual)
* Gender Accuracy
* Race Accuracy

---

## 🎥 Webcam Prediction

```bash
python predict.py
```

This opens your webcam and:

* Detects face in real-time
* Predicts age, gender, and race
* Displays live results

---

## 📊 Example Results

```
--- Test Metrics ---
Total Loss         : 3.0009
Age MAE (actual)   : 8.81 years
Gender Accuracy    : 89.66%
Race Accuracy      : 86.85%
```

---

## 📚 Dataset

Download the UTKFace dataset from:
🔗 [https://susanqq.github.io/UTKFace/](https://susanqq.github.io/UTKFace/)

Then place it in:

```
/Gender and Age Classification/UTKFace/
```

---

## 🧾 Dependencies

Key libraries:

* `tensorflow`
* `opencv-python`
* `numpy`
* `scikit-learn`

---

## 👨‍💻 Author

**Rajveer Singh Dhaked**
IIT JODHPUR
🔗 GitHub: [Rajveer-Dhaked-Singh](https://github.com/Rajveer-Dhaked-Singh)

---

## 📜 License

MIT License —


