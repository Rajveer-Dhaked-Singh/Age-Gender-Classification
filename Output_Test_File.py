import numpy as np
from tensorflow.keras.models import load_model
from Preprocess import load_data, mean_age, std_age
from sklearn.metrics import classification_report, mean_absolute_error

# Load test data
X_test, y_age_test, y_gender_test, y_race_test = load_data(split='test', limit=None)

# Normalize age (as done during training)
y_age_test_norm = (y_age_test - mean_age) / std_age

# Load trained multi-output model
model = load_model('model_output_keras.keras')

# Evaluate model
results = model.evaluate(
    X_test,
    {'age_output': y_age_test_norm, 'gender_output': y_gender_test, 'race_output': y_race_test},
    verbose=2
)

# Print metrics with their names
print("\n--- Raw Keras Evaluation ---")
for name, val in zip(model.metrics_names, results):
    print(f"{name}: {val:.4f}")

# Extract useful metrics
metrics = dict(zip(model.metrics_names, results))
age_mae_norm = metrics.get("age_output_mae", 0.0)
gender_acc = metrics.get("gender_output_accuracy", 0.0)
race_acc = metrics.get("race_output_accuracy", 0.0)

# Convert normalized MAE to actual years
age_mae = age_mae_norm * std_age

print("\n--- Test Metrics ---")
print(f"Age MAE (normalized): {age_mae_norm:.4f}")
print(f"Age MAE (actual)   : {age_mae:.2f} years")
print(f"Gender Accuracy    : {gender_acc * 100:.2f}%")
print(f"Race Accuracy      : {race_acc * 100:.2f}%")

# Make predictions
age_pred_norm, gender_logits, race_logits = model.predict(X_test, verbose=0)

# Un-normalize predicted age
age_pred = age_pred_norm * std_age + mean_age

# Handle gender logits (if shape is (N, 1), use threshold; else use argmax for multi-class)
if gender_logits.shape[1] == 1:
    gender_pred = (gender_logits > 0.5).astype("int").reshape(-1)
else:
    gender_pred = np.argmax(gender_logits, axis=1)

race_pred = np.argmax(race_logits, axis=1)

# Evaluation reports
print("\nGender Classification Report:")
print(classification_report(y_gender_test, gender_pred, target_names=["Male", "Female"]))

print("Race Classification Report:")
try:
    print(classification_report(y_race_test, race_pred))
except:
    print("Race classification report failed — possibly due to unexpected race labels.")

# Final Age MAE
print(f"\nAge MAE (final, actual years): {mean_absolute_error(y_age_test, age_pred):.2f} years")
