import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
                                     GlobalAveragePooling2D, BatchNormalization, SeparableConv2D)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from Preprocess import load_data
from sklearn.model_selection import train_test_split

# Load and normalize data
X, y_age, y_gender, y_race = load_data(limit=20000)
mean_age = np.mean(y_age)
std_age = np.std(y_age)
y_age_norm = (y_age - mean_age) / std_age

# Split data
X_train, X_val, age_train, age_val, gen_train, gen_val, race_train, race_val = train_test_split(
    X, y_age_norm, y_gender, y_race, test_size=0.2, random_state=42
)

# Model architecture
input_img = Input(shape=(200, 200, 3))

x = SeparableConv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)

x = Dropout(0.4)(x)

# Output branches
age_output = Dense(1, name='age_output')(x)  # Regression
gender_output = Dense(1, activation='sigmoid', name='gender_output')(x)  # Binary
race_output = Dense(5, activation='softmax', name='race_output')(x)  # 5-class

# Full model
model = Model(inputs=input_img, outputs=[age_output, gender_output, race_output])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=5e-4),
    loss={
        'age_output': 'mse',
        'gender_output': 'binary_crossentropy',
        'race_output': tf.keras.losses.SparseCategoricalCrossentropy()
    },
    loss_weights={
        'age_output': 10,
        'gender_output': 7,
        'race_output': 5
    },
    metrics={
        'age_output': ['mae'],
        'gender_output': ['accuracy', tf.keras.metrics.AUC(name='auc')],
        'race_output': ['accuracy']
    }
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_age_output_mae', patience=8, restore_best_weights=True, verbose=1, mode='min'),
    ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    ModelCheckpoint('model_output_keras.keras', save_best_only=True, verbose=1)
]


# Train model directly with NumPy arrays
model.fit(
    x=X_train,
    y={
        'age_output': age_train,
        'gender_output': gen_train,
        'race_output': race_train
    },
    validation_data=(
        X_val,
        {
            'age_output': age_val,
            'gender_output': gen_val,
            'race_output': race_val
        }
    ),
    epochs=50,
    batch_size=64,
    callbacks=callbacks
)
