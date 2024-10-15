import tarfile
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Extract dataset from tar.gz if needed
fname = 'fer2013.tar.gz'
if fname.endswith("tar.gz"):
    tar = tarfile.open(fname, "r:gz")
    tar.extractall()
    tar.close()

# Load and preprocess the dataset
df = pd.read_csv('fer2013/fer2013.csv')
label_to_text = {0: 'anger', 1: 'disgust', 2: 'fear', 3: 'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

# Convert pixel strings to numpy arrays
img_array = df.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
img_array = np.stack(img_array, axis=0)

# Resize images to (224, 224) and convert to RGB
img_array_resized = np.array([tf.image.resize(img, (224, 224)).numpy() for img in img_array])
img_array_rgb = np.repeat(img_array_resized, 3, axis=-1)  # Convert grayscale to RGB

# Labels
labels = df.emotion.values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(img_array_rgb, labels, test_size=0.1, random_state=42)

# Normalize data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Load pre-trained MobileNetV2 without top layers
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

# Add custom layers
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model checkpoint and learning rate reduction callbacks
checkpoint_path = 'best_model.keras'
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path,
                                      monitor='val_accuracy',
                                      verbose=1,
                                      save_best_only=True,
                                      mode='max')

reduce_lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=1e-6)

# Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=32),
          epochs=20,
          validation_data=(X_test, y_test),
          callbacks=[checkpoint_callback, reduce_lr_callback])

# Load the best model
final_model = tf.keras.models.load_model(checkpoint_path)

# Model is now trained and can be saved or deployed for inference
