import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Set dataset path
dataset_path = "C:/Users/sarka/OneDrive/Desktop/AgriSens-master/AgriSens-master/PLANT-DISEASE-IDENTIFICATION/New Plant Diseases Dataset(Augmented)/train"

# Load and preprocess dataset
training_set = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=123
)

# Class names
class_names = training_set.class_names
print("Class names:", class_names)

# Normalize pixel values
normalization_layer = layers.Rescaling(1./255)
training_set = training_set.map(lambda x, y: (normalization_layer(x), y))

# Split training and validation datasets
val_size = int(0.2 * len(training_set))
val_ds = training_set.take(val_size)
train_ds = training_set.skip(val_size)


# Prefetch for performance
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50
)

# Save model
model.save("trained_plant_disease_model.keras")
print("Model saved as trained_plant_disease_model.keras")

# Plot accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

plt.show()
