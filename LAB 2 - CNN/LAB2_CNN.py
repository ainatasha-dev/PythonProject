import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Normalize
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Build CNN model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)), # Add this explicit Input layer
    layers.Conv2D(32, (3,3), activation='relu'), # Remove input_shape from here
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(train_images, train_labels, epochs=10)

# Evaluate
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("\nTest Accuracy:", test_acc)

import numpy as np

# Pick a random image from the test set
image_index = 44
plt.imshow(test_images[image_index].reshape(28, 28), cmap='Greys')
plt.show()

# Predict the digit
pred = model.predict(test_images[image_index].reshape(1, 28, 28, 1))
print(f"Prediction: {np.argmax(pred)}")
print(f"Actual Label: {test_labels[image_index]}")