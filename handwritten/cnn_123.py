import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load and preprocess the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
x_train = np.expand_dims(x_train, axis=-1)  # Add channel dimension
x_test = np.expand_dims(x_test, axis=-1)    # Add channel dimension

# Create a CNN model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1000, activation='softmax'))  # Adjust units for three-digit labels


# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Save the model
model.save('handwritten_cnn123.model')

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')

# Load an example image and make predictions
img = cv2.imread("digit123.png", cv2.IMREAD_GRAYSCALE)  # Read in grayscale
img = cv2.resize(img, (28, 28))
img = np.invert(img)  # Invert the colors
img = np.expand_dims(img, axis=-1)  # Add channel dimension
img = tf.keras.utils.normalize(img, axis=1)  # Normalize the image

# Make predictions
prediction = model.predict(np.array([img]))
predicted_label = np.argmax(prediction)
print(f'This digit is probably {predicted_label}')

# Display the image
plt.imshow(img[:, :, 0], cmap=plt.cm.binary)
plt.show()
