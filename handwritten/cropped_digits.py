import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.io import loadmat
import cv2

# Load the trained CNN model
model = tf.keras.models.load_model('handwritten_cnn.model')

# Path to the .mat file
mat_file_path = 'C:/Users/User/Desktop/handwritten/train_32x32.mat'

# Load .mat file
mat_data = loadmat(mat_file_path)
X = mat_data['X']
y = mat_data['y']

# Iterate over images in the .mat file
for i in range(X.shape[3]):
    # Extract image and label
    img = X[:, :, :, i]
    label = int(y[i])

    # Preprocess the image
    img = cv2.resize(img, (28, 28))
    img = np.invert(img)
    img = tf.keras.utils.normalize(img, axis=(0, 1))  # Normalize along height and width

    # Ensure the image has the correct shape (28, 28, 1)
    img = np.expand_dims(img[:, :, 0], axis=-1)

    # Make predictions
    prediction = model.predict(np.array([img]))
    predicted_label = np.argmax(prediction)

    print(f'Image: {i}, True Label: {label}, Predicted Label: {predicted_label}')

    # Display the image
    plt.imshow(img[:, :, 0], cmap=plt.cm.binary)
    plt.title(f'True Label: {label}, Predicted Label: {predicted_label}')
    plt.show()
