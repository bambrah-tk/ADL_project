import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


model = tf.keras.models.load_model('handwritten_cnn.model')

image_directory = 'digits'

# Iterate over PNG images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(".png"):
        # Load and preprocess the image for prediction
        img_path = os.path.join(image_directory, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        img = np.invert(img)
        img = np.expand_dims(img, axis=-1)
        img = tf.keras.utils.normalize(img, axis=1)

        # Make predictions
        prediction = model.predict(np.array([img]))
        predicted_label = np.argmax(prediction)

        print(f'Image: {filename}, Predicted Label: {predicted_label}')

        # Display the image
        plt.imshow(img[:, :, 0], cmap=plt.cm.binary)
        plt.title(f'Predicted Label: {predicted_label}')
        plt.show()

#evaluation
# two missclasifications
# 5 was classified as 0
# 9 as 2