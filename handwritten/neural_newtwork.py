import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#mnist = tf.keras.datasets.mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train = tf.keras.utils.normalize(x_train, axis=1)
#x_test = tf.keras.utils.normalize(x_test, axis=1)
#model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
#model.add(tf.keras.layers.Dense(128,activation='relu'))
#model.add(tf.keras.layers.Dense(128, activation='relu'))
#model.add(tf.keras.layers.Dense(10, activation='softmax'))
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#model.fit(x_train, y_train, epochs = 1)

#model.save('handwritten.model')

model = tf.keras.models.load_model('handwritten.model')
#loss, accuracy = model.evaluate(x_test, y_test)
#print(loss)
#print(accuracy)

img =cv2.imread("digit1.png", cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = np.invert(img)
img = np.array([img])
img = tf.keras.utils.normalize(img, axis=1)
prediction = model.predict(img)
print(f"This digit is prob {np.argmax(prediction)}")
plt.imshow(img[0], cmap=plt.cm.binary)
plt.show()