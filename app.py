# this version of app allows to just upload one image

import os
import streamlit as st
import cv2
import numpy as np
from keras.models import load_model, Model
import keras.backend as K


# Function to preprocess the image
def preprocess_image(img):
    """
    Converts image to shape (32, 128, 1) & normalize
    """
    w, h = img.shape
    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape
    
    img = img.astype('float32')
    
    if w < 32:
        add_zeros = np.full((32-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape
    
    if h < 128:
        add_zeros = np.full((w, 128-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape
        
    if h > 128 or w > 32:
        dim = (128,32)
        img = cv2.resize(img, dim)
    
    img = cv2.subtract(255, img)
    
    img = np.expand_dims(img, axis=2)
    
    img = img / 255
    
    return img

char_set = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" 

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

# function to perform OCR prediction
def perform_ocr(img_path):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = preprocess_image(img)

        # Load the model
        loaded_model = load_model('/kaggle/input/model/ocr_word_model_1.h5', custom_objects={'<lambda>': ctc_lambda_func})
        dense_layer = loaded_model.get_layer('dense')
        prediction_model = Model(inputs=loaded_model.input, outputs=dense_layer.output)

        # Get predictions
        test_prediction = prediction_model.predict([np.array([img]), np.zeros((1, 19)), np.ones((1, 1)) * 31, np.ones((1, 1)) * 19])
        test_decoded = K.ctc_decode(test_prediction, input_length=np.ones(test_prediction.shape[0]) * test_prediction.shape[1], greedy=True)[0][0]
        test_out = K.get_value(test_decoded)[0]

        # Decode predictions
        predicted_text = ''.join([char_set[int(p)] for p in test_out if int(p) != -1])

        return predicted_text
    except:
        return "Error processing image"

# actual Streamlit app
def main():
    st.title("OCR Streamlit App")

    uploaded_file = st.file_uploader("Choose an image...", type="png")

    if uploaded_file is not None:

        file_name = uploaded_file.name

        st.image(uploaded_file, caption=f"Uploaded Image: {file_name}", use_column_width=True)
        st.write("")

        classification_status = st.empty()

        classification_status.text("Classifying...")

        # save the uploaded file temporarily
        temp_file_path = "temp_image.png"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # perform OCR on the uploaded image
        predicted_text = perform_ocr(temp_file_path)

        classification_status.text(f"Predicted Text: {predicted_text}")

        # remove the temporary file
        os.remove(temp_file_path)

if __name__ == "__main__":
    main()