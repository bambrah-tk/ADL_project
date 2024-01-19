# this version of app allows to just upload several images

import os
import streamlit as st
import cv2
import numpy as np
from keras.models import load_model, Model
import keras.backend as K


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
        add_zeros = np.full((32 - w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape

    if h < 128:
        add_zeros = np.full((w, 128 - h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape

    if h > 128 or w > 32:
        dim = (128, 32)
        img = cv2.resize(img, dim)

    img = cv2.subtract(255, img)

    img = np.expand_dims(img, axis=2)

    img = img / 255

    return img

char_set = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def perform_ocr(img_path):
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = preprocess_image(img)

        loaded_model = load_model('/kaggle/input/model/ocr_word_model_1.h5',
                                  custom_objects={'<lambda>': ctc_lambda_func})
        dense_layer = loaded_model.get_layer('dense')
        prediction_model = Model(inputs=loaded_model.input, outputs=dense_layer.output)

        test_prediction = prediction_model.predict(
            [np.array([img]), np.zeros((1, 19)), np.ones((1, 1)) * 31, np.ones((1, 1)) * 19])
        test_decoded = K.ctc_decode(test_prediction,
                                    input_length=np.ones(test_prediction.shape[0]) * test_prediction.shape[1],
                                    greedy=True)[0][0]
        test_out = K.get_value(test_decoded)[0]

        predicted_text = ''.join([char_set[int(p)] for p in test_out if int(p) != -1])

        return predicted_text, img
    except:
        return "Error processing image", None


# function to perform OCR on all images
def perform_ocr_on_folder(folder_path):
    results = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            original_img = cv2.imread(img_path)
            predicted_text, preprocessed_img = perform_ocr(img_path)
            results.append((filename, predicted_text, original_img, preprocessed_img))
    return results


# actual Streamlit app
def main():
    st.title("OCR Streamlit App")

    uploaded_files = st.file_uploader("Choose a folder...", type="png", accept_multiple_files=True)

    classification_status = st.empty()

    if uploaded_files is not None:
        classification_status.text("Classifying...")

        # again create a temporary folder to save uploaded images
        temp_folder_path = "temp_folder"
        os.makedirs(temp_folder_path, exist_ok=True)

        # save the uploaded files temporarily
        for uploaded_file in uploaded_files:
            with open(os.path.join(temp_folder_path, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getvalue())

        # perform OCR on all images in the temporary folder
        results = perform_ocr_on_folder(temp_folder_path)

        # here we display results
        for filename, predicted_text, original_img, preprocessed_img in results:
            st.success(f"Image: {filename}, Predicted Text: {predicted_text}")
            st.image(original_img, caption=f"Original Image: {filename}", use_column_width=True)

        classification_status.text("Classification complete!")

        # remove the temporary folder and files
        for filename in os.listdir(temp_folder_path):
            uploaded_file.seek(0)
            os.remove(os.path.join(temp_folder_path, filename))
        os.rmdir(temp_folder_path)


if __name__ == "__main__":
    main()