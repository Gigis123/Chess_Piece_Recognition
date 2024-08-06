import streamlit as st
import numpy as np
import requests
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from keras.models import load_model
model = load_model('best_model.h5')
class_names = ["Bishop", "King", "Knight", "Pawn", "Queen", "Rook"]

# Function to predict from url
def predict_from_url(url):
    response = requests.get(url, stream=True)
    img = Image.open(response.raw)
    img = img.resize((224, 224))
    img = img.convert('RGB')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_batch = img_batch / 255.0
    prediction_inf = model.predict(img_batch)
    result_max_proba = prediction_inf.argmax(axis=-1)[0]
    result_class = class_names[result_max_proba]
    return prediction_inf, result_class


# Function to predict from file 
def predict_from_file(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0) 
    img_batch = img_batch / 255.0
    prediction_inf = model.predict(img_batch)
    result_max_proba = prediction_inf.argmax(axis=-1)[0]
    result_class = class_names[result_max_proba]
    return prediction_inf, result_class

# Function to run model
def run():
    st.write("# Chess Computer Vision Prediction Model")
    st.write('### welcome to our chess piece computer vision model using VGG19')
    st.write('This model aims to classify chess pieces from images uploaded by the user or using a url from a website we will try to classify the chess pieces which includes the following')
    st.write('[Bishop, King, Knight, Pawn, Queen, Rook]')

    st.write('-'* 50)

    st.write('##### Please choose your prefered method of upload')
    option = st.selectbox('Choose Input Method:', ('Use Own File', 'Use URL'))

    if option == 'Use Own File':
        st.write('##### Please upload your file below')
        uploaded_file = st.file_uploader("Please upload a jpg or jpeg file", type=['jpg', 'jpeg'])
        if uploaded_file is not None:
            bytes_data = uploaded_file.read()
            st.image(bytes_data, caption='Uploaded Image.', use_column_width=True)

            # Save file locally
            with open(f'temp.jpg', 'wb') as f:
                f.write(bytes_data)

            # Perform prediction
            prediction_inf, result_class = predict_from_file('temp.jpg')

            st.write('Result:', result_class)
            st.write('Confidence:', prediction_inf)
            st.write('Class Names (in order):', class_names)


    elif option == 'Use URL':
        st.write('##### Please enter an image url below')
        url = st.text_input('Enter Image URL:', placeholder='https://i.ebayimg.com/images/g/o-0AAOSwvQ1kTvuh/s-l1200.webp')
        if url != '':
            try:
                prediction_inf, result_class = predict_from_url(url)
                st.image(url, caption='Provided Image.', use_column_width=True)
                st.write('Prediction:', result_class)
                st.write('Confidence:', prediction_inf)
                st.write('Class Names (in order):', class_names)
            except Exception as e:
                st.write("Error loading image from URL:", str(e))

    st.write('-'* 50)
    
if __name__ == '__main__':
    run()