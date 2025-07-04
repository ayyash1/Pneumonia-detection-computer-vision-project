import base64
import streamlit as st
import numpy as np

from PIL import Image, ImageOps

def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)


def classify(image, model, class_names):
    '''
    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
      a tuple of class name and confidence score for the prediction

    '''
    #convert image to (224,244)
    image = ImageOps.fit(image, (224,224), Image.Resampling.LANCZOS)

    #convert image to numpy array
    image_array = np.asarray(image)

    #normalize image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    #set the model input
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)     # determined by the first position in the shape tuple, in this case 1
    data[0] = normalized_image_array     # Load the image into the array


    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name[2:], confidence_score
