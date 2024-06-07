# Import libraries only once when the server is started
import streamlit as st

# Streamlit app

st.set_page_config(page_title='Pneumonia Classifier', layout='wide')
st.title('Pneumonia Classifier')

image_width, image_height = 256, 256


# CSS to style the image uploader and the predicted images
st.markdown("""
    <style>
    .uploaded-image {
        width: auto;
        height: auto;
        object-fit: cover;
        border: 5px solid transparent;
        display: inline-block;
    }
    .normal-border {
        border-color: green !important;        
    }
    .pneumonia-border {
        border-color: red !important;
    }
    </style>
    """, unsafe_allow_html=True)


# This piece of code runs once when the server is started
@st.cache_resource
def load_libraries():
    import numpy as np
    from PIL import Image
    from tensorflow.keras.models import load_model
    import base64
    print('Imported libraries')
    return np, Image, load_model, base64


@st.cache_resource
def init_model(model_path):
    model = load_model(model_path)
    print('Model cached')
    return model


np, Image, load_model, base64 = load_libraries()

# Load the model
cutoff = 0.24
model_path = '../models/kaggle/working/export/Pneumonia_ROC_0975_cutoff_024.keras'
model = init_model(model_path)


# Function to convert image to base64
def image_to_base64(image):
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()



def load_image_into_numpy_array(image):
    image = image.convert('RGB')
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def predict_image(filename):
    image = Image.open(filename).resize((image_width, image_height))
    image_np = load_image_into_numpy_array(image)
    exp = np.true_divide(image_np, 255.0)
    expanded = np.expand_dims(exp, axis=0)
    return model.predict(expanded)[0][0]


def interpret_prediction(prediction):
    if prediction < cutoff:
        return 'NORMAL'
    else:
        return 'PNEUMONIA'




# make image uploader that supports drag and drop and multiple files
uploaded_files = st.file_uploader('Choose an image...',
                                  type=['jpg', 'jpeg', 'png', 'webp', 'heic'],
                                  accept_multiple_files=True,
                                  label_visibility='collapsed')
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        # Convert image to base64
        img_base64 = image_to_base64(image)

        prediction = predict_image(uploaded_file)
        label = interpret_prediction(prediction)



        # Set border class based on prediction
        border_class = "normal-border" if label == "NORMAL" else "pneumonia-border"
        st.markdown(f"""
                            <div class="uploaded-image {border_class}">
                                <img src="data:image/jpeg;base64,{img_base64}" width="500"/>
                            </div>
                            <p>Prediction: {label} | Probability: {prediction:.2f}</p>
                            <hr>
                            """, unsafe_allow_html=True)

# Predictions dataframe