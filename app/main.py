# Import libraries only once when the server is started
import streamlit as st

# import pandas as pd

# Streamlit app
st.set_page_config(page_title='PneumoCheck', page_icon='🦠', initial_sidebar_state='expanded')
col1, col2 = st.columns([1, 2])

with col1:
    st.image('pictures/pneumocheck_detailed_logo.png', width=120, use_column_width=False)
with col2:
    st.title('PneumoCheck')
    st.write('Predict pneumonia from chest X-ray images.')

# Sidebar
st.sidebar.markdown('''
# About the model
- The model is based on the [InceptionV3](https://keras.io/api/applications/inceptionv3/) model with 
[ImageNet](https://www.image-net.org/) weights.
- Additional layers were added and tuned on the Chest X-Ray datasets from Kaggle 
[[1]](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) 
[[2]](https://www.kaggle.com/datasets/salonimate/covid-pneumonia-nomal-xray).
- The model achieved an AUC-ROC score of 0.984:
''')
st.sidebar.image('pictures/roc_curve.png', use_column_width=True)
st.sidebar.markdown('''
- The confusion matrix shows the model's performance on a test set of 1525 images:
''')
st.sidebar.image('pictures/confusion_matrix.png', use_column_width=True)
st.sidebar.markdown('''
- The cutoff probability was set to 0.24 to maximize both Sensitivity and Specificity. But it can be adjusted to suit 
the use case.
''')
st.sidebar.image('pictures/performance_hist.png', use_column_width=True)

# Sidebar footer copyright
st.sidebar.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        width: 100%;
        text-align: left;
        padding: 20px;
        color: #aaaaaa;
    }
    </style>
`    """, unsafe_allow_html=True)

st.sidebar.markdown('<div class="footer">© Anton Bakulin 2024</div>', unsafe_allow_html=True)

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
        border-color: #32cd33 !important;        
    }
    .pneumonia-border {
        border-color: #ff4500 !important;
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
def init_model(weights_folder):
    # model = load_model(model_path)
    from model.build_model import build_model
    model = build_model(weights_folder=weights_folder)
    print('Model cached')
    return model


np, Image, load_model, base64 = load_libraries()

# Load the model
cutoff = 0.24
weights_folder = 'model/weights'
# model_path = '../models/Pneumonia_ROC_0975_cutoff_024.keras'

model = None

try:
    model = init_model(weights_folder=weights_folder)
except ValueError:
    st.error('Model not found!\n\nPlease download it from \
        [HuggingFace](https://huggingface.co/airenare/InceptionV3_Pneumonia_CNN_v1/resolve/main/Pneumonia_ROC_0975_cutoff_024.keras?download=true)\
            (1.89 GB) and put it in the __/models__ folder.')


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
if model:
    uploaded_files = st.file_uploader('Choose an image...',
                                      type=['jpg', 'jpeg', 'png', 'webp', 'heic'],
                                      accept_multiple_files=True,
                                      label_visibility='collapsed')
    predictions = {}
    col1, col2, col3 = st.columns([1, 3, 1])
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            # Convert image to base64
            img_base64 = image_to_base64(image)

            prediction = predict_image(uploaded_file)
            label = interpret_prediction(prediction)
            predictions[uploaded_file.name] = [label, prediction]

            # Set border class based on prediction
            border_class = "normal-border" if label == "NORMAL" else "pneumonia-border"
            with col2:
                st.markdown(f"""
                                    <li>{uploaded_file.name}</li>
                                    <div class="uploaded-image {border_class}">
                                        <img src="data:image/jpeg;base64,{img_base64}" width="500"/>
                                    </div>
                                    <p>Prediction: {label} | Score: {prediction:.2f}</p>
                                    <hr>
                                    """, unsafe_allow_html=True)
                # st.markdown(f"__________________________")
        # Predictions dataframe from the predictions dictionary
        import pandas as pd
        predictions_df = pd.DataFrame(predictions).T.reset_index()

    if predictions:
        # Set column names
        predictions_df.columns = ['File Name', 'Predicted Label', 'Score']
        st.write(predictions_df)
        # Button to save predictions to a CSV file with download location dialog
        st.download_button(label='Download predictions',
                           data=predictions_df.to_csv(index=False),
                           file_name='predictions.csv',
                           mime='text/csv')
