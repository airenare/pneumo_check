from tensorflow.keras.models import load_model

model = load_model('../models/hidden/kaggle/working/export/Pneumonia_ROC_0975_cutoff_024.keras')

# save weights
model.save_weights('../models/hidden/kaggle/working/export/Pneumonia_ROC_0975_cutoff_024.weights.h5')

# save model as json file
model_json = model.to_json()
with open('../models/hidden/kaggle/working/export/Pneumonia_ROC_0975_cutoff_024.json', 'w') as json_file:
    json_file.write(model_json)

# load model from json file
from tensorflow.keras.models import model_from_json

json_file = open('../models/hidden/kaggle/working/export/Pneumonia_ROC_0975_cutoff_024.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights('../models/hidden/kaggle/working/export/Pneumonia_ROC_0975_cutoff_024.weights.h5')

# Predict test image
import numpy as np
from PIL import Image

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
    return loaded_model.predict(expanded)[0][0]

def interpret_prediction(prediction):
    if prediction < cutoff:
        return 'NORMAL'
    else:
        return 'PNEUMONIA'

cutoff = 0.24
image_width = 256
image_height = 256
# Test image
filename = '../images/chest_xray/test/NORMAL/IM-0001-0001.jpeg'

prediction = predict_image(filename)
interpretation = interpret_prediction(prediction)
print(f'Prediction: {prediction}, Interpretation: {interpretation}')