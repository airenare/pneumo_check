import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


print(tf.__version__)

def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, color_mode='grayscale', target_size=(128, 128))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


def predict_image(image_path, model):
    img = preprocess_image(image_path)
    pred = model.predict(img)
    class_names = ['NORMAL', 'PNEUMONIA']
    return class_names[np.argmax(pred)]


# Load the model
model = tf.keras.models.load_model('../models/model_final_1/model_final_1.keras')

# Image dimensions
img_height = 128
img_width = 128

# # Test image directory
# test_path = '../images/combined/test'
#
# # Create a dataframe with filenames of all images in the dataset in the first column and labels in the second
# test_df = pd.DataFrame(columns=['filename', 'label'])
#
# for subdir, dirs, files in os.walk(test_path):
#     for file in files:
#         # Get the label from the subdirectory name
#         label = subdir.split(os.path.sep)[-1]
#         # Add the filename and label to the dataframe
#         test_df = pd.concat([test_df, pd.DataFrame({'filename': os.path.join(subdir, file), 'label': label},
#                                                    index=[0])],
#                             ignore_index=True)
#
# # Iterate through the filename column of test_df and feed image paths to predict_image function,
# # add result to the new column
# test_df['prediction'] = test_df['filename'].apply(lambda x: predict_image(x, model))
# print(test_df.head())
#
# # Calculate accuracy, recall, precision and f1 score
# accuracy = accuracy_score(test_df['label'], test_df['prediction'])
# recall = recall_score(test_df['label'], test_df['prediction'], average='macro')
# precision = precision_score(test_df['label'], test_df['prediction'], average='macro')
# f1_score_macro = f1_score(test_df['label'], test_df['prediction'], average='macro')
# f1_score_micro = f1_score(test_df['label'], test_df['prediction'], average='micro')
# f1_score_weighted = f1_score(test_df['label'], test_df['prediction'], average='weighted')
#
# print(f"Accuracy: {accuracy:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"Precision: {precision:.2f}")
# print(f"F1 Score (Macro): {f1_score_macro:.2f}")
# print(f"F1 Score (Micro): {f1_score_micro:.2f}")
# print(f"F1 Score (Weighted): {f1_score_weighted:.2f}")



# Single image test
# Load the image
img_path = '../images/combined/test/NORMAL/IM-0001-0001.jpeg'

# Make prediction
predicted_class = predict_image(img_path, model)

# Print the prediction
print(f"Predicted class: {predicted_class}")

# Load the image
img_path = '../images/combined/test/PNEUMONIA/person14_virus_44.jpeg'

# Make prediction
predicted_class = predict_image(img_path, model)

# Print the prediction
print(f"Predicted class: {predicted_class}")


############################################################################################################
# model = tf.keras.Sequential([
#     layers.RandomRotation(0.2),
#     layers.RandomZoom(0.2),
#     layers.RandomFlip("horizontal"),
#     layers.RandomContrast(0.2),
#     layers.Conv2D(32, 3, activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(64, 3, activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(128, 3, activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(2, activation='softmax')
# ])
#
# # Compile the model
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.build(input_shape=(None, img_height, img_width, 1))
#
# model.summary()
#
# # Load the model weights
# model.load_weights('../models/model_1/model_weights.h5')
#
# test_image_path = "../images/showcase_images/pneumonia_3.jpg"
#
# # Preprocess the image
# def preprocess_image(image_path, img_height, img_width):
#     # Load the image
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         raise ValueError(f"Image at path {image_path} could not be loaded.")
#
#     # Resize the image
#     img = cv2.resize(img, (img_width, img_height))
#
#     # Normalize the image
#     img = img / 255.0
#
#     # Add a batch dimension
#     img = np.expand_dims(img, axis=0)
#
#     # Add a channel dimension
#     img = np.expand_dims(img, axis=-1)
#
#     return img
#
# # Get the predicted class
# def predict_class(image_path):
#     # Preprocess the image
#     img = preprocess_image(image_path, img_height, img_width)
#
#     # Make a prediction
#     predictions = model.predict(img)
#     predicted_class = np.argmax(predictions, axis=1)
#
#     return predicted_class[0]
#
# # Test prediction
# print(f"Predicted class for the test image: {predict_class(test_image_path)}")
#
# # Create image dataset from directory
# def create_image_dataset_from_directory(directory, img_height, img_width):
#     # Create the image dataset
#     image_dataset = image_dataset_from_directory(
#         directory,
#         image_size=(img_height, img_width),
#         color_mode='grayscale',
#         batch_size=32
#     )
#     return image_dataset
#
# test_ds = create_image_dataset_from_directory('../images/chest_xray/test', img_height, img_width)
#
# # Evaluate the model
# model.evaluate(test_ds)
#
# # Get the true labels and predicted labels
# true_labels = []
# predicted_labels = []
#
# for images, labels in test_ds:
#     predictions = model.predict(images)
#     predicted_labels.extend(np.argmax(predictions, axis=1))
#     true_labels.extend(labels.numpy())
#
# # Create a confusion matrix
# cm = confusion_matrix(true_labels, predicted_labels)
# print(cm)
#
# # Predict all images in the ../images/chest_xray/test/NORMAL folder
# normal_image_paths = os.listdir('../images/chest_xray/test/PNEUMONIA')
# normal_image_paths = [os.path.join('../images/chest_xray/test/PNEUMONIA', image) for image in normal_image_paths]
#
# for image_path in normal_image_paths:
#     predicted_class = predict_class(image_path)
#     print(f"Predicted class for {image_path}: {predicted_class}")
