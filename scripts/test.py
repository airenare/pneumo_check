import numpy as np
import pandas as pd
import os
import cv2
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model

print("Done")

# Load the model
model = load_model("../models/mobil_dense_stack_v1.keras")

# Load the test image
test_image_path = "../showcase_images/pneumonia_1.jpg"
test_image_path = "../showcase_images/pneumonia_3.jpg"
# test_image_path = '../showcase_images/normal_1.jpg'

# Create a list of test images that are within the showcase_images folder
test_images = os.listdir("../showcase_images")
# Leave only the images that have the .jpg and .webp extensions
test_images = [
    image
    for image in test_images
    if image.endswith(".jpg")
    or image.endswith(".webp")
    or image.endswith(".jpeg")
    or image.endswith(".png")
]


# Image preprocessing functions
def process_image(image):
    """Takes the image array and normalizes it
    by dividing it by 255 and resizing it to (224,224)."""

    image = image / 255
    image = cv2.resize(image, (224, 224))
    return image


def predict_image(image_path, model):
    """Takes the image array and the model and returns the
    prediction of the model."""
    print(image_path)
    im = cv2.imread(image_path)
    test_image = np.array(im)
    processed_image = process_image(test_image)
    processed_image = np.expand_dims(processed_image, axis=0)
    prediction = model.predict(processed_image)

    return prediction


def interpret_prediction(prediction):
    """Takes the prediction and returns the class of the prediction."""

    if prediction[0][0] < 0.5:
        return "Normal"
    else:
        return "Pneumonia"


# Make the prediction
# prediction = interpret_prediction(predict_image(test_image_path, model))
# print(prediction)

# Make prediction for all images in the showcase_images folder
for image in test_images:
    prediction = interpret_prediction(
        predict_image("../showcase_images/" + image, model)
    )
    print(f"{image} is {prediction}")


# Get all images from the showcase_images/New-CNP-Dataset/validation folder and organize them into 3 lists based on their class (folder name within the validation folder)
def get_images_from_folder(folder_path):
    """Takes the path of the folder and returns a list of images within that folder."""

    images = os.listdir(folder_path)
    images = [
        image
        for image in images
        if image.endswith(".jpg")
        or image.endswith(".webp")
        or image.endswith(".jpeg")
        or image.endswith(".png")
    ]
    return images


test_image_folder = "../showcase_images/New-CNP-Dataset/validation"
# Make sure that labels are only folders and not files
labels = [
    label
    for label in os.listdir(test_image_folder)
    if os.path.isdir(test_image_folder + "/" + label)
]
# labels = os.listdir(test_image_folder)
images = [get_images_from_folder(test_image_folder + "/" + label) for label in labels]

# Create a dataframe with the image paths and their labels
image_paths = []
image_labels = []
for i, label in enumerate(labels):
    for image in images[i]:
        image_paths.append(test_image_folder + "/" + label + "/" + image)
        image_labels.append(label)

df = pd.DataFrame({"image_path": image_paths, "label": image_labels})

# Make predictions for all images in the dataframe and add the predictions to the dataframe's predictions column
df["prediction"] = df["image_path"].apply(
    lambda x: interpret_prediction(predict_image(x, model))
)
df["prediction"] = df["prediction"].str.lower()
df["label_no_covid"] = df["label"]
df["label_no_covid"] = df["label_no_covid"].replace("covid", "pneumonia")

print(df.head())

# Calculate the accuracy of the model
df["correct"] = df["label_no_covid"] == df["prediction"]
accuracy = df["correct"].sum() / df["correct"].count()
print(f"Accuracy: {accuracy}")

# Save the dataframe to a csv file
df.to_csv("../predictions.csv", index=False)
print("Done")

# plot the confusion matrix
conf_matrix = confusion_matrix(df["label_no_covid"], df["prediction"])
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Stacked Model\nConfusion Matrix")
plt.savefig("../stacked_model_confusion_matrix.png")
plt.show()

print("Done")
