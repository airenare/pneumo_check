import sklearn.metrics
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import io
import glob
import scipy.misc
import numpy as np
import pandas as pd
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import shutil
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras import Model
import matplotlib
from tensorflow.keras.optimizers import RMSprop
import os
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg

repo_url = 'https://github.com/adleberg/medical-ai'
repo_dir_path = os.path.abspath(os.path.join('.', os.path.basename(repo_url)))


def load_image_into_numpy_array(image):
    image = image.convert('RGB')
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


IMAGE_HEIGHT, IMAGE_WIDTH = 256, 256
LEARNING_RATE = 0.0001

# Prepare the data
finding = "cardiomegaly"
finding = finding.capitalize()

# Imported imaged folder
data_dir = "../images/medical-ai"

# Load the labels into a DataFrame
df = pd.read_csv(f"{data_dir}/labels.csv")

# Get the positive and negative labels
positives = df.loc[df["label"] == finding]
negatives = df.loc[df["label"] == "No Finding"]

# Number of positive examples
n = len(positives)

if n == 0:
    print("No studies found! Maybe check your spelling?")
    assert (n > 0)
else:
    print(f"Found {n} studies with {finding}")

# Set the train/test split ratio
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2
TRAIN_N = int(n * TRAIN_RATIO)
TEST_N = int(n * TEST_RATIO)
print(f"Number of Positives in Train:\t{TRAIN_N}, "
      f"Number of Positives in Test:\t{TEST_N}")

# Split the data into training and testing dataframes
train_labels = pd.concat([positives[:TRAIN_N], negatives[:TRAIN_N]])
test_labels = pd.concat([positives[TRAIN_N:], negatives[TRAIN_N:]])

print(f"Train DF:\t{test_labels.shape[0]}\nTest DF:\t{train_labels.shape[0]}")

# Prepare the data
rootdir = "../images/medical-ai/images/"


def move_files():
    # Create folders for train/test positive/negative
    os.makedirs(rootdir + finding + "/test/positive", exist_ok=True)
    os.makedirs(rootdir + finding + "/test/negative", exist_ok=True)
    os.makedirs(rootdir + finding + "/train/positive", exist_ok=True)
    os.makedirs(rootdir + finding + "/train/negative", exist_ok=True)

    # copy images to new directories for training purposes
    for idx, image in positives[:TRAIN_N].iterrows():
        source = rootdir + image["filename"]
        dst = rootdir + finding + "/train/positive/" + image["filename"]
        shutil.copy(source, dst)

    for idx, image in positives[TRAIN_N:].iterrows():
        source = rootdir + image["filename"]
        dst = rootdir + finding + "/test/positive/" + image["filename"]
        shutil.copy(source, dst)

    for idx, image in negatives[:TRAIN_N].iterrows():
        source = rootdir + image["filename"]
        dst = rootdir + finding + "/train/negative/" + image["filename"]
        shutil.copy(source, dst)

    for idx, image in negatives[TRAIN_N:n].iterrows():
        source = rootdir + image["filename"]
        dst = rootdir + finding + "/test/negative/" + image["filename"]
        shutil.copy(source, dst)

    print("Done moving " + str(n * 2) + " images to positive and negative folders.")


# Move files -- DISABLE IF FILES ARE ALREADY MOVED
# move_files()

# Load the images into memory to visualize them
positive_imgs, negative_imgs = [], []
IMAGE_HEIGHT, IMAGE_WIDTH = 256, 256

for idx, row in positives[:6].iterrows():
    image_path = rootdir + row["filename"]
    image = Image.open(image_path).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    positive_imgs.append(load_image_into_numpy_array(image))

for idx, row in negatives[:6].iterrows():
    image_path = rootdir + row["filename"]
    image = Image.open(image_path).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    negative_imgs.append(load_image_into_numpy_array(image))

# Visualize the images
for idx, img in enumerate(positive_imgs[:6]):
    plt.subplot(2, 3, idx + 1)
    plt.title(finding)
    plt.imshow(positive_imgs[idx])
plt.show()

for idx, img in enumerate(negative_imgs[:6]):
    plt.subplot(2, 3, idx + 1)
    plt.title("No Findings")
    plt.imshow(negative_imgs[idx])
plt.show()

# Create the model
# Load the InceptionV3 model
pre_trained_model = InceptionV3(
    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), weights='imagenet', include_top=False)

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

x = layers.Flatten()(last_output)  # Flatten the output layer to 1 dimension
x = layers.Dense(1024, activation='relu')(x)  # Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dropout(0.2)(x)  # Add a dropout rate of 0.2
x = layers.Dense(1, activation='sigmoid')(x)  # Add a final sigmoid layer for classification

model = Model(pre_trained_model.input, x)  # Configure and compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
print("Done compiling the model!")


# Define our example directories and files
base_dir = rootdir = "/kaggle/working/medical-ai/images/"
train_dir = os.path.join(base_dir, finding, 'train')
test_dir = os.path.join(base_dir, finding, 'test')

train_pos_dir = os.path.join(train_dir, 'positive')
train_neg_dir = os.path.join(train_dir, 'negative')
test_pos_dir = os.path.join(test_dir, 'positive')
test_neg_dir = os.path.join(test_dir, 'negative')