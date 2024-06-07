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
from sklearn.metrics import confusion_matrix, auc
from PIL import Image

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

print(tf.__version__)

# Image dimensions
image_height = 256
image_width = 256


def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, color_mode='rgb', target_size=(image_height, image_width))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


def predict_image(image_path, model):
    img = preprocess_image(image_path)
    pred = model.predict(img)
    print(f"Prediction: {pred} | Cut-off: {cutoff}")
    class_names = ['NORMAL', 'PNEUMONIA']
    cls = class_names[0] if pred < cutoff else class_names[1]
    return cls


# Load the model
model_path = '../models/kaggle/working/export/Pneumonia_ROC_0975_cutoff_024.keras'
model = tf.keras.models.load_model(model_path)
# model = tf.keras.models.load_model('../models/model_final_1/model_final_1.keras')

# Prefiction's cutoff
cutoff = 0.24

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
"""predicted_class = predict_image(img_path, model)

# Print the prediction
print(f"Label: {img_path.split("/")[-2]} | Predicted: {predicted_class}")

# Load the image
img_path = '../images/combined/test/PNEUMONIA/person14_virus_44.jpeg'
img_path = '../images/chest_xray/train/PNEUMONIA/person1_bacteria_1.jpeg'


# Make prediction
predicted_class = predict_image(img_path, model)

# Print the prediction
print(f"Label: {img_path.split("/")[-2]} | Predicted: {predicted_class}")"""

test_neg_dir = '../images/chest_xray/test/NORMAL'
test_pos_dir = '../images/chest_xray/test/PNEUMONIA'


# df_results = pd.DataFrame(columns=['filename', 'label', 'prediction'])

# for img in normal_list:
#     img_path = f'../images/combined/val/NORMAL/normal/{img}'
#     predicted_class = predict_image(img_path, model)
#     df_results = pd.concat([df_results, pd.DataFrame({'filename': img, 'label': 'NORMAL', 'prediction': predicted_class},
#                                                      index=[0])], ignore_index=True)
#
# for img in pneumo_list:
#     img_path = f'../images/combined/val/PNEUMONIA/pneumonia/{img}'
#     predicted_class = predict_image(img_path, model)
#     df_results = pd.concat([df_results, pd.DataFrame({'filename': img, 'label': 'PNEUMONIA', 'prediction': predicted_class},
#                                                      index=[0])], ignore_index=True)
# Function to load image into numpy array


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


def show_df_row(row):
    image_path = row["filepath"]
    image = Image.open(image_path).resize((image_width, image_height))
    img = load_image_into_numpy_array(image)
    exp = np.true_divide(img, 255.0)
    expanded = np.expand_dims(exp, axis=0)
    pred = model.predict(expanded)[0][0]
    guess = "neg"
    if pred > 0.5:
        guess = "pos"
    title = "Image: " + row["filename"] + " Label: " + row["label"] + " Guess: " + guess + " Score: " + str(pred)
    plt.title(title)
    plt.imshow(img)
    plt.show()
    return


results = []
counter = 0
for image in os.listdir(test_neg_dir):
    filename = test_neg_dir + "/" + image
    confidence = predict_image(filename)
    guess = 'pos' if confidence > 0.5 else 'neg'
    results.append([filename, image, "neg", guess, confidence])
    counter += 1
    print(counter) if counter%500 == 0 else None

for image in os.listdir(test_pos_dir):
    filename = test_pos_dir + "/" + image
    confidence = predict_image(filename)
    guess = 'pos' if confidence > 0.5 else 'neg'
    results.append([filename, image, "pos", guess, confidence])
    counter += 1
    print(counter) if counter % 500 == 0 else None

sorted_results = sorted(results, key=lambda x: x[4], reverse=True)
df = pd.DataFrame(data=sorted_results, columns=["filepath", "filename", "label", "guess", "confidence"])

print("Done inference!")

################################################################

print(df[::5][['filename', 'label',"guess","confidence"]])

################################################################

from matplotlib.ticker import FormatStrFormatter

pos = df.loc[df['label'] == "PNEUMONIA"]["confidence"]
neg = df.loc[df['label'] == "NORMAL"]["confidence"]
fig, ax = plt.subplots()
n, bins, patches = plt.hist([pos, neg], np.arange(0.0, 1.1, 0.1).tolist(),
                            edgecolor='black', linewidth=0.5, density=False,
                            histtype='bar', stacked=True, color=['green', 'red'],
                            label=['PNEUMONIA', 'NORMAL'])
plt.xlabel('Confidence')
plt.ylabel('N')
plt.xticks(bins)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
plt.title('Confidence scores for different values')
plt.legend(loc="lower right", fontsize=16)
plt.show()


################################################################

def create_with_cutoff(cutoff):
    __, ax = plt.subplots()
    TP = df.loc[(df['label'] == "pos") & (df["confidence"] > cutoff)]["confidence"]
    FP = df.loc[(df['label'] == "neg") & (df["confidence"] > cutoff)]["confidence"]
    FN = df.loc[(df['label'] == "pos") & (df["confidence"] < cutoff)]["confidence"]
    TN = df.loc[(df['label'] == "neg") & (df["confidence"] < cutoff)]["confidence"]
    plt.hist([TP, FP, TN, FN], np.arange(0.0, 1.1, 0.1).tolist(),
             edgecolor='black', linewidth=0.5, density=False, histtype='bar',
             stacked=True, color=['limegreen', 'forestgreen', 'orangered', 'salmon'],
             label=['TP', 'FP', 'TN', 'FN'])
    plt.xlabel('Confidence')
    plt.ylabel('N')
    plt.xticks(bins)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.title('Confidence scores for different values')
    plt.axvline(cutoff, color='k', linestyle='dashed', linewidth=2)
    plt.legend(loc="lower right", fontsize=16)
    sens = round(len(TP) / (len(TP) + len(FN)), 2)
    spec = round(len(TN) / (len(TN) + len(FP)), 2)
    stats = "sensitivity: " + str(sens) + "\n" + "specificity: " + str(spec) + "\n\n" + "TP: " + str(
        len(TP)) + "\n" + "FP: " + str(len(FP)) + "\n" + "TN: " + str(len(TN)) + "\n" + "FN: " + str(len(FN))
    plt.text(0.05, 0.05, stats, fontsize=14, transform=ax.transAxes)
    plt.show()


create_with_cutoff(cutoff)


################################################################
def create_auc_curve(classifications):
    squares = {}
    for x in classifications:
        conf = x[4]
        TP, FP, TN, FN = 0, 0, 0, 0
        for row in classifications:
            assert (row[2] == "neg" or row[2] == "pos")
            if row[2] == "neg":
                if float(row[4]) < conf:
                    TN += 1
                else:
                    FP += 1
            else:
                if float(row[4]) > conf:
                    TP += 1
                else:
                    FN += 1
        squares[conf] = [TP, FP, TN, FN]

    sens_spec = {}
    for entry in squares:
        sens = squares[entry][0] / float(squares[entry][0] + squares[entry][3])
        spec = squares[entry][2] / float(squares[entry][2] + squares[entry][1])
        sens_spec[entry] = (1 - spec, sens)
    return squares, sens_spec


squares, sens_spec = create_auc_curve(sorted_results)

x = []
y = []
for point in sens_spec.keys():
    x.append(sens_spec[point][0])
    y.append(sens_spec[point][1])

auc = auc(x, y)

plt.figure()
lw = 2
plt.plot(x, y, color='darkorange', lw=lw, label='ROC curve (area = %0.3f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Sensitivity')
plt.xlabel('1-specificity')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right", fontsize=20)
plt.show()

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
