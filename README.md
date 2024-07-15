# PneumoCheck 
Pneumonia Detection in Chest X-rays Using Convolutional Neural Networks


## Overview
This project aims to develop a convolutional neural network (CNN) to classify chest X-ray images into two categories: normal (no pneumonia) and pneumonia (infected). The project includes data preprocessing, model development and training, evaluation, and a web application for real-time prediction.

## Dataset
The dataset used is a combined datasets: the [Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and [covid-pneumonia-nomal-xray](https://www.kaggle.com/datasets/salonimate/covid-pneumonia-nomal-xray) from Kaggle. Number of images:
- **Training Data**: 3117 normal images and 3117 pneumonia images
- **Test Data**: 779 normal images and 779 pneumonia images

## Project Structure
- **notebooks/**: Contains Jupyter notebooks for data preprocessing, model development, and evaluation.
  - `01_DataWrangling_EDA.ipynb`
  - `02_Modeling.ipynb`
  - `02_Modeling_2.ipynb`
  - `02_Modeling_Final.ipynb`
  - `03_Testing.ipynb`
  - `pneumonia-inceptionv3.ipynb` - to be renamed to `04_Transfer_Learning.ipynb`
- **app/**: Contains the Streamlit web application.
  - `main.py`
- **models/**: The final model is 
	- The final model file is 1.89 GB, so it is not uploaded here.
 	- To build the model, you can run the __/scripts/build_model.py__ script. It will build a model from InceptionV3, add new layers and weights, and save the model to the /models directory. The same logic is implemented in the application (app/main.py), but the model is cached and not saved to the disk.
 	- It can also be downloaded from the [HuggingFace](https://huggingface.co/airenare/InceptionV3_Pneumonia_CNN_v1/blob/main/Pneumonia_ROC_0975_cutoff_024.keras) (1.89 GB).
- **scripts/**: Contains data loading/wrangling script `data_preprocessing.py` and other testing scripts.
- **example_images/**: Contains images for the app's showcase.

## Preprocessing and Modeling
The data preprocessing steps include: 
- Merging datasets
- Class balancing
- Data augmentation: 
	- scaling, rotation, width shift, height shift, shear, zoom

Multiple CNN architectures were explored, including baseline models, deeper architectures, and transfer learning with pre-trained models like InceptionV3, VGG16 and ResNet50.

## Evaluation
Models were evaluated using confusion matrices and metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. 
![CM](https://github.com/airenare/pneumonia_xray/blob/main/app/pictures/confusion_matrix.png?raw=true)

All possible classification cutoff points were assessed, and the best one was determined to be 0.24:

![Cutoff](https://github.com/airenare/pneumonia_xray/blob/main/app/pictures/performance_hist.png?raw=true)

The best-performing model achieved:
- **Accuracy**: 0.96
- **Precision**: 0.97
- **Recall**: 0.95
- **F1-score**: 0.96
- **ROC-AUC**: 0.984

![ROC](https://github.com/airenare/pneumonia_xray/blob/main/app/pictures/ROC_curve.png?raw=true)

## Web Application
A [Streamlit](https://streamlit.io/) web application is provided to demonstrate the model's capabilities. Users can upload chest X-ray images, and the app will predict whether the image shows normal lungs or pneumonia.
![App_screenshot](https://github.com/airenare/pneumonia_xray/blob/main/app/pictures/app_screenshot.png?raw=true)
### Running the App
1. Clone the repository:
   ```bash
   git clone https://github.com/airenare/pneumonia_xray.git
   cd pneumonia_xray
   ```
2. Install the required packages:
```
pip install -r requirements.txt
```
3. Run the Streamlit app
```
streamlit run app/main.py
```
