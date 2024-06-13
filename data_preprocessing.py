# data_preprocessing.py
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_data(dataset_path):
    data = []
    labels = []
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.BMP'):
                    image_path = os.path.join(folder_path, file)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    image = cv2.resize(image, (96, 96))  # Resize images to 96x96 pixels
                    data.append(image)
                    labels.append(folder)
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

def preprocess_data(data, labels):
    # Normalize the data
    data = data / 255.0
    data = np.expand_dims(data, axis=-1)  # Add channel dimension

    # Encode the labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)

    return data, labels_categorical, label_encoder

def split_data(data, labels, test_size=0.2):
    return train_test_split(data, labels, test_size=test_size, random_state=42)
