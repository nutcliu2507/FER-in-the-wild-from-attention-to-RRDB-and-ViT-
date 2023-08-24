import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from math import ceil
from sklearn import model_selection
import pandas as pd
from tensorflow.keras.layers import *

def readRAFDB():
    label_path = r"C:\Users\Black\Desktop\research\program\AlexNet-master\RaFD\basic\EmoLabel\list_patition_label.txt"
    img_path = r"C:\Users\Black\Desktop\RaFD\basic\Image/aligned"

    x_train = []
    y_train = []
    x_test = []
    y_test = []
    with open(label_path) as f:
        for i in f:
            i = i.strip()
            name, cls = i.split(" ")
            a, b = name.split(".")
            a += "_aligned."
            name = a + b
            img = load_img(os.path.join(img_path, name))
            # use the gray
            img = img.convert("L")
            img = img_to_array(img)
            if i.split("_")[0] == "train":
                x_train.append(img)
                y_train.append(int(cls) - 1)
            else:
                x_test.append(img)
                y_test.append(int(cls) - 1)
    x_train = np.array(x_train).reshape(-1, 100, 100, 1)
    x_test = np.array(x_test).reshape(-1, 100, 100, 1)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, x_test, y_train, y_test

def readFERplus():
    data = pd.read_csv('fer2013.csv')
    labels = pd.read_csv('fer2013new.csv')

    orig_class_names = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt',
                        'unknown', 'NF']

    n_samples = len(data)
    w = 48
    h = 48

    y = np.array(labels[orig_class_names])
    X = np.zeros((n_samples, w, h, 1))
    for i in range(n_samples):
        X[i] = np.fromstring(data['pixels'][i], dtype=int, sep=' ').reshape((h, w, 1))

    # Using mask to remove unknown or NF images
    y_mask = y.argmax(axis=-1)
    mask = y_mask < orig_class_names.index('unknown')
    X = X[mask]
    y = y[mask]


    # Convert to probabilities between 0 and 1
    y = y[:, :-2] * 0.1

    # Add contempt to neutral and remove it
    y[:, 0] += y[:, 7]
    y = y[:, :7]

    # Normalize image vectors
    X = X / 255.0
    test_size = ceil(len(X) * 0.1)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=42)
    x_train, x_val, y_train, y_val = model_selection.train_test_split(x_train, y_train, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test, x_val, y_val




