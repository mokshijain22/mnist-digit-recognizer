import os
import cv2
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.utils import resample

# ---------------------------
# LOAD MNIST
# ---------------------------
(X_train, y_train), _ = mnist.load_data()

X_train = X_train / 255.0
X_train = X_train.reshape(-1,28,28,1)

# ---------------------------
# LOAD OPERATORS
# ---------------------------
def load_operators(path):
    images, labels = [], []

    mapping = {'+':10, '-':11, 'x':12, '/':13, '=':14}

    for op in mapping:
        folder = os.path.join(path, op)

        for file in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, file), 0)
            img = img / 255.0
            images.append(img.reshape(28,28,1))
            labels.append(mapping[op])

    return np.array(images), np.array(labels)


X_ops, y_ops = load_operators("../dataset/operators")

# ---------------------------
# COMBINE DATA
# ---------------------------
X_train = np.concatenate((X_train, X_ops))
y_train = np.concatenate((y_train, y_ops))


# ---------------------------
# BALANCE DATA
# ---------------------------
def balance_dataset(X, y):
    unique = np.unique(y)
    max_count = max([np.sum(y == i) for i in unique])

    X_bal, y_bal = [], []

    for i in unique:
        Xi = X[y == i]
        yi = y[y == i]

        Xi_res, yi_res = resample(Xi, yi,
                                 replace=True,
                                 n_samples=max_count,
                                 random_state=42)

        X_bal.append(Xi_res)
        y_bal.append(yi_res)

    return np.concatenate(X_bal), np.concatenate(y_bal)


X_train, y_train = balance_dataset(X_train, y_train)

# ---------------------------
# MODEL
# ---------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(15, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ---------------------------
# TRAIN
# ---------------------------
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# ---------------------------
# SAVE
# ---------------------------
model.save("../math_model.h5")

print("Model saved as math_model.h5")