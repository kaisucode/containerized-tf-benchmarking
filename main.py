
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time


def preprocess(): 
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    X_train_scaled = X_train/255
    X_test_scaled = X_test/255

    y_train_encoded = keras.utils.to_categorical(y_train, num_classes = 10, dtype = 'float32')
    y_test_encoded = keras.utils.to_categorical(y_test, num_classes = 10, dtype = 'float32')
    return X_train_scaled, y_train_encoded


def get_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32,32,3)),
        keras.layers.Dense(3000, activation='relu'),
        keras.layers.Dense(1000, activation='relu'),
        keras.layers.Dense(10, activation='sigmoid')    
    ])
    model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model


X_train_scaled, y_train_encoded = preprocess()
X_train_scaled = X_train_scaled[:100]
y_train_encoded = y_train_encoded[:100]
start = time.perf_counter()
elapsed = [start]
with tf.device('/CPU:0'):
    model_cpu = get_model()
    model_resnet101 = tf.keras.applications.resnet.ResNet101(weights=None, input_shape=(32, 32, 3), classes=10)
    model_resnet101.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    model_cpu.fit(X_train_scaled, y_train_encoded, epochs = 1)
    elapsed.append(time.perf_counter())

    model_resnet101.fit(X_train_scaled, y_train_encoded, epochs = 1)
    elapsed.append(time.perf_counter())

print("elapsed")
print(elapsed)
for i in range(1, len(elapsed)): 
    print("Elapsed %.3f seconds" % (elapsed[i] - elapsed[i - 1]))


