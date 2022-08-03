
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

def concat_data(x, y, val): 
    return x[:val], y[:val]

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


def benchmark_model(a_model): 

    a_model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    startTime = time.perf_counter()
    hist = a_model.fit(X_train_scaled, y_train_encoded, epochs = 1)
    elapsed_time = time.perf_counter() - startTime
    #  accuracy = hist.history['accuracy'][-1]
    best_accuracy = hist.history['accuracy'][argmin(history.history['loss'])]
    return elapsed_time, best_accuracy


X_train_scaled, y_train_encoded = concat_data(*preprocess(), 100)
#  start = time.perf_counter()
#  elapsed = [start]

model_list = {
        "resnet101": tf.keras.applications.resnet.ResNet101(weights=None, input_shape=(32, 32, 3), classes=10)

        }

with tf.device('/CPU:0'):
    for model_name in model_list: 

        elapsed_time, best_accuracy = benchmark_model(model_list[model_name])
        print("model_name: ", model_name)
        print("elapsed_time: ", elapsed_time)
        print("best_accuracy: ", best_accuracy)
        print("---------------")

#  print("elapsed: ", elapsed)
#  for i in range(1, len(elapsed)): 
#      print("Elapsed %.3f seconds" % (elapsed[i] - elapsed[i - 1]))


