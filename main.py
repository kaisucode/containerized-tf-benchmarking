
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import time

def preprocess(): 
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    X_train_scaled = X_train/255
    X_test_scaled = X_test/255

    y_train_encoded = keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
    y_test_encoded = keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')
    return X_train_scaled, y_train_encoded

def concat_data(x, y, val): 
    return x[:val], y[:val]

def benchmark_model(a_model, args): 

    a_model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    startTime = time.perf_counter()
    hist = a_model.fit(X_train_scaled, y_train_encoded, epochs=args.num_epochs)
    elapsed_time = time.perf_counter() - startTime
    best_accuracy = hist.history['accuracy'][np.argmin(hist.history['loss'])]
    return elapsed_time, best_accuracy
    #  accuracy = hist.history['accuracy'][-1]
    #  print(hist.history)

def parseArguments(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_data", type=int, default=50)
    parser.add_argument("--is_gpu", action="store_true")
    args = parser.parse_args()
    return args


def log_info(model_name, elapsed_time, best_accuracy): 
    ret_str = [
            "model_name: {}\n".format(best_accuracy),
            "elapsed_time: {}\n".format(elapsed_time),
            "best_accuracy: {0:.2f}%\n".format(best_accuracy),
            "---------------\n"
            ]
    return "".join(ret_str)


if __name__ == "__main__": 

    logs = ""
    args = parseArguments()

    X_train_scaled, y_train_encoded = concat_data(*preprocess(), args.num_data)
    model_list = {
            "resnet101": tf.keras.applications.resnet.ResNet101(weights=None, input_shape=(32, 32, 3), classes=10),
            "resnet152": tf.keras.applications.resnet.ResNet152(weights=None, input_shape=(32, 32, 3), classes=10),
            "vgg16": tf.keras.applications.VGG16(weights=None, input_shape=(32, 32, 3), classes=10),
            "inceptionv3": tf.keras.applications.InceptionV3(weights=None, input_shape=(32, 32, 3), classes=10)
            }

    if args.is_gpu: 
        print("Benchmarking with GPU")
        with tf.device('/GPU:0'):
            for model_name in model_list: 
                elapsed_time, best_accuracy = benchmark_model(model_list[model_name], args)
                logs += log_info(model_name, elapsed_time, best_accuracy)
    else: 
        print("Benchmarking with CPU")
        with tf.device('/CPU:0'):
            for model_name in model_list: 
                elapsed_time, best_accuracy = benchmark_model(model_list[model_name], args)
                logs += log_info(model_name, elapsed_time, best_accuracy)

    print(logs)

