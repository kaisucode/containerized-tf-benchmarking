
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time

def get_compiled_model():
    # Make a simple 2-layer densely-connected neural network.
    inputs = keras.Input(shape=(784,))
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dense(256, activation="relu")(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def get_dataset(num_devices = 1):
    batch_size = 32 * num_devices
    num_val_samples = 10000

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Preprocess the data (these are Numpy arrays)
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Reserve num_val_samples samples for validation
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]
    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
    )


#  def preprocess(): 
#      (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

#      X_train_scaled = X_train/255
#      X_test_scaled = X_test/255

#      y_train_encoded = keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
#      y_test_encoded = keras.utils.to_categorical(y_test, num_classes=10, dtype='float32')
#      return X_train_scaled, y_train_encoded

#  def concat_data(x, y, val): 
#      return x[:val], y[:val]

#  def benchmark_model(a_model, args): 
#      BATCH_SIZE = 5000

#      startTime = time.perf_counter()

#      if args.has_two_gpu: 
#          hist = a_model.fit(X_train_scaled, y_train_encoded, epochs=args.num_epochs, BATCH_SIZE * 2)
#      else: 
#          hist = a_model.fit(X_train_scaled, y_train_encoded, epochs=args.num_epochs, BATCH_SIZE)

#      elapsed_time = time.perf_counter() - startTime
#      best_accuracy = hist.history['accuracy'][np.argmin(hist.history['loss'])]
#      return elapsed_time, best_accuracy

def parseArguments(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_data", type=int, default=50)
    parser.add_argument("--is_gpu", action="store_true")
    parser.add_argument("--has_two_gpu", action="store_true")
    args = parser.parse_args()
    return args


def log_info(model_name, elapsed_time, best_accuracy): 
    ret_str = [
            "model_name: {}\n".format(model_name),
            "elapsed_time: {}\n".format(elapsed_time),
            "best_accuracy: {:.2%}\n".format(best_accuracy),
            "---------------\n"
            ]
    return "".join(ret_str)


def load_models(): 
    model_list = {
            "resnet101": tf.keras.applications.resnet.ResNet101(weights=None, input_shape=(32, 32, 3), classes=10),
            "resnet152": tf.keras.applications.resnet.ResNet152(weights=None, input_shape=(32, 32, 3), classes=10),
            "vgg16": tf.keras.applications.VGG16(weights=None, input_shape=(32, 32, 3), classes=10),
            }

    for model_name in model_list: 
        model_list[model_name].compile(optimizer='SGD',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model_list


if __name__ == "__main__": 

    logs = ""
    args = parseArguments()


    if args.has_two_gpu: 
        print("Benchmarking with multiple GPUs")

        # Create a MirroredStrategy.
        #  mirrored_strategy = tf.distribute.MirroredStrategy()
        mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/GPU:0", "/GPU:1"])
        print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))
        num_devices = mirrored_strategy.num_replicas_in_sync

        compiled_model_list = {}
        # Open a strategy scope.
        with mirrored_strategy.scope():
            model_list = load_models()

        # Train the model on all available devices.
        train_dataset, val_dataset, test_dataset = get_dataset(num_devices)

        for model_name in model_list: 
            model = model_list[model_name]
            startTime = time.perf_counter()
            hist = model.fit(train_dataset, epochs=5, validation_data=val_dataset)
            elapsed_time = time.perf_counter() - startTime
            best_accuracy = hist.history['accuracy'][np.argmin(hist.history['loss'])]
            logs += log_info(model_name, elapsed_time, best_accuracy)
            print(logs)

    elif args.is_gpu: 
        print("Benchmarking with GPU")
        with tf.device('/GPU:0'):
            model_list = load_models()
            for model_name in model_list: 
                elapsed_time, best_accuracy = benchmark_model(model_list[model_name], args)
                logs += log_info(model_name, elapsed_time, best_accuracy)
                print(logs)

    print(logs)

