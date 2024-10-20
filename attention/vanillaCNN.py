import tensorflow as tf
from tensorflow.keras import datasets, layers, models


import numpy as np

def load_data(data_byUser):
    # Load and preprocess CIFAR-10 dataset
    command = "datasets" + "." + str(data_byUser) + "." + "load_data()"
    (x_train, y_train), (x_test, y_test) = eval(command)

    print("load data success")

    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test

def architechture():
    # Build a simple CNN model without attention mechanism
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model 
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    print("model compile success")
    return model


x_train, y_train, x_test, y_test = load_data("cifar10")

# print(x_test.shape, y_test.shape)
# # Train the model
vanillaCNN = architechture()
history = vanillaCNN.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))
print(vanillaCNN.summary())
print(history)