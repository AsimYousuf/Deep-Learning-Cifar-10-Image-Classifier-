import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

#Loading Cifar 10 Dataset
(training_images, training_labels), (testing_images, testing_labels) = cifar10.load_data()

#Normalizing data between 0 and 1
training_images, testing_images = training_images/255, testing_images/255

#Conversion of labels into one hot encoding
training_labels = to_categorical(training_labels, 10)
testing_labels = to_categorical(testing_labels, 10)

#Creating the neural network

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (2,2), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (2,2), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

#Compilation of model
model.compile(optimizer = 'adam',
              loss= 'categorical_crossentropy', metrics=['accuracy'])

#Training the model
model.fit(training_images, training_labels, epochs=30,
          validation_data=(testing_images, testing_labels))
