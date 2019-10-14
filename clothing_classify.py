#Fashion MNIST is a multiclass problem.  Hopefully you
#will be able to achieve at least 97% accuracy on unseen data

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape each image to be 28 x 28 x 1.
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
# Reshaping your images is often one of the most difficult
# aspects of machine learning with image data.

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# The current range of X_train and X_test is 0 to 255.
# The code below is equivalent to X_train = X_train / 255.
# This scales each value to be between 0 and 1.
X_train /= 255
X_test /= 255

#Lets compile the model
model.add(Dense(1, input_shape=(28, 28, 1), activation='linear'))
model.add(Flatten())
model.add(Dense(128, input_shape=(28, 28, 1), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit and train the models
model.fit(X_train, Y_train, batch_size=256, epochs=5, verbose=1)

# Evaluate model on test data.
score = model.evaluate(X_test, Y_test, verbose=0)
labels = model.metrics_names
# print(f'{labels[0]}: {score[0]}')
# print(f'{labels[1]}: {score[1]}')
