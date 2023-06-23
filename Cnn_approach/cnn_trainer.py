import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras import optimizers

IMAGE_SIZE = (154, 102)
NUM_CLASSES = 15  # Number of user classes
NUM_CHANNELS = 1  # Number of color channels (1 for grayscale, 3 for RGB)
EPOCHS = 1  # Number of epochs to train
BATCH_SIZE = 80 # Batch size for training
LEARNING_RATE = 0.001  # Learning rate for optimizer

X = np.load('eye_data.npy')
y = np.load('eye_labels.npy')

# Convert X and y to numpy arrays
X = np.array(X)
y = np.array(y)

# Reshape X to match the input shape of CNN (batch_size, height, width, channels)
X = X.reshape(-1, IMAGE_SIZE[1], IMAGE_SIZE[0], NUM_CHANNELS)

# Convert y to one-hot encoded vectors
y = to_categorical(y, num_classes=NUM_CLASSES)

# Split X and y into training, validation and test sets (80%, 10%, 10%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# Define data generator
def data_generator(X, y, batch_size):
    num_samples = X.shape[0]
    while True:
        indices = np.random.choice(num_samples, size=batch_size, replace=False)
        X_batch = X[indices]
        y_batch = y[indices]
        yield X_batch, y_batch

# Create data generators
train_generator = data_generator(X_train, y_train, batch_size=BATCH_SIZE)


# Define CNN architecture using Keras Sequential model

model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu',input_shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], NUM_CHANNELS)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax')) 



# Compile the model
model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(learning_rate=LEARNING_RATE), metrics=["accuracy"])

# Train the model using the generators
model.fit_generator(train_generator, steps_per_epoch=len(X_train) // BATCH_SIZE, epochs=EPOCHS)

# Save the model
model.save('model_both.h5')
