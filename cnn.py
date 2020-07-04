# Convolutional Neural Network

# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection

#classifier.add(Dense(output_dim = 128, activation = 'relu'))
#instead of above statement, use the below ones
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 2,
                         validation_data = test_set,
                         validation_steps = 2000)#number of imgs in test set

#Save the model
classifier.save_weights('model_wieghts.hdf5',overwrite=True)#use this
classifier.save('model_keras.hdf5')
#Load weights
classifier.load_weights('model_wieghts.hdf5')#use this
#Load model
import tensorflow as tf
classifier=tf.keras.models.load_model("model_keras.hdf5")


test_loss, test_acc = classifier.evaluate(test_set, verbose=2)
print('\nTest accuracy:', test_acc)

#Predict new images
import cv2
import tensorflow as tf

CATEGORIES = ["Cat","Dog"]

def prepare(filepath):
    IMG_SIZE = 64
    img_array = cv2.imread(filepath,cv2.IMREAD_UNCHANGED)
    new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 3)



prediction = classifier.predict([prepare('abcd.jpg')])
print(CATEGORIES[ int( prediction[0][0] ) ] )





################################################

#Save/Load Model
#Save Classifier
from keras.models import model_from_json
# serialize model to JSON
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")
# Load Classifier
# load json and create model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into new model
classifier.load_weights("model.h5")
print("Loaded model from disk")



################################################

#prediction from udemy/git-code
# Making a new prediction

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/validation_set/cat.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
print(result)
training_set.class_indices
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'
  
print(prediction)


################################################