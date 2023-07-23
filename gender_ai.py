#!/usr/bin/env python
# coding: utf-8

# In[110]:


import os
import tensorflow as tf
import keras
from tensorflow import keras
from keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-white")
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os
import shutil
from keras.preprocessing import image
import cv2


# In[111]:


img_train_male = "dataset/Training/male"
img_train_female = "dataset/Training/female"
img_valid_male = "dataset/Validation/male"
img_valid_female = "dataset/Validation/female"
size = 64


# In[112]:


def preprocess_image(img_path):
    img = keras.utils.load_img(img_path, target_size=(size,size))
    tensor = keras.utils.img_to_array(img)
    tensor /= 255.0 
    return tensor


# In[113]:


categories = ["male", "female"]
nb_class = len(categories)
x = []
y = []

# variables to test the face with a mask
test_on_male_x = []
test_on_male_y = []

# variables to test the face with a mask
test_female_x = []
test_female_y = []


# In[114]:


a=0
# preprocess the images, each of which is a face with mask
for i in os.listdir(img_train_male):
    img_path = os.path.join(img_train_male, i)
    img_tensor = preprocess_image(img_path)
    x.append(img_tensor)
    y.append(0)
    if a < 10:
        print ("len\[x\]:  %d" % len(x) )
    a=a+1


# In[115]:


# preprocess the images, each of which is a face without mask
for i in os.listdir(img_train_female):
    img_path = os.path.join(img_train_female, i)
    img_tensor = preprocess_image(img_path)
    x.append(img_tensor)
    y.append(1)


# In[116]:


x = np.array(x)
y = np.array(y)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1)
Y_train = keras.utils.to_categorical(Y_train, 2)
Y_test = keras.utils.to_categorical(Y_test, 2)


# In[117]:


# make a deep learning model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), padding="same", activation="relu", input_shape=(size, size, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# start - add custum layers
model.add(Conv2D(64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# end - add custum layers
model.add(Flatten())
model.add(Dense(128, activation="linear"))
model.add(Dense(2, activation="softmax"))
print(model.summary())


# In[118]:


model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])


# In[119]:


early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.1)
loss, accuracy = model.evaluate(X_test, Y_test)
print("Test accuracy:", accuracy)


# In[120]:


# evaluate a deep learning model
prediction = model.predict(X_test)
loss, acc = model.evaluate(X_test, Y_test, verbose=2)


# In[121]:


# draw images with accuracies and labels
topCnt = 8 *10
if len(X_test) < topCnt:
   topCnt = len(X_test)
   
plt.figure(figsize=(15,15))
for idx in range(topCnt):
   plt.subplot(8, 10, idx+1)
   plt.xticks([])
   plt.yticks([])
   plt.grid(False)
   plt.imshow(X_test[idx], cmap=plt.cm.binary)
   
   if prediction[idx][0] > prediction[idx][1]:
       label = "Male " + str(int(prediction[idx][0] * 100)) + " %"
   else:
       label = "Female " + str(int(prediction[idx][1] * 100)) + " %"
   plt.xlabel(label)
plt.show()

