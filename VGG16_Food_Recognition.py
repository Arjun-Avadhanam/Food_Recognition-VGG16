from pathlib import Path
import os
import shutil
import numpy as np
from keras.utils import load_img
from keras.utils import img_to_array
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation,Dense,Dropout,Flatten,GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.models import Model
from keras.applications import VGG16,EfficientNetB0,ResNet50
from keras.utils import image_dataset_from_directory
from sklearn.model_selection import train_test_split
import pandas as pd
import streamlit as st



input_shape = (224, 224, 3)
num_classes = 8
batch_size = 10
num_epochs = 25
learning_rate = 0.0001

# Load the VGG16 model without the top classification layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)


# Add custom layers on top of the VGG16 base model
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

layers_to_freeze = 15
# Freeze the layers in the base model
for layer in base_model.layers[:layers_to_freeze]:
    layer.trainable = False

# Compile the model
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Set the directories for your dataset
train_dir = 'dataset2\\train'
validation_dir = 'dataset2\\val'
test_dir = "test_dataset"

# Generate data batches from the directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_generator = validation_datagen.flow_from_directory(
    test_dir,
    target_size= input_shape[:2],
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

print(model.summary())

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.n // batch_size,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.n // batch_size
)

model.save('vgg16_food_recognition_model.h5')

test_loss,accuracy = model.evaluate(test_generator)

prediction = model.predict(test_generator)

print(prediction)

