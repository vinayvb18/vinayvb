#!/usr/bin/env python
# coding: utf-8

# #  Import the Required Libraries

# In[1]:


import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import cv2
import random
from tensorflow.keras.optimizers import Adam,SGD
from sklearn.metrics import confusion_matrix, classification_report


# # Loading the Image 

# In[2]:


data_path = r"C:\Users\ragha\AIML Python\reduced bird image data\train\Common-Kingfisher\Common-Kingfisher_765.jpg"


# In[3]:


image = cv2.imread(data_path) 
#cv2.imread() method loads an image from the path name


# In[4]:


cv2.imread(data_path)
#loads image into 3-D that is for RGB


# In[5]:


plt.imshow(image)


# In[6]:


image.shape
#(height of Image , Wiidth of Image  )


# # Training and Validation Datasets

# In[7]:


train = ImageDataGenerator(rescale=1/255)
valid = ImageDataGenerator(rescale=1/255)


# In[61]:


BATCH_SIZE = 34


# In[62]:


train_dataset = train.flow_from_directory(r'C:\Users\ragha\AIML Python\reduced bird image data\train',
                                          target_size=(256,256),
                                          color_mode='rgb',
                                          class_mode='categorical',
                                          batch_size=BATCH_SIZE,
                                          shuffle=True)


valid_dataset = valid.flow_from_directory(r'C:\Users\ragha\AIML Python\reduced bird image data\valid',
                                          target_size=(256,256),
                                          color_mode='rgb',
                                          class_mode='categorical',
                                           batch_size=BATCH_SIZE,
                                          shuffle=True)

"""
The ImageDataGenerator class is very useful in image classification. 
There are several ways to use this generator, depending on the method we use, 
here we will focus on flow_from_directory takes a path to the directory containing images sorted in sub directories and 
image augmentation parameters.

"""



# # Get the Label Mappings

# In[12]:


labels = {value: key for key, value in train_dataset.class_indices.items()}

print("Label Mappings for classes present in the training and validation datasets\n")
for key, value in labels.items():
    print(f"{key} : {value}")


# #  Plotting Sample Training Images

# In[13]:


fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(15, 12))
idx = 0

for i in range(2):
    for j in range(5):
        label = labels[np.argmax(train_dataset[0][1][idx])]
        ax[i, j].set_title(f"{label}")
        ax[i, j].imshow(train_dataset[0][0][idx][:, :, :])
        ax[i, j].axis("off")
        idx += 1

plt.tight_layout()
plt.suptitle("Sample Training Images", fontsize=21)
plt.show()


# # Training a CNN Model

# # We have a 256x256 RGB ( 3x3 kernel size) image.¶
# # We create 128 filters with 3x3 kernel size which shift 1 pixel at a time. The output has the same height'width dimension (i.e. 256x256) as the input and is activated by ReLU. Then, a maximum pooling with filters=(2,2) and stride=(2,2) is applied to a 256x256 output from previous Conv2D to downsample # 

# # Experiment 1 : STOCHASTIC GRADIENT DESCENT
# # Batch Size = 1
# # Model 1

# In[35]:


model = tf.keras.models.Sequential([                                    
                                    tf.keras.layers.Conv2D(128,(3,3), activation = 'relu',input_shape =(256,256,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.BatchNormalization(),
                                
                                    ##
                                    tf.keras.layers.Conv2D(64,(3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.BatchNormalization(),
                                    ##
                                    tf.keras.layers.Conv2D(32,(3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.BatchNormalization(),
                                    ##
                                    ##
                                    tf.keras.layers.Flatten(),
                                    ##
                                    tf.keras.layers.Dense(256, activation='relu'),
                                    tf.keras.layers.Dropout(0.5),    
                                    tf.keras.layers.Dense(8, activation='softmax')
                                  ])


# In[36]:


model.summary()


# In[37]:


optimizer = Adam(learning_rate=0.001)


# In[38]:


model.compile(loss = 'categorical_crossentropy',
              optimizer = optimizer,
              metrics = ['accuracy'])


# In[40]:


model_fit = model.fit(
    train_dataset,
    epochs = 10 ,
    batch_size = 1,
    validation_data=valid_dataset)


# In[41]:


model_fit.history


# # Plotting Training Accuracy  VS  Validatiion Accuracy

# In[24]:


def acc_curve(history):
    
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.title('Training and Validation Accuracy', fontsize=20)
    plt.xlabel('Epochs', fontsize = 15)
    plt.ylabel('Accuracy', fontsize = 15)
    plt.legend()
    plt.show()



# In[42]:


acc_curve(model_fit)


# # Plotting Training Loss  VS  Validatiion Loss

# In[25]:


def loss_curve(history):  
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Training and Validation Loss', fontsize=20)
    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend()
    plt.show()


# In[43]:


loss_curve(model_fit)


# # Experiment 2: Mini Batch Gradient Descent
# # batch_size = 34
# # Using some data (more than one sample but less than entire dataset)
# # Model 2

# In[63]:


model2 = tf.keras.models.Sequential([                                    
                                    tf.keras.layers.Conv2D(128,(3,3), activation = 'relu',input_shape =(256,256,3)),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.BatchNormalization(),
                                
                                    ##
                                    tf.keras.layers.Conv2D(64,(3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.BatchNormalization(),
                                    ##
                                    tf.keras.layers.Conv2D(32,(3, 3), activation='relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    tf.keras.layers.BatchNormalization(),
                                    ##
                                    ##
                                    tf.keras.layers.Flatten(),
                                    ##
                                    tf.keras.layers.Dense(256, activation='relu'),
                                    tf.keras.layers.Dropout(0.5),    
                                    tf.keras.layers.Dense(8, activation='softmax')
                                  ])


# In[64]:


model2.summary()


# In[65]:


optimizer = Adam(learning_rate=0.001)


# In[66]:


model2.compile(loss = 'categorical_crossentropy',
              optimizer = optimizer,
              metrics = ['accuracy'])


# In[67]:


model2_fit = model2.fit(
    train_dataset,
    epochs = 10,
    batch_size = 34 ,
    validation_data=valid_dataset)


# In[70]:


model2_fit.history


# In[71]:


acc_curve(model2_fit)


# In[72]:


loss_curve(model2_fit)


# # Testing the Model on Test Set

# In[73]:


test = r'C:\Users\ragha\AIML Python\reduced bird image data\valid'


# In[75]:


test_datagen = ImageDataGenerator(rescale=1.0/255)

test_dataset = test_datagen.flow_from_directory(test,
                                          target_size=(256,256),
                                          color_mode='rgb',
                                          class_mode='categorical',
                                          batch_size=50,
                                          shuffle=True)


# # Model Prediction on the Test Dataset

# In[96]:


predictions = model2.predict(test_dataset)


# In[98]:


fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12, 10))
idx = 0

for i in range(2):
    for j in range(5):
        predicted_label = labels[np.argmax(predictions[idx])]
        ax[i, j].set_title(f"{predicted_label}")
        ax[i, j].imshow(test_dataset[0][0][idx])
        ax[i, j].axis("off")
        idx += 1

plt.tight_layout()
plt.suptitle("Test Dataset Predictions", fontsize=20)
plt.show()


# In[80]:


test_loss, test_accuracy = model2.evaluate(test_dataset , batch_size = 200)


# In[81]:


print(f"Test Loss:     {test_loss}")
print(f"Test Accuracy: {test_accuracy}")


# # Plotting the Classification Metrics

# In[82]:


y_pred = np.argmax(predictions, axis=1)
y_true = test_dataset.classes


# In[83]:


import seaborn as sns


# In[84]:


cf_mtx = confusion_matrix(y_true, y_pred)

group_counts = ["{0:0.0f}".format(value) for value in cf_mtx.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_mtx.flatten()/np.sum(cf_mtx)]
box_labels = [f"{v1}\n({v2})" for v1, v2 in zip(group_counts, group_percentages)]
box_labels = np.asarray(box_labels).reshape(8, 8)

plt.figure(figsize = (12, 10))
sns.heatmap(cf_mtx, xticklabels=labels.values(), yticklabels=labels.values(),
           cmap="YlGnBu", fmt="", annot=box_labels)
plt.xlabel('Predicted Classes')
plt.ylabel('True Classes')
plt.show()


# # Wrong predictions

# In[85]:


errors = (y_true - y_pred != 0)
y_true_errors = y_true[errors]
y_pred_errors = y_pred[errors]


# In[86]:


test_images = test_dataset.filenames
test_img = np.asarray(test_images)[errors]


# In[92]:


fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12, 10))
idx = 0

for i in range(2):
    for j in range(5):
        idx = np.random.randint(0, len(test_img))
        true_index = y_true_errors[idx]
        true_label = labels[true_index]
        predicted_index = y_pred_errors[idx]
        predicted_label = labels[predicted_index]
        ax[i, j].set_title(f"True Label: {true_label} \n Predicted Label: {predicted_label}")
        img_path = os.path.join(test, test_img[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, j].imshow(img)
        ax[i, j].axis("off")

plt.tight_layout()
plt.suptitle('Wrong Predictions made on test set', fontsize=20)
plt.show()

