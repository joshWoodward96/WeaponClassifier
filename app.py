import os
import cv2
import imghdr
import tensorflow as tf
from keras.utils import plot_model
from keras.metrics import Precision, Recall, BinaryAccuracy
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import time
import numpy as np
    
# Tensorflow Version
print(tf.__version__)

# DATA CLEANSING

# Dataset Directory
input_data = 'dataset'

#Image Extentions List
exts = ['jpeg', 'jpg']

# Cleanse Dataset by Removing Images With Invalid Extensions
for imgClass in os.listdir(input_data):
    if imgClass != '.DS_Store':
        for image in os.listdir(os.path.join(input_data, imgClass)):
            path = os.path.join(input_data, imgClass, image)
            try:
                img = cv2.imread(path)
                tip = imghdr.what(path)
                if tip not in exts:
                    print('Invalid Extension {}'.format(path))
                    os.remove(path)
            except Exception as e:
                    print('Error with {}'.format(path))


# Assign Weapons Dataset
data = tf.keras.utils.image_dataset_from_directory(input_data)

# Data Augmentation Function
def augmentation(image):
  image = tf.image.random_brightness(image, 0.2)
  image = tf.image.random_contrast(image, 0.5, 2.0)
  image = tf.image.random_saturation(image, 0.75, 1.25)
  image = tf.image.random_hue(image, 0.1)
  image = tf.image.random_flip_left_right(image)
  return image


# An iterator that converts elements to numpy
data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])  
        # 1 = Random Images
        # 0 = Knife

# PREPROCESSING
data = data.map(lambda x,y: (x/255, y))
data = data.map(lambda x,y: (augmentation(x), y))
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()

# SPLIT DATA
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)



print("Building CNN")

# Initialise CNN
cnn = tf.keras.models.Sequential()

# First Convolutional Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[256,256,3]))

# Max Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Second Convolutional Layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))

# Max Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Output Layer
cnn.add(tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01), activation='linear'))

cnn.compile(optimizer = 'adam', loss = 'hinge', metrics = ['accuracy'])

cnn.summary()


# Display CNN Architecture
plot_model(cnn, show_shapes=True)

logdir='logs'

# Tensorboard
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Train the Model
history = cnn.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

# Display Accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = cnn.evaluate(test, verbose=2)

# Display Training Loss
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()


# Metrics
pre = Precision()
rec = Recall()
acc = BinaryAccuracy()

# Prediction with validation data
start = time.time()
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = cnn.predict(X)
    yhat[yhat<0] = 0
    yhat[yhat>0] = 1
            
    pre.update_state(y,yhat)
    rec.update_state(y,yhat)
    acc.update_state(y,yhat)
end = time.time()
# F1
f1 = f1_score(y, yhat, average="binary")

# Confusion Matrix
cm = confusion_matrix(y,yhat)

cm_display = ConfusionMatrixDisplay(cm, display_labels=['Knife', 'Random Images'])

cm_display.plot()

cm_display.ax_.set(
    title='Confusion Matrix',
    xlabel = 'Predicted',
    ylabel='Actual'
)


# Print Metrics
print(f'Precision:{pre.result().numpy()}, Recall:{rec.result().numpy()}, Accuracy:{acc.result().numpy()}, F1: {f1}')
print(f'Prediction time per image: {(end - start)/23}')
# Display Confusion Matrix
plt.show()