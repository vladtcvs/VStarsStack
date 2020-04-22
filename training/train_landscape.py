import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
from skimage.color import rgb2gray
import tensorflow as tf
import cv2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

import json
import random
import sys

import common

from skimage import data, img_as_float
from skimage import exposure

size = 28
epochs = 32

def load_data(data_directory):
	skys = common.listfiles(data_directory + "/sky", ".png")
	lands = common.listfiles(data_directory + "/landscape", ".png")
	sky = []
	land = []
	for _, filename in skys:
		image = imageio.imread(filename)
		image = common.prepare_image_for_model(image)
		sky.append(image)

	for _, filename in lands:
		image = imageio.imread(filename)
		image = common.prepare_image_for_model(image)
		land.append(image)

	return sky, land

def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf

def display_images(images, labels, sample):
	ns = len(sample)
	fig = plt.figure(figsize=(8, 5))
	axes = np.zeros((2, ns), dtype=np.object)

	axes[0, 0] = plt.subplot(2, ns, 1)
	axes[1, 0] = plt.subplot(2, ns, ns+1)
	for i in range(1, ns):
		axes[0, i] = plt.subplot(2, ns, 1+i, sharex=axes[0, 0], sharey=axes[0, 0])
		axes[1, i] = plt.subplot(2, ns, ns+1+i)

	for i in range(ns):
		index = sample[i]
		img = images[index]
		label = labels[index]
		ax_img, ax_hist, ax_cdf = plot_img_and_hist(img, axes[:, i])
		ax_img.set_title("%i: %i" % (index, label))

	fig.tight_layout()
	plt.show()


ROOT_PATH = "landscape_model/"
train_data_directory = os.path.join(ROOT_PATH, "train")
test_data_directory = os.path.join(ROOT_PATH, "test")

train_sky, train_land = load_data(train_data_directory)
train_images = np.array(train_sky + train_land)

train_sky_labels = np.zeros(len(train_sky)) + 1
train_land_labels = np.zeros(len(train_land))
train_labels = np.append(train_sky_labels, train_land_labels)


# display histograms
#ns = 8
#sample = random.sample(range(len(train_images)), ns)
#display_images(train_images, train_labels, sample)
#sys.exit()

train_images = np.reshape(train_images, (len(train_images), train_images.shape[1], train_images.shape[2], 1))

inputs = Input(shape = (28,28,1))

conv1 = Conv2D(64, (7,7), padding="valid", activation="relu")(inputs)
pool1 = MaxPool2D(pool_size=(2, 2), padding='valid')(conv1)
norm1 = BatchNormalization()(pool1)

conv2 = Conv2D(32, (3,3), padding="same", activation="relu")(norm1)
pool2 = MaxPool2D(pool_size=(2, 2), padding='same')(conv2)
norm2 = BatchNormalization()(pool2)

conv3 = Conv2D(16, (3,3), padding="same", activation="relu")(norm2)
pool3 = MaxPool2D(pool_size=(2, 2), padding='same')(conv3)
norm3 = BatchNormalization()(pool3)

flat = Flatten()(norm3)
dense1 = Dense(16, activation="relu")(flat)
dense2 = Dense(2, activation="relu")(dense1)

predictions = Dense(2, activation='softmax')(dense2)


#model = tf.keras.Sequential([
#		Conv2D(64, (7,7), padding="valid", activation="relu"),
#		MaxPool2D(pool_size=(2, 2), padding='valid'),
#		BatchNormalization(),

#		Conv2D(32, (3,3), padding="valid", activation="relu"),
#		MaxPool2D(pool_size=(2, 2), padding='valid'),
#		BatchNormalization(),

#		Conv2D(16, (3,3), padding="valid", activation="relu"),
#		MaxPool2D(pool_size=(2, 2), padding='valid'),
#		BatchNormalization(),

#		Flatten(),
#		Dense(16, activation='relu'),
#		Dense(2, activation='relu'),
#	])

model = Model(inputs=inputs, outputs=predictions)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=epochs)

model.summary()

test_sky, test_land = load_data(test_data_directory)
test_images = np.array(test_sky + test_land)

test_sky_labels = np.zeros(len(test_sky)) + 1
test_land_labels = np.zeros(len(test_land))
test_labels = np.append(test_sky_labels, test_land_labels)

test_images = np.reshape(test_images, (len(test_images), test_images.shape[1], test_images.shape[2], 1))

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Accuracy: {:.3f}".format(test_acc))

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
test_predict = probability_model.predict(test_images)

test_predict = [np.argmax(test_predict[i]) for i in range(len(test_predict))]


#print("Truth", test_labels)
#print("Predict", test_predict)

tp = 0.0
tn = 0.0
fp = 0.0
fn = 0.0
wrong = []
for i in range(len(test_labels)):
	if test_labels[i] == 0 and test_predict[i] == 0:
		tn += 1
	elif test_labels[i] == 1 and test_predict[i] == 1:
		tp += 1
	elif test_labels[i] == 1 and test_predict[i] == 0:
		fn += 1
	elif test_labels[i] == 0 and test_predict[i] == 1:
		fp += 1

print("TP: %i, FP: %i, TN: %i, FN: %i" % (tp, fp, tn, fn))

with open(ROOT_PATH + "/model.json", "w") as f:
	f.write(probability_model.to_json())
probability_model.save_weights(ROOT_PATH + "/model.weights")

