import os
import sys
import argparse
from mobilenetv2 import MobileNetV2
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
def main():

	parser = argparse.ArgumentParser()
	# Required arguments.
	parser.add_argument(
		"--classes",
		default=1,
		help="The number of classes of dataset.")
	# Optional arguments.
	parser.add_argument(
		"--size",
		default=224,
		help="The image size of train sample.")
	parser.add_argument("--checkpoint",
		default ='train',
		help="Give the checkpoint directory."
		)
	args = parser.parse_args()

	'''
	print("Creating a mobilenetv2 model...")
	earlystop = EarlyStopping(monitor='val_acc', patience=30, verbose=0, mode='auto')
	myModel=MobileNetV2((224, 224, 3), 1, loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

	createdModel=myModel.create_model()

	print("MobileNetV2 summary")
	print(createdModel.summary())


	#checkpoint
	checkpoint_path = args.checkpoint+"/cp-{epoch:05d}.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)

	# Create a callback that saves the model's weights
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
	save_weights_only=True,
	verbose=1,
	period=100)




	#save the model
	# Save the weights using the `checkpoint_path` format
	createdModel.save_weights(checkpoint_path.format(epoch=0))

	createdModel.fit(train_images, 
			train_labels,
			epochs=50, 
			callbacks=[cp_callback],
			validation_data=(test_images,test_labels),
			verbose=0)'''
	myModel=MobileNetV2((224, 224, 3), 1, loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

	createdModel=myModel.create_model()

	print("MobileNetV2 summary")
	print(createdModel.summary())
	#checkpoint
	checkpoint_path = args.checkpoint+"/cp-{epoch:05d}.ckpt"
	checkpoint_dir = os.path.dirname(checkpoint_path)

	# Create a callback that saves the model's weights
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
		save_weights_only=True,
		verbose=1,
		period=100)
	train_generator, validation_generator, count1, count2 = generate(32, 224)
	history = createdModel.fit_generator(
        train_generator,
        validation_data=validation_generator,
        steps_per_epoch=500,
        validation_steps=100,
        epochs=1000,
        callbacks=[cp_callback])

def generate(batch, size):
	"""Data generation and augmentation

	# Arguments
	batch: Integer, batch size.
	size: Integer, image size.

	# Returns
	train_generator: train set generator
	validation_generator: validation set generator
	count1: Integer, number of train set.
	count2: Integer, number of test set.
	"""

	#  Using the data Augmentation in traning data
	ptrain = 'WIDER_train/images'
	pval = 'WIDER_val/images'

	datagen1 = ImageDataGenerator(
			rescale=1. / 255,
			shear_range=0.2,
			zoom_range=0.2,
			rotation_range=90,
			width_shift_range=0.2,
			height_shift_range=0.2,
			horizontal_flip=True)

	datagen2 = ImageDataGenerator(rescale=1. / 255)

	train_generator = datagen1.flow_from_directory(
			ptrain,
			target_size=(size, size),
			batch_size=batch,
			class_mode='categorical')

	validation_generator = datagen2.flow_from_directory(
			pval,
			target_size=(size, size),
			batch_size=batch,
			class_mode='categorical')

	count1 = 0
	for root, dirs, files in os.walk(ptrain):
		for each in files:
			count1 += 1

	count2 = 0
	for root, dirs, files in os.walk(pval):
		for each in files:
			count2 += 1

	return train_generator, validation_generator, count1, count2
if __name__ == '__main__':
	main()