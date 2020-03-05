from mobilenetv2 import MobileNetV2

import tensorflow as tf
def main():

	'''parser = argparse.ArgumentParser()
    # Required arguments.
    parser.add_argument(
        "--classes",
        help="The number of classes of dataset.")
    # Optional arguments.
    parser.add_argument(
        "--size",
        default=224,
        help="The image size of train sample.")
    parser.add_argument(
        "--epochs",
        default=300,
        help="The number of train iterations.")
    parser.add_argument("--checkpoint",
    	default ='train',
    	help="Give the checkpoint directory."
    	)
     args = parser.parse_args()


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
	myModel=MobileNetV2((224, 224, 3), 1, loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

	createdModel=myModel.create_model()

	print("MobileNetV2 summary")
	print(createdModel.summary())

if __name__ == '__main__':
	main()