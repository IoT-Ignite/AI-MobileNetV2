import tensorflow as tf


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.activations import relu
from tensorflow.keras.models import Model
class MobileNetV2:
	def __init__(self, input_shape, output, optimizer, loss, metrics):
		self.input_shape=input_shape;
		self.num_classes=output
		self.optimizer=optimizer
		self.loss=loss
		self.metrics=metrics

	
	'''
		inputs -> input
		filters -> the output 
		kernel -> always (3, 3)
		t -> expansion factor t
		stride -> stride number
		n-> layer repeat times

		Returns-> the output tensor

	'''
	def block(self, inputs, filters, kernel, t, stride, n):

		#x= self.bottleneck(inputs, filters, kernel, stride, t)
		x= self.bottleneck(inputs, filters, kernel,stride, t)
		for i in range(1, n):
			x= self.bottleneck(x, filters, kernel,1, t, True)

		return x
	def bottleneck(self, inputs, filters, kernels, stride, t, addStatus=False):
		#inputs = K.placeholder(shape=(2, 4, 5))
    	#print(K.int_shape(inputs)[-1])-> output 5
		new_channel = tf.keras.backend.int_shape(inputs)[-1]*t
		cchannel = int(filters * 1.0)
		x= self.conv2D(inputs, new_channel, (1, 1), (1,1))
		x= layers.DepthwiseConv2D(kernels, strides=(stride, stride), depth_multiplier=1, padding='same')(x)
		x= layers.BatchNormalization()(x)
		x= layers.Activation(self.ReLU6)(x)

		x= layers.Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
		x= layers.BatchNormalization()(x)

		#if the stride is 1, add the input and the current model
		if(addStatus==True):
			print("yesss   >>", x.shape, inputs.shape)
			new= layers.Add()([x, inputs])
		return x
	




	def conv2D(self, inputs, filters, kernel, stride):
		x= layers.Conv2D(filters, kernel, padding='same', strides=stride)(inputs)
		x= layers.BatchNormalization()(x)
		x= layers.Activation(self.ReLU6)(x)
		return x

	def ReLU6(self, inp):
		#model.add(Activation('relu'))
		return relu(inp, max_value=6.0)

	def _make_divisible(self, v, divisor, min_value=None):
		if min_value is None:
			min_value = divisor
		new_v = max(min_value, int(v+divisor/2) // divisor * divisor)
		print(new_v)
		# Make sure that round down does not go down by more than 10%.
		if new_v < 0.9 * v:
			new_v += divisor
		return new_v
	def create_model(self):
		inputs = layers.Input(shape=self.input_shape)
		first_filters =self._make_divisible(32*1.0, 8)
		#We always use kernel size 3X 3 as is standardfor modern networks
		kernel=(3, 3)
		#add the first 2d conv. layer
		x = self.conv2D(inputs, first_filters, kernel, (2, 2))

		#Bottleneck
		x= self.block(x, 16, kernel, t=1, stride=1, n=1)
		x= self.block(x, 24, kernel, t=6, stride=2, n=2)
		x= self.block(x, 32, kernel, t=6, stride=2, n=3)
		x= self.block(x, 64, kernel, t=6, stride=2, n=4)
		x= self.block(x, 96, kernel, t=6, stride=1, n=3)
		x= self.block(x, 160, kernel, t=6, stride=2, n=3)
		x= self.block(x, 320, kernel, t=6, stride=1, n=1)

		x= self.conv2D(x, 1280, (1,1), stride=(1,1))
		x= layers.GlobalAveragePooling2D()(x)
		x= layers.Reshape((1, 1, 1280))(x)
		x= layers.Dropout(0.3, name='Dropout')(x)
		x= layers.Conv2D(self.num_classes, (1, 1), padding='same')(x)
		x = layers.Activation('sigmoid', name='sigmoid')(x)
		output = layers.Reshape((self.num_classes,))(x)

		model =Model(inputs, output)
		print(model.summary())
		model.compile(optimizer=self.optimizer, loss=tf.keras.losses.BinaryCrossentropy(),  metrics=self.metrics)
		
		return model
#myModel=MobileNetV2((224, 224, 3), 1, loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

#createdModel=myModel.create_model()
	


