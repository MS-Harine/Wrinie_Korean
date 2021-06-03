from keras.models import Model
from keras.layers import Input, add, concatenate, Activation
from keras.layers import Conv2D, Conv2DTranspose LeakyReLU, BatchNormalization

def create_encoder(input_shape, conv_dim=64):
	encoding_layers = dict()
	inputs = Input(shape=input_shape)
	
	x = Conv2D(conv_dim)(inputs)
	encoding_layers['e1'] = x
	
	for idx, filters in enumerate([2, 4, 8, 8, 8, 8, 8]):
		x = LeakyReLU(0.2)(x)
		x = Conv2D(conv_dim * filters)(x)
		x = BatchNormalization()(x)

		encoding_layers['e' + str(idx + 2)] = x
	
	outputs = x
	model = Model(inputs, outputs)
	return model, encoding_layers

def create_decoder(input_shape, encoding_layers, conv_dim=64):
	inputs = Input(shape=input_shape)
	
	x = LeakyReLU(0.2)(inputs)
	x = Conv2DTranspose(conv_dim * 8)(x)
	x = BatchNormalization()(x)
	
	residual = encoding_layers['e8']
	x = add([x, residual])

	for idx, filters in enumerate([8, 8, 8, 8, 4, 2]):
		x = LeakyReLU(0.2)(x)
		x = Conv2DTranspose(conv_dim * filters)(x)
		x = BatchNormalization()(x)

		residual = encoding_layers['e' + str(7 - idx)]
		x = add([x, residual])
	
	x = LeakyReLU(0.2)(x)
	x = Conv2DTranspose(1)(x)
	outputs = Activation('tanh')(x)

	model = Model(inputs, outputs)
	return model

def create_generator(input_shape, style_vec, font_idx):
	encoder, encoding_layers = create_encoder(input_shape)
	embedded = concatenate([encoder, style_vec], axis=1)
	decoder = create_decoder(encoder.output_shape, encoding_layers)
	
	autoencoder = Model(encoder, decoder)
	return autoencoder

def create_discriminator(input_shape, category_num, disc_dim=64):
	inputs = Input(input_shape)
	
	x = LeakyReLU(0.2)(inputs)
	x = Conv2D(disc_dim)(x)
	x = BatchNormalization()(x)

	for filters in [2 4 8]:
		x = LeakyReLU(0.2)(x)
		x = Conv2D(disc_dim * filters)(x)
		x = BatchNormalization(x)

	x = Flatten()(x)
	tf_loss_logit = Dense(1, name='tf_output')(x)
	cat_loss = Dense(category_num, name='category_output')(x)

	model = Model(inputs=[inputs], outputs=[tf_loss_logit, cat_loss])
	return model
