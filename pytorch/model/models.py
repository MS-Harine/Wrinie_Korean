import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, dilation=1, 
		   ltrue=True, lvalue=0.2, btrue=True):
	layers = []
	if ltrue:
		layers.append(nn.LeakyReLU(lvalue))
	layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, 
							stride=stride, padding=padding, dilation=dilation))
	if btrue:
		layers.append(nn.BatchNorm2d(out_dim))
	return nn.Sequential(*layers)

def deconv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, dilation=1,
			 dtrue=False, btrue=True):
	layers = []
	layers.append(nn.LeakyReLU(0.2))
	layers.append(nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel_size,
									 stride=stride, padding=padding, dilation=dilation))
	if btrue:
		layers.append(nn.BatchNorm2d(out_dim))
	if dtrue:
		layers.append(nn.Dropout(0.5))
	return nn.Sequential(*layers)

class Encoder(nn.Module):
	def __init__(self, img_dim=1, conv_dim=64):
		super(Encoder, self).__init__()
		self.conv1 = conv2d(img_dim, conv_dim, ltrue=False, btrue=False)
		self.conv2 = conv2d(conv_dim, conv_dim*2)
		self.conv3 = conv2d(conv_dim*2, conv_dim*4)
		self.conv4 = conv2d(conv_dim*4, conv_dim*8)
		self.conv5 = conv2d(conv_dim*8, conv_dim*8)
		self.conv6 = conv2d(conv_dim*8, conv_dim*8)
		self.conv7 = conv2d(conv_dim*8, conv_dim*8)
		self.conv8 = conv2d(conv_dim*8, conv_dim*8)
	
	def forward(self, images):
		encode_layers = dict()

		e1 = self.conv1(images)
		encode_layers['e1'] = e1.clone()
		e2 = self.conv2(e1)
		encode_layers['e2'] = e2.clone()
		e3 = self.conv3(e2)
		encode_layers['e3'] = e3.clone()
		e4 = self.conv4(e3)
		encode_layers['e4'] = e4.clone()
		e5 = self.conv5(e4)
		encode_layers['e5'] = e5.clone()
		e6 = self.conv6(e5)
		encode_layers['e6'] = e6.clone()
		e7 = self.conv7(e6)
		encode_layers['e7'] = e7.clone()
		encoded_source = self.conv8(e7)
		encode_layers['e8'] = encoded_source.clone()

		return encoded_source, encode_layers

class Decoder(nn.Module):
	def __init__(self, img_dim=1, embedded_dim=640, conv_dim=64):
		super(Decoder, self).__init__()
		self.deconv1 = deconv2d(embedded_dim, conv_dim*8, dtrue=True)
		self.deconv2 = deconv2d(conv_dim*16, conv_dim*8, kernel_size=4, dtrue=True)
		self.deconv3 = deconv2d(conv_dim*16, conv_dim*8, kernel_size=4, dtrue=True)
		self.deconv4 = deconv2d(conv_dim*16, conv_dim*8, kernel_size=4)
		self.deconv5 = deconv2d(conv_dim*16, conv_dim*4, kernel_size=4)
		self.deconv6 = deconv2d(conv_dim*8, conv_dim*2, kernel_size=4)
		self.deconv7 = deconv2d(conv_dim*4, conv_dim*1, kernel_size=4)
		self.deconv8 = deconv2d(conv_dim*2, img_dim, btrue=False)

	def forward(self, embedded, encode_layers):
		d1 = self.deconv1(embedded)
		d1 = torch.cat((d1, encode_layers['e7']), dim=1)
		d2 = self.deconv2(d1)
		d2 = torch.cat((d2, encode_layers['e6']), dim=1)
		d3 = self.deconv3(d2)
		d3 = torch.cat((d3, encode_layers['e5']), dim=1)
		d4 = self.deconv4(d3)
		d4 = torch.cat((d4, encode_layers['e4']), dim=1)
		d5 = self.deconv5(d4)
		d5 = torch.cat((d5, encode_layers['e3']), dim=1)
		d6 = self.deconv6(d5)
		d6 = torch.cat((d6, encode_layers['e2']), dim=1)
		d7 = self.deconv7(d6)
		d7 = torch.cat((d7, encode_layers['e1']), dim=1)
		d8 = self.deconv8(d7)
		fake_target = torch.tanh(d8)
	
		return fake_target

def StyleVector(category_num, embedding_dim=128):
	return torch.normal(mean=0, std=0.01, size=(category_num, 128, 1, 1))

def Generator(encoder, decoder, image, style_vec, font_idx):
	encoded_source, encode_layers = encoder(image)
	embeded = torch.cat((encoded_source, style_vec[font_idx]), 1)
	fake_target = decoder(embeded, encode_layers)
	return fake_target, encoded_source, encode_layers

"""
class Generator(nn.Module):
	def __init__(self, category_nums, img_dim=1, conv_dim=64, embedding_dim=640):
		super(Generator, self).__init__()
		self.encoder = Encoder(img_dim=img_dim, conv_dim=conv_dim)
		self.decoder = Decoder(img_dim=img_dim, conv_dim=conv_dim, embedded_dim=embedding_dim)
		self.style_vec = torch.normal(mean=0, std=0.01, size=(category_nums, 128, 1, 1))
		
	def forward(self, image, font_idx, style_vec=None):
		encoded_source, encode_layers = self.encoder(image)
		if not style_vec:
			style_vec = self.style_vec
		embeded = torch.cat((encoded_source, style_vec[font_idx]), 1)
		fake_target = self.decoder(embeded, encode_layers)
		return fake_target, encoded_source, encode_layers, style_vec
"""

class Discriminator(nn.Module):
	def __init__(self, category_num, img_dim=2, disc_dim=64):
		super(Discriminator, self).__init__()
		self.conv1 = conv2d(img_dim, disc_dim, kernel_size=3, padding=1)
		self.conv2 = conv2d(disc_dim, disc_dim*2, kernel_size=3, padding=1)
		self.conv3 = conv2d(disc_dim*2, disc_dim*4, kernel_size=3, padding=1)
		self.conv4 = conv2d(disc_dim*4, disc_dim*8, kernel_size=3, padding=1)
		self.fc1 = nn.Linear(disc_dim*8*8*8, 1)
		self.fc2 = nn.Linear(disc_dim*8*8*8, category_num)
	
	def forward(self, images):
		batch_size = images.shape[0]
		h1 = self.conv1(images)
		h2 = self.conv2(h1)
		h3 = self.conv3(h2)
		h4 = self.conv4(h3)

		tf_loss_logit = self.fc1(h4.reshape(batch_size, -1))
		tf_loss = torch.sigmoid(tf_loss_logit)
		cat_loss = self.fc2(h4.reshape(batch_size, -1))

		return tf_loss, tf_loss_logit, cat_loss
