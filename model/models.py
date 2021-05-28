import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
	def __init__(self, img_dim=1, conv_dim=64):
		super(Encoder, self).__init__()
		self.conv1 = conv2d(img_dim, conv_dim, k_size=5, stride=2, pad=2, dilation=2)
		self.conv2 = conv2d(conv_dim, conv_dim*2, k_size=5, stride=2, pad=2, dilation=2)
		self.conv3 = conv2d(conv_dim*2, conv_dim*4, k_size=5, stride=2, pad=1, dilation=1)
		self.conv4 = conv2d(conv_dim*4, conv_dim*8)
		self.conv5 = conv2d(conv_dim*8, conv_dim*8)
		self.conv6 = conv2d(conv_dim*8, conv_dim*8)
		self.conv7 = conv2d(conv_dim*8, conv_dim*8)
		self.conv8 = conv2d(conv_dim*8, conv_dim*8)

		self.lrelu = nn.LeakyRelu(0.2)
		self.batchnorm = nn.BatchNorm2d(
	
	def forward(self, images):
		encode_layers = dict()

		e1 = F.batch_norm(F.relu(self.conv1(images)))
		encode_layers['e1'] = e1
		e2 = F.relu(self.conv2(e1))
		encode_layers['e2'] = e2
		e3 = F.relu(self.conv3(e2))
		encode_layers['e3'] = e3
		e4 = F.relu(self.conv4(e3))
		encode_layers['e4'] = e4
		e5 = F.relu(self.conv5(e4))
		encode_layers['e5'] = e5
		e6 = F.relu(self.conv6(e5))
		encode_layers['e6'] = e6
		e7 = F.relu(self.conv7(e6))
		encode_layers['e7'] = e7
		encoded_source = F.relu(self.conv8(e7))
		encode_layers['e8'] = encoded_source

		return encoded_source, encode_layers

class Decoder(nn.Module):
	def __init__(self, img_dim=1, embedded_dim=640, conv_dim=64):
		super(Decoder, self).__init__()
		self.deconv1 = deconv2d(embedded_dim, conv_dim*8)
		self.deconv2 = deconv2d(conv_dim*16, conv_dim*8, k_size=4)
		self.deconv3 = deconv2d(conv_dim*16, conv_dim*8, k_size=5, dilation=2)
		self.deconv4 = deconv2d(conv_dim*16, conv_dim*8, k_size=4, dilation=2, stride=2)
		self.deconv5 = deconv2d(conv_dim*16, conv_dim*4, k_size=4, dilation=2, stride=2)
		self.deconv6 = deconv2d(conv_dim*8, conv_dim*2, k_size=4, dilation=2, stride=2)
		self.deconv7 = deconv2d(conv_dim*4, conv_dim*1, k_size=4, dilation=2, stride=2)
		self.deconv8 = deconv2d(conv_dim*2, img_dim, k_size=4, dilation=2, stride=2)

		self.dropout = nn.Dropout(0.25)

	def forward(self, embedded, encode_layers):
		d1 = F.relu(self.deconv1(embedded))
		d1 = torch.cat((d1, encoded_layers['e7']), dim=1)
		d2 = F.relu(self.deconv2(d1))
		d2 = torch.cat((d2, encode_layers['e6']), dim=1)
		d3 = F.relu(self.deconv3(d2))
		d3 = torch.cat((d3, encode_layers['e5']), dim=1)
		d4 = F.relu(self.deconv4(d3))
		d4 = torch.cat((d4, encode_layers['e4']), dim=1)
		d5 = F.relu(self.deconv5(d4))
		d5 = torch.cat((d5, encode_layers['e3']), dim=1)
		d6 = F.relu(self.deconv6(d5))
		d6 = torch.cat((d6, encode_layers['e2']), dim=1)
		d7 = F.relu(self.deconv7(d6))
		d7 = torch.cat((d7, encode_layers['e1']), dim=1)
		fake_target = torch.tanh(self.deconv8(d7))
	
		return fake_target

class Generator(nn.Module):
	def __init__(self, img_dim=1, conv_dim=64, embedding_dim=640):
		self.encoder = Encoder(img_dim=img_dim, conv_dim=conv_dim)
		self.decoder = Decoder(img_dim=img_dim, conv_dim=conv_dim, embedding_dim=embedding_dim)
		self.style_vec = torch.normal(mean=0, std=0.01, size=(embedding_dim, 1, 1))
		
	def forward(self, image, style_vec=None):
		encoded_source, encode_layers = self.encoder(images)
		if not style_vec:
			style_vec = self.style_vec
		embeded = torch.cat((encoded_source, style_vec), dims=1)
		fake_target = self.decoder(embeded, encoded_layers)
		return fake_target, encoded_source, encode_layers, style_vec

class Discriminator(nn.Module):
	def __init__(self, category_num, img_dim=2, disc_dim=64):
		super(Discriminator, self).__init__()
		self.conv1 = conv2d(img_dim, disc_dim)
		self.conv2 = conv2d(disc_dim, disc_dim*2)
		self.conv3 = conv2d(disc_dim*2, disc_dim*4)
		self.conv4 = conv2d(disc_dim*4, disc_dim*8);
		self.fc1 = 
