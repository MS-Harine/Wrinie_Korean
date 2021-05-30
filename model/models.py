import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
	def __init__(self, img_dim=1, conv_dim=64):
		super(Encoder, self).__init__()
		self.conv1 = nn.conv2d(img_dim, conv_dim, kernel_size=5, stride=2, padding=2, dilation=2)
		self.conv2 = nn.conv2d(conv_dim, conv_dim*2, kernel_size=5, stride=2, padding=2, dilation=2)
		self.conv3 = nn.conv2d(conv_dim*2, conv_dim*4, kernel_size=5, stride=2, padding=1, dilation=1)
		self.conv4 = nn.conv2d(conv_dim*4, conv_dim*8, kernel_size=3, padding=1)
		self.conv5 = nn.conv2d(conv_dim*8, conv_dim*8, kernel_size=3, padding=1)
		self.conv6 = nn.conv2d(conv_dim*8, conv_dim*8, kernel_size=3, padding=1)
		self.conv7 = nn.conv2d(conv_dim*8, conv_dim*8, kernel_size=3, padding=1)
		self.conv8 = nn.conv2d(conv_dim*8, conv_dim*8, kernel_size=3, padding=1)
	
	def forward(self, images):
		encode_layers = dict()

		e1 = self.conv1(images)
		encode_layers['e1'] = e1
		e2 = F.batch_norm(self.conv2(F.leaky_relu(e1, 0.2)))
		encode_layers['e2'] = e2
		e3 = F.batch_norm(self.conv3(F.leaky_relu(e2, 0.2)))
		encode_layers['e3'] = e3
		e4 = F.batch_norm(self.conv4(F.leaky_relu(e3, 0.2)))
		encode_layers['e4'] = e4
		e5 = F.batch_norm(self.conv5(F.leaky_relu(e4, 0.2)))
		encode_layers['e5'] = e5
		e6 = F.batch_norm(self.conv6(F.leaky_relu(e5, 0.2)))
		encode_layers['e6'] = e6
		e7 = F.batch_norm(self.conv7(F.leaky_relu(e6, 0.2)))
		encode_layers['e7'] = e7
		encoded_source = F.batch_norm(self.conv8(F.leaky_relu(e7, 0.2)))
		encode_layers['e8'] = encoded_source

		return encoded_source, encode_layers

class Decoder(nn.Module):
	def __init__(self, img_dim=1, embedded_dim=640, conv_dim=64):
		super(Decoder, self).__init__()
		self.deconv1 = nn.ConvTranspose2d(embedded_dim, conv_dim*8, kernel_size=3, padding=1)
		self.deconv2 = nn.ConvTranspose2d(conv_dim*16, conv_dim*8, kernel_size=4, padding=1)
		self.deconv3 = nn.ConvTranspose2d(conv_dim*16, conv_dim*8, kernel_size=5, padding=1, dilation=2)
		self.deconv4 = nn.ConvTranspose2d(conv_dim*16, conv_dim*8, kernel_size=4, padding=1, dilation=2)
		self.deconv5 = nn.ConvTranspose2d(conv_dim*16, conv_dim*4, kernel_size=4, padding=1, dilation=2, stride=2)
		self.deconv6 = nn.ConvTranspose2d(conv_dim*8, conv_dim*2, kernel_size=4, padding=1, dilation=2, stride=2)
		self.deconv7 = nn.ConvTranspose2d(conv_dim*4, conv_dim*1, kernel_size=4, padding=1, dilation=2, stride=2)
		self.deconv8 = nn.ConvTranspose2d(conv_dim*2, img_dim, kernel_size=4, padding=1, dilation=2, stride=2)

	def forward(self, embedded, encode_layers):
		d1 = F.dropout(F.batch_norm(self.deconv1(F.leaky_relu(embedded, 0.2))))
		d1 = torch.cat((d1, encoded_layers['e7']), dim=1)
		d2 = F.dropout(F.batch_norm(self.deconv2(F.leaky_relu(d1, 0.2))))
		d2 = torch.cat((d2, encode_layers['e6']), dim=1)
		d3 = F.dropout(F.batch_norm(self.deconv3(F.leaky_relu(d2, 0.2))))
		d3 = torch.cat((d3, encode_layers['e5']), dim=1)
		d4 = F.batch_norm(self.deconv4(F.leaky_relu(d3, 0.2)))
		d4 = torch.cat((d4, encode_layers['e4']), dim=1)
		d5 = F.batch_norm(self.deconv5(F.leaky_relu(d4, 0.2)))
		d5 = torch.cat((d5, encode_layers['e3']), dim=1)
		d6 = F.batch_norm(self.deconv6(F.leaky_relu(d5, 0.2)))
		d6 = torch.cat((d6, encode_layers['e2']), dim=1)
		d7 = F.batch_norm(self.deconv7(F.leaky_relu(d6, 0.2)))
		d7 = torch.cat((d7, encode_layers['e1']), dim=1)
		d8 = F.batch_norm(self.deconv8(F.leaky_relu(d7, 0.2)))
		fake_target = torch.tanh(d8)
	
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
		self.conv1 = nn.conv2d(img_dim, disc_dim, kernel_size=3, padding=1)
		self.conv2 = nn.conv2d(disc_dim, disc_dim*2, kernel_size=3, padding=1)
		self.conv3 = nn.conv2d(disc_dim*2, disc_dim*4, kernel_size=3, padding=1)
		self.conv4 = nn.conv2d(disc_dim*4, disc_dim*8, kernel_size=3, padding=1)
		self.fc1 = nn.Linear(disc_dim*8*8*8, 1)
		self.fc2 = nn.Linear(disc_dim*8*8*8, category_num)
	
	def forward(self, images):
		batch_size = images.shape[0]
		h1 = self.conv1(F.leaky_relu(images, 0.2))
		h2 = F.batch_norm(self.conv2(F.leaky_relu(h1, 0.2)))
		h3 = F.batch_norm(self.conv3(F.leaky_relu(h2, 0.2)))
		h4 = F.batch_norm(self.conv4(F.leaky_relu(h3, 0.2)))

		tf_loss_logit = self.fc1(h4.reshape(batch_size, -1))
		tf_loss = torch.sigmoid(tf_loss_logit)
		cat_loss = self.fc2(h4.reshape(batch_size, -1))

		return tf_loss, tf_loss_logit, cat_loss
