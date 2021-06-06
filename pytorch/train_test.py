import os
import glob
import time
import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from torchvision.utils import save_image

from model.models import Encoder, Decoder, StyleVector
from data.word_dataset import WordImageDataset

class Trainer:
	def __init__(self, dataset, model_dir=None):
		self.dataset = dataset
		self.model_dir = model_dir
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.style_vec = StyleVector(dataset.get_categories())
	
	def train(self, freeze_encoder=False, save_model_dir=None, save_img_path=None,
			  epochs=10, batch_size=4, log_step=100, sample_step=1000, model_save_epoch=5, **kwargs):
		# Data Loader
		data_loader = DataLoader(self.dataset, batch_size, **kwargs)
		
		# Create Models
		encoder = Encoder().to(self.device)
		decoder = Decoder().to(self.device)
		
		# Losses
		l1_criterion = nn.L1Loss(reduction='mean').to(self.device)
		mse_criterion = nn.MSELoss(reduction='mean').to(self.device)

		if freeze_encoder:
			params = list(decoder.parameters())
		else:
			params = list(encoder.parameters()) + list(decoder.parameters())
		optimizer = Adam(params, betas=(0.5, 0.999))
		
		l1_losses, const_losses = list(), list()
		step = 0
		for epoch in range(epochs):
			for i, data in enumerate(data_loader):
				source, real_target, style_idx = data[0].to(self.device), data[1].to(self.device), data[2]
				style_idx = style_idx.cpu().detach().numpy()

				encoded_source, encoding_layers = encoder(source)
				embedded = torch.cat((encoded_source, self.style_vec[style_idx]), 1)
				fake_target = decoder(embedded, encoding_layers)

				# Loss
				encoded_fake = encoder(fake_target)[0]
				const_loss = mse_criterion(encoded_source, encoded_fake)
				
				l1_loss = l1_criterion(real_target, fake_target)
				
				loss = l1_loss + const_loss

				encoder.zero_grad()
				decoder.zero_grad()
				loss.backward()
				optimizer.step()
				
				l1_losses.append(int(l1_loss.data))
				const_losses.append(int(const_loss.data))
				
				if (i + 1) % log_step == 0:
					time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
					log_format = 'Epoch [%d/%d], step [%d/%d], l1_loss: %.4f, c_loss: %.4f' % \
								 (epoch + 1, epochs, i + 1, len(data_loader), l1_loss.item(), const_loss.item())
					print(time_stamp, log_format)
				
				if (i + 1) % sample_step == 0 and save_img_path:
					with torch.no_grad():
						fixed_source = self.dataset[0][0].to(self.device)
						es, el = encoder(fixed_source)
						fixed_fake_image = decoder(torch.cat((es, self.style_vec[0]), 1), el)
						save_image(((fixed_fake_image + 1) / 2).clamp(0, 1), \
								   os.path.join(save_img_path, 'fake_samples-%d-%d.png' % \
															   (epoch + 1, i + 1)))
		losses = [l1_losses, const_losses]
		return losses






# ----------------------------------------------
# Test only
# ----------------------------------------------

def read_file(path):
	lines = ''
	with open(path, 'r') as f:
		while True:
			line = f.readline()
			if not line:
				break
			lines += line.replace(" ", "")
	return lines

def read_dir(path, extension):
	return glob.glob(os.path.join(path, '*.' + extension))

if __name__ == '__main__':
	source_text = read_file('./data/hangeul.txt')
	source_font = './data/fonts/source/source_font.ttf'
	target_font = read_dir('./data/fonts/target_test', 'ttf')

	epochs = 10
	batch_size = 20

	save_img_path = './model/test_result'
	print(torch.cuda.is_available())

	dataset = WordImageDataset(source_text, source_font, target_font)
	trainer = Trainer(dataset)
	trainer.train(epochs = epochs, batch_size=batch_size, save_img_path=save_img_path)
