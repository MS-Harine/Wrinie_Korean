import os, glob
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from .models import Generator, Discriminator
from ../data/word_dataset import WordImageDataset

class Trainer:
	def __init__(self, dataset, embedding_dir=None, model_dir=None):
		self.dataset = dataset
		self.model_dir = model_dir
		self.embedding_dir = embedding_dir
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

		self.embedding_list = glob.glob(os.path.join(embedding_dir, '*.style.pt'))
	
	def train(self, epochs=10, train_test_ratio=.7, restore_files=None,
			  test_step=100, **kwargs):
		# Data Loader
		train_size = int(train_test_ratio * len(self.dataset))
		test_size = len(self.dataset) - train_size

		train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])
		train_loader = DataLoader(train_dataset, **kwargs)
		test_loader = DataLoader(train_dataset, **kwargs)
		
		# Create Models
		generator = Generator().to(self.device)
		discriminator = Discriminator(category_num=self.dataset.get_categories()).to(self.device)
		
		if self.model_dir:
			# [generator_filename, discriminator_filename] in restore_files parameter
			generator_path, discriminator_path = restore_files
			generator.load_state_dict(torch.load(os.path.join(self.model_dir, generator_path)))
			discriminator.load_state_dict(torch.load(os.path.join(self.model_dir, discriminator_path)))
		
		# Losses
		l1_criterion = nn.L1loss(size_average=True).to(self.device)
		bce_criterion = nn.BCEWithLogitsLoss(size_average=True).to(self.device)
		mse_criterion = nn.MSELoss(size_average=True).to(self.device)
		
		# Optimizer
		g_optimizer = Adam(list(generator.encoder.parameters()) + list(generator.decoder.parameters()), 
						   betas=(0.5, 0.999))
		d_optimizer = Adam(discriminator.parameters(), betas=(0.5, 0.999))
		
		# Training
		l1_losses, const_losses, category_losses, d_losses, g_losses = list(), list(), list(), list(), list()
		step = 0
		for epoch in range(epochs):
			for i, data, label, style_idx in enumerate(train_loader):
				data, label = data.to(self.device), label.to(self.device)
				style = torch.load(self.embedding_list[style_idx]).to(self.device) if self.embedding_dir else None
				
				fake_target, encoded_source, _, style_vec = generator(data, style)
				
