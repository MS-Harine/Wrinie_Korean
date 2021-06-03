import os
import argparse
from glob import glob

from model.train import Trainer
from data.word_dataset import WordImageDataset

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
	return glob(os.path.join(path, '*.' + extension))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Wrinie model handler')
	parser.add_argument('-w', '--word_path', type=str, dest='word_path', required=True,
						help='Text to create font images')
	parser.add_argument('-s', '--source', type=str, dest='source', required=True,
						help='Input the .ttf file path as source')
	parser.add_argument('-t', '--target', type=str, dest='target', required=True,
						help='Input the directory where .ttf files as target are exist')
	parser.add_argument('-e', '--epochs', type=int, dest='epoch', default=10,
						help='Total training epoch')
	parser.add_argument('-b', '--batch_size', type=int, dest='batch_size', default=4,
						help='Training batch size')
	parser.add_argument('-m', '--save_dir', type=str, dest='save_path',
						help='Directory to save models')
	parser.add_argument('-i', '--image_dir', type=str, dest='image_dir',
						help='Directory to save images while training')
	parser.add_argument('-r', '--restore_dir', type=str, dest='restore_dir',
						help='Directory to restore models to training')
	parser.add_argument('-f', '--freeze_encoder', type=bool, dest='freeze_encoder', default=False,
						help='Freeze encoder while trining')
	parser.add_argument('-n', '--step', type=int, nargs='+', dest='steps', default=[100],
						help='Steps to print training informations, [logging_step, imagging_step, save_model_step]')
	args = parser.parse_args()

	# Parsing arguments
	source_text = read_file(args.word_path)
	source_font = args.source
	target_font = read_dir(args.target, 'ttf')
	
	epochs = args.epoch
	batch_size = args.batch_size
	freeze_encoder = args.freeze_encoder
	
	save_dir = args.save_path
	image_dir = args.image_dir
	restore_dir = read_dir(args.restore_dir, 'pkl') if args.restore_dir is not None else None
	
	log_step = args.steps[0]
	sample_step = args.steps[1] if len(args.steps) > 1 else None
	model_step = args.steps[2] if len(args.steps) > 2 else None
	
	
	# Create dataset
	dataset = WordImageDataset(source_text, source_font, target_font)
	trainer = Trainer(dataset)
	trainer.train(epochs = epochs,
				  batch_size = batch_size,
				  freeze_encoder = freeze_encoder,
				  save_model_dir = save_dir,
				  save_img_path = image_dir,
				  restore_files = restore_dir,
				  log_step = log_step,
				  sample_step = sample_step,
				  model_save_epoch = model_step)
