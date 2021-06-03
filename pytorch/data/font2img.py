import os
import argparse

from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np

def tight_crop_image(img):
	img_size = img.shape[0]
	width = np.where(img_size - np.sum(img, axis=0) > 1)
	height = np.where(img_size - np.sum(img, axis=1) > 1)
	y1, y2 = height[0][0], height[0][-1]
	x1, x2 = width[0][0], width[0][-1]
	cropped_image = img[y1:y2, x1:x2]
	
	return cropped_image

def add_padding(img, image_size=128):
	h, w = img.shape
	pad_value = 0
	
	pad_x_width = (image_size - w) // 2
	pad_x = np.full((h, pad_x_width), pad_value, dtype=np.float32)
	img = np.concatenate((pad_x, img), axis=1)
	img = np.concatenate((img, pad_x), axis=1)
	
	pad_y_height = (image_size - h) // 2
	pad_y = np.full((pad_y_height, w), pad_value, dtype=np.float32)
	img = np.concatenate((pad_y, img), axis=0)
	img = np.concatenate((img, pad_y), axis=0)

	return img

def draw_single_char(ch, font_path, canvas_size, show_img=False):
	image = Image.new('L', (canvas_size, canvas_size), color=255)
	drawing = ImageDraw.Draw(image)
	font = ImageFont.truetype(font_path, canvas_size - 20)

	w, h = drawing.textsize(ch, font=font)
	drawing.text(((canvas_size - w) / 2, (canvas_size - h) / 2),
				 ch, fill=0, font=font);

	flag = np.sum(np.array(image)) == 255 * canvas_size * canvas_size
	if flag:
		return None
	
	image = add_padding(tight_crop_image(np.asarray(image)), canvas_size)
	if len(image.shape) == 2:
		image = np.expand_dims(image, axis=0)

	if show_img:
		img = Image.fromarray(image)
		plt.imshow(img)
		plt.show()
	
	return image


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create image from character using font')
	parser.add_argument('ch', 
						type=lambda x: x if x.isalpha() and len(x) <= 1 else False,
						help='Target ch which want to create the image')
	parser.add_argument('font_path', 
						type=str,
						help='Target font which want to create the image')
	parser.add_argument('-s', '--canvas_size', 
						type=int, 
						default=128, 
						required=False,
						help='Image size')
	args = parser.parse_args()

	image = draw_single_char(args.ch, os.path.abspath(args.font_path), args.canvas_size, True)
	plt.imshow(image, cmap='gray')
	plt.show()

