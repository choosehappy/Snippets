import argparse
import glob
import os
from PIL import Image
import numpy as np
from pptx import Presentation
from pptx.util import Inches
import cv2
from io import BytesIO
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.models import DenseNet
from albumentations import *
from albumentations.pytorch import ToTensor
import torch
import tables


class Dataset(object):
	def __init__(self, fname ,img_transform=None):
		#nothing special here, just internalizing the constructor parameters
		self.fname=fname

		self.img_transform=img_transform
		
		with tables.open_file(self.fname,'r') as db:
			self.classsizes=db.root.classsizes[:]
			self.nitems=db.root.imgs.shape[0]
		
		self.imgs = None
		self.labels = None
		
	def __getitem__(self, index):
		#opening should be done in __init__ but seems to be
		#an issue with multithreading so doing here. need to do it everytime, otherwise hdf5 crashes

		with tables.open_file(self.fname,'r') as db:
			self.imgs=db.root.imgs
			self.labels=db.root.labels
			self.fnames=db.root.filenames

			#get the requested image and mask from the pytable
			img = self.imgs[index,:,:,:]
			label = self.labels[index]
		
		
		img_new = img
		
		if self.img_transform:
			img_new = self.img_transform(image=img)['image']

		return img_new, label, img
	def __len__(self):
		return self.nitems

def addimagetoslide(slide,image_stream,left,top, height, width, resize = 1.0, comment = ''):
	# res = cv2.resize(img , None, fx=resize,fy=resize ,interpolation=cv2.INTER_CUBIC) #since the images are going to be small, we can resize them to prevent the final pptx file from being large for no reason
	# image_stream = BytesIO()
	# Image.fromarray(res).save(image_stream,format="PNG")
	slide.shapes.add_picture(image_stream, left, top ,height,width)
	txBox = slide.shapes.add_textbox(left, Inches(1), width, height)
	tf = txBox.text_frame
	tf.text = comment

def generate_powerpoint(h5_path, model, pptx_save_path, img_transform, device, criteria_dict:dict):
	dset = Dataset(h5_path, img_transform)
	dloader = DataLoader(dset, batch_size=1, 
								shuffle=True, num_workers=8,pin_memory=True)

	# If model is set to None, we assume that predictions have already been generated and saved in the "prediction"
	# dataset of the h5 file.
	filters=tables.Filters(complevel=6, complib='zlib')
	img_dtype = tables.UInt8Atom()
	if (model != None):
		predictions = []
		for img, _, _ in tqdm(dloader):
			img = img.to(device)
			pred = model(img)
			p=pred.detach().cpu().numpy()
			predflat=np.argmax(p,axis=1).flatten()
			predictions.append(predflat)
			

		# TODO something wrong here
		with tables.open_file(h5_path, 'a') as f:
			try:
				predictions_dataset = f.create_carray(f.root, "predictions", img_dtype, np.array(predictions).shape)
			except:
				predictions_dataset = f.root.predictions

			predictions_dataset[:] = predictions
			
	
	# select indices for each category TN, FN, FP, TP
	with tables.open_file(h5_path, 'r') as f:
		gts = np.array(f.root.labels[:]).flatten()
		preds = np.array(f.root.predictions[:]).flatten()
		print(gts.shape)

	ppt = Presentation()
	grid = np.meshgrid(np.arange(5), np.arange(8), indexing='ij')
	cartesian_coords = zip(grid[0].flatten(), grid[1].flatten())

	slidenames = criteria_dict.keys()
	for slidename in slidenames:
		criterion = criteria_dict[slidename]
		inds = np.argwhere(np.logical_and(gts==criterion[0], preds==criterion[1])).flatten()	# find indices where the criterion is true
		num_imgs = cartesian_coords.shape[0] if len(inds) >= cartesian_coords.shape[0] else len(inds) 
		selected_inds = np.random.choice(inds, num_imgs, replace=False)				# choose a subset of max 30 elements from this set
		print(f'Number of images in {slidename}: {len(inds)}')
		blank_slide_layout = ppt.slide_layouts[6]
		slide = ppt.slides.add_slide(blank_slide_layout)
		txBox = slide.shapes.add_textbox(Inches(ppt.slide_height.inches/2), Inches(0.2), Inches(2), Inches(1))	# add slide title
		txBox.text_frame.text = slidename
		for j, ind in tqdm(enumerate(selected_inds)):
			coord = cartesian_coords[j]
			imgnew, label, imgold = dset[ind]
			with tables.open_file(dset.fname, 'r') as f:	# we need to get the image_filename manually.
				img_filename = f.root.filenames[ind]
			plt.imshow(np.moveaxis(imgnew.numpy(), 0, -1))
			plt.title(img_filename.decode("utf-8").split('/')[-1])
			plt.tick_params(color='b',bottom=False, left=False)
			plt.xticks([])
			plt.yticks([])
			with BytesIO() as img_buf:
				plt.savefig(img_buf, format='png', bbox_inches='tight')
				plt.close()
				im = Image.open(img_buf)
				addimagetoslide(slide, img_buf, top=Inches(coord[0]*1.2+1),left=Inches(coord[1]*1.2 + 0.25), width=Inches(im.height/im.width), height=Inches(1))
				im.close()
	ppt.save(pptx_save_path)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--h5path', type=str, help='The path to an h5 file that contains "patch", "ground_truth_label", and "fname" datasets.')	
	parser.add_argument('--model_checkpoint', '-c', type=str, default=None, help='The path to a model checkpoint for the torch.load() method.')
	parser.add_argument('--ppt_save', type=str, help='The full path to save the generated powerpoint.')

	parser.add_argument('--patch_size', type=int, default=224, help='The width of a square patch.')
	parser.add_argument('--gpuid', type=int, default=0, help='The device id.')
	parser.add_argument('--str_criteria', '-s', default=["TRUE NEGATIVES", "FALSE NEGATIVES", "FALSE POSITIVES", "TRUE POSITIVES"], type=str, nargs="+")
	parser.add_argument('--criteria', default=['00', '10', '01', '11'], type=str, nargs="+", help="Each item is a pair of ground-truth,prediction labels.")
	args = parser.parse_args()

	##### DENSENET PARAMS #####
	num_classes=3    #number of classes in the data mask that we'll aim to predict
	in_channels= 3  #input channel of the data, RGB = 3

	growth_rate=32 
	block_config=(2, 2, 2, 2)
	num_init_features=64
	bn_size=4
	drop_rate=0

	##### COMPOSE IMAGE TRANSFORM OBJECT #####
	img_transform = Compose([	# we choose not to apply any image transforms.
		# VerticalFlip(p=.5),
		# HorizontalFlip(p=.5),
		# HueSaturationValue(hue_shift_limit=(-25,0),sat_shift_limit=0,val_shift_limit=0,p=1),
		# Rotate(p=1, border_mode=cv2.BORDER_CONSTANT,value=0),
		RandomSizedCrop((args.patch_size,args.patch_size), args.patch_size,args.patch_size),
		# CenterCrop(args.patch_size,args.patch_size),
		ToTensor()
	])
	
	##### SET DEVICE #####
	print(torch.cuda.get_device_properties(args.gpuid))
	torch.cuda.set_device(args.gpuid)
	device = torch.device(f'cuda:{args.gpuid}' if torch.cuda.is_available() else 'cpu')

	if args.model_checkpoint:
		##### INITIALIZE MODEL #####
		model = DenseNet(growth_rate=growth_rate, block_config=block_config,
					num_init_features=num_init_features, 
					bn_size=bn_size, 
					drop_rate=drop_rate, 
					num_classes=num_classes).to(device)

		checkpoint = torch.load(args.model_checkpoint)
		model.load_state_dict(checkpoint["model_dict"])
		model.eval()
		model.to(device)
		print(model)
		print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")
	else:
		model = None

	##### INITIALIZE CRITERIA DICT #####
	criteria_dict = {}
	for idx in range(len(args.str_criteria)):
		str_criteria = args.str_criteria[idx]
		criteria = [int(item) for item in list(args.criteria[idx])]		# parse criteria into tuples

		criteria_dict[str_criteria] = criteria
	##### CALL GENERATE_POWERPOINT()  #####
	generate_powerpoint(args.h5path, model, args.ppt_save, img_transform, device, criteria_dict)
	
	