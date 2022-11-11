# -*- coding: utf-8 -*-
# blogpost - http://andrewjanowczyk.com/de-array-a-tissue-microarray-(tma)-using-qupath-and-python
# co-written with Fan Fan - fxf109@case.edu, 2021
import os
import cv2
import argparse
import numpy as np
import openslide
from distutils.util import strtobool
from tqdm.autonotebook import tqdm

parser = argparse.ArgumentParser (description='Crop each spot in Tissue Micro Array.')
parser.add_argument('wsi_name', help="The prefix name of the WSI file.",  type=str)
parser.add_argument('txt_name', help="The prefix name of the txt file.",  type=str)
parser.add_argument('-s', '--tmaspotsize', help="TMA spot size, default 6000", default=6000, type=int)
parser.add_argument('-o', '--outdir', help="Target output directory", default="./output", type=str)
parser.add_argument('-l', '--level', help="The desired downsample level at which to read each spot.", default=0, type=int)

args = parser.parse_args()
print(f"args: {args}")

# +
wsi_filename = args.wsi_name
txt_filename = args.txt_name
tmaspot_size = args.tmaspotsize
outdir = args.outdir

print(f"Extracting patches for {wsi_filename} into {outdir}")
# -

if not os.path.isdir(f"{outdir}"):
    os.mkdir(f"{outdir}")

slide = openslide.OpenSlide(wsi_filename)

# Print the slide's downsample info
level_dims = slide.level_dimensions[args.level]
level_downsample = slide.level_downsamples[args.level]
print(f'Downsample at level {args.level} is: {level_downsample}')
print(f'WSI dimensions at level {args.level} are: {level_dims}.')

bounds_x = float(slide.properties['openslide.bounds-x']) if ("openslide.bounds-x") in slide.properties.keys() else 0
bounds_y = float(slide.properties['openslide.bounds-y']) if ("openslide.bounds-y") in slide.properties.keys() else 0
ratio_x = 1.0/float(slide.properties['openslide.mpp-x'])
ratio_y = 1.0/float(slide.properties['openslide.mpp-y'])

dataset = np.loadtxt(txt_filename, dtype=str, skiprows=1)
print(f"Number of rows in txt file ：{len(dataset)}")  #note that you aren't guaranteed to get exactly this many spots out, simply because some may be set to false or are missing

for row in tqdm(dataset):
    fname,label,missing,x,y=row
    if(not strtobool(missing)):
        x = (float(x)*ratio_x) + bounds_x
        y = (float(y)*ratio_y) + bounds_y
        print(f"Extracting spot {label} at location", (x, y))
        scaled_spot_size = (int(tmaspot_size/level_downsample), int(tmaspot_size/level_downsample))
        tmaspot = np.asarray(slide.read_region((int(x - tmaspot_size*0.5),int(y-tmaspot_size*0.5)), args.level, scaled_spot_size))[:, :, 0:3]
        tmaspot = cv2.cvtColor(tmaspot,cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{outdir}/{label}.png", tmaspot)
    else:
        print(f'The spot {label} is missing, skipping!')

print('Extracted all the spots!')

