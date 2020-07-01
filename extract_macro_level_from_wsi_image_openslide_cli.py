# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import cv2
import glob
import argparse
from pathlib import Path
import os

#os.environ['PATH'] = 'C:\\research\\openslide\\bin' + ';' + os.environ['PATH'] #can either specify openslide bin path in PATH, or add it dynamically
import openslide


# +
parser = argparse.ArgumentParser(description='Make output for entire image using Unet')
parser.add_argument('input_pattern',help="input filename pattern. try: *.tif",nargs="*")


parser.add_argument('-l','--levels',help="comma seperated list of levels to extract, with 'm' as macro, try: m,6")
parser.add_argument('-o', '--outdir', help="outputdir, default is present working directory", default="./", type=str)
parser.add_argument('-s', '--suffix', help="suffix to append to end of file, default is nothing", default="", type=str)
parser.add_argument('-t', '--type', help="file type, default is png, other options are tif", default="png", type=str)
parser.add_argument('-f', '--force', help="force  output even if it exists", default=False, action="store_true")


args = parser.parse_args()

#args = parser.parse_args(["-f","-lm,6",r"D:\research\16_12\*.mrxs"])
# -

if len(args.input_pattern) > 1:  # bash has sent us a list of files
    files = args.input_pattern
else:  # user sent us a wildcard, need to use glob to find files
    files = glob.glob(args.input_pattern[0])

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)


for fname in files:
    fnameout=f"{Path(fname).stem}{args.suffix}_!!_{args.type if args.type[0] == '.' else '.'+args.type}"
    fnameout=Path(args.outdir,fnameout)
    
    if not args.force and os.path.exists(fnameout):
        print(f"Skipping {fnameout} as output file exists and --force is not set")
        continue

    osh=openslide.OpenSlide(fname)
    
    for level in args.levels.split(","):
        if level =='m': 
            img=osh.associated_images["macro"]
        else:
            level = int(level)
            img = osh.read_region((0, 0), level, osh.level_dimensions[level])
        img = np.asarray(img)[:, :, 0:3]
        cv2.imwrite(str(fnameout).replace("!!",str(level)),cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    
