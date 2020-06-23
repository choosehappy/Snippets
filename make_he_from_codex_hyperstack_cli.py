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

from PIL import Image
import numpy as np
import cv2
import glob
import argparse
from pathlib import Path
import os
from tqdm.autonotebook import tqdm

# +
parser = argparse.ArgumentParser(description='Make output for entire image using Unet')
parser.add_argument('input_pattern',help="input filename pattern. try: *.tif",nargs="*")

parser.add_argument('-o', '--outdir', help="outputdir, default is present working directory", default="./", type=str)
parser.add_argument('-s', '--suffix', help="suffix to append to end of file, default is _HE", default="_HE", type=str)
parser.add_argument('-t', '--type', help="file type, default is png, other options are tif", default="png", type=str)
parser.add_argument('-f', '--force', help="force  output even if it exists", default=False, action="store_true")

args = parser.parse_args()

# -

if len(args.input_pattern) > 1:  # bash has sent us a list of files
    files = args.input_pattern
else:  # user sent us a wildcard, need to use glob to find files
    files = glob.glob(args.input_pattern[0])

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)


for fname in tqdm(files):
    fnameout=f"{Path(fname).stem}{args.suffix}{args.type if args.type[0] == '.' else '.'+args.type}"
    fnameout=Path(args.outdir,fnameout)
    
    if not args.force and os.path.exists(fnameout):
        print(f"Skipping {fnameout} as output file exists and --force is not set")
        continue

    
    img=Image.open(fname)

    img.seek(5)
    r=np.array(img)
    r=(r//255).astype(np.uint8)

    img.seek(6)
    g=np.array(img)
    g=(g//255).astype(np.uint8)

    img.seek(7)
    b=np.array(img)
    b=(b//255).astype(np.uint8)


    a=np.stack((r,g,b),axis=2)

    hsv=cv2.cvtColor(a,cv2.COLOR_RGB2HSV)
    v = hsv[:,:,2]
    # -- this assumes tma spots with white background, we aim to push the background brightness to white
    qval = 255-np.quantile(v.reshape(-1),.90)
    hsv[:,:,2] = v + np.minimum(255 - v, qval)
    #hsv[:,:,2] = v+ (255-v.max())
    

    ab=cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

    cv2.imwrite(str(fnameout),cv2.cvtColor(ab,cv2.COLOR_RGB2BGR))


