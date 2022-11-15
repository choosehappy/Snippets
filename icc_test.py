# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import os
os.add_dll_directory(r"c:\research\openslide\bin")
import openslide

# +
import io

from PIL import Image
from PIL import ImageCms
import numpy as np
import matplotlib.pyplot as plt

import tifffile
# -

fname =r"c:/research/TCGA-D8-A1XR-01Z-00-DX2.A103FB8B-4397-4DD4-8587-90A736407484.svs"

osh=openslide.OpenSlide(fname)

#Openslide will actually tell us if there is an Aperio profile avaialble
icc=osh.properties.get("aperio.ICC Profile", "NA")
print(icc)

#Need to set this to none, otherwise PIL raises an error as its concerned our image is too big and is in fact a decompression bomb
Image.MAX_IMAGE_PIXELS = None

# PIL version
icc = Image.open(fname).info.get('icc_profile') 
f = io.BytesIO(icc)
prf = ImageCms.ImageCmsProfile(f)

# we can see the length of our ICC profile in bytes
len(icc)

#and the associated profile itself
icc

#other properties are available, though not always set. check the documentation for more information
prf.profile.creation_date

ImageCms.getProfileName(prf)

# TIFFFile version
with tifffile.TiffFile(fname) as tif:
    tag = tif.pages[0].tags[34675]
    f = io.BytesIO(tag.value)
prf = ImageCms.ImageCmsProfile(f)

#read a region of the WSI
img=osh.read_region((60000,60000),0,(2000,2000)).convert("RGB")

#create a profile for RGB 
rgbp=ImageCms.createProfile("sRGB")
#and build a transform to for our RGB to ICC space, so that we can apply it faster later
#Update Nov2022, it was pointed out to me that the previous version of this code had the two profiles switched
#however when I tested both versions, the 'swapped' version appeared to result in a better colored image
#practically speaking, I'm unsure what to make of that...i leave it to the user to decide is it better for it to be correct or for it to look nicer?
#experiments to be done....

#icc2rgb = ImageCms.buildTransformFromOpenProfiles(rgbp, prf, "RGB", "RGB") #inverted version
icc2rgb = ImageCms.buildTransformFromOpenProfiles(prf,rgbp, "RGB", "RGB")   #correct version

# #%%timeit
#apply the transform, we can uncomment the previous timeit line to understand how fast tihs occurs, should be about 95ms on a relatively old laptop
result = ImageCms.applyTransform(img, icc2rgb)


#all done, convert to array's for easier manipulation and plotting
img=np.asarray(img)
result=np.asarray(result)

# %matplotlib notebook
#put in notebook mode, so that we can see the pixel values by hovering over pixels

#plot raw image on the left next to calibrated image on right
fig, ax = plt.subplots(1,2, figsize=(10,4))
ax[0].imshow(img)
ax[1].imshow(result)

# %matplotlib inline 

for z,(i,r) in enumerate(zip(img.swapaxes(2,0),result.swapaxes(2,0))): #iterate over each channel
    i=i.reshape(-1) #reshape to a vector
    r=r.reshape(-1)
    
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    
    #plot and compute histograms next to each other for raw / calibrated values, for each channel
    ax[0].set_title(f'Histogram of Raw values: Channel {z}')
    _=ax[0].hist(i,bins=255,density=True)
    
    
    ax[1].set_title(f'Histogram of ICC Applied values: Channel {z}')
    _=ax[1].hist(r,bins=255,density=True)
    
    
    plt.show()


#make the default figure size larger
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10,10)

#show the original image for reference against subsequent bwr versions
plt.imshow(result)
plt.show()

#imported from histoqc.com, an easy way to overlay a binary mask on an image
from skimage import color
def blend2Images(img, mask):
    if (img.ndim == 3):
        img = color.rgb2gray(img)
    if (mask.ndim == 3):
        mask = color.rgb2gray(mask)
    img = img[:, :, None] * 1.0  # can't use boolean
    mask = mask[:, :, None] * 1.0
    out = np.concatenate((mask, img, mask), 2)
    return out


# %matplotlib inline
for z,(i,r) in enumerate(zip(img.swapaxes(2,0).swapaxes(1,2),result.swapaxes(2,0).swapaxes(1,2))):
    diff = (r.astype(int)-i.astype(int)) #IMPORTANT: previously images were uint8, 
                                         #so they could not be 'negative' (interger wraparound/overflow) 
                                         #first need to convert to the signed equivalent type
            
    fig, ax = plt.subplots(1,3, figsize=(20,20))
    
    ax[0].set_title(f"BWR Image: Channel {z} [min: {diff.min()}, max: {diff.max()} ]")
    _=ax[0].imshow(diff,cmap="bwr")
    

    ax[1].set_title(f"Overlay positive shift")
    _=ax[1].imshow(blend2Images(result,diff>0))

    
    
    ax[2].set_title(f"Overlay negative shift")
    _=ax[2].imshow(blend2Images(result,diff<0))

    
    plt.show()


