# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import torch
import torchvision

import cv2
import numpy as np
 
device=torch.device("cuda:0")
# -


# !wget http://www.andrewjanowczyk.com/wp-content/uploads/2016/08/10279_500_f00182_original.jpg

#--- make a more highly compressed version 
data=cv2.imread('10279_500_f00182_original.jpg')
cv2.imwrite('10279_500_f00182_original_highly.jpg',data,[int(cv2.IMWRITE_JPEG_QUALITY), 70])

# !ls -lh *.jpg

# +
#test reading decoding and copying
# -

# %%timeit
data=cv2.imread('10279_500_f00182_original.jpg')
at=torch.from_numpy(data).to(device)

# +
#test reading and using nvJpeg to decode on device
# -

# %%timeit
data=torchvision.io.read_file("10279_500_f00182_original.jpg")
at = torchvision.io.decode_jpeg(data, device=device)

# +
#---- More highly compressed version
# -

# %%timeit
data=cv2.imread('10279_500_f00182_original_highly.jpg')
at=torch.from_numpy(data).to(device)

# %%timeit
data=torchvision.io.read_file("10279_500_f00182_original_highly.jpg")
at = torchvision.io.decode_jpeg(data, device=device)


# +
#---- Large random image
# -

a=np.random.uniform(low=0,high=255,size=(10000,10000,3)).astype(np.uint8)

cv2.imwrite('output.jpg',a) ## produces a 111MB file

# %%timeit
data=cv2.imread('output.jpg')
at=torch.from_numpy(data).to(device)

# %%timeit
data=torchvision.io.read_file("output.jpg")
img = torchvision.io.decode_jpeg(data, device=device)
