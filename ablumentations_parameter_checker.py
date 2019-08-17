# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# MIT License
# Developed by in collaboration with Sacheth Chandramouli <sxc868@case.edu>

# +
from __future__ import print_function

import cv2
import PIL
import numpy as np
import tables
import ipywidgets as widgets
import matplotlib.pyplot as plt

from albumentations import *

from ipywidgets import interact, interactive, fixed, interact_manual
# -

fname = r"C:\research\PytorchDigitalPathology\segmentation_epistroma_unet\epistroma_train.pytable"
def get_item(index):
    with tables.open_file(fname,'r') as db:
        img=db.root.img
        mask=db.root.mask
        img = img[index,:,:,:]
        mask = mask[index,:,:]
    return (img,mask)


# +
## HSV Augmentation
i_widget = widgets.IntSlider(min=0, max=100, step=1, value=0, continuous_update=False, description='Index From Pytable')
x_widget = widgets.IntSlider(min=-180, max=180, step=1, value=0, continuous_update=False, description='Hue Shift')
y_widget = widgets.IntSlider(min=0, max=100, step=1, value=0,continuous_update=False, description='Saturation Shift')
z_widget = widgets.IntSlider(min=0, max=100, step=1, value=0,continuous_update=False,description='Value Shift')

def interactive_patch_visualization(i,x,y,z,plot):
    img_or, mask = get_item(i)
    img = augmentations.functional.shift_hsv(img_or, x,y,z)
    fig, ax = plt.subplots(1,2, figsize=(10,10))
    ax[0].imshow(img)
    ax[1].imshow(mask)
    if plot==True:
        augmen = core.composition.OneOf([HueSaturationValue(always_apply=True,hue_shift_limit=x, sat_shift_limit=y, val_shift_limit=z)])
        data = {"image":img_or, "mask":mask}
        fig1, ax1 = plt.subplots(3,3, figsize=(10,10))
        for x in range(0,3):
            for y in range(0,3):
                augmented = augmen(**data)
                aug_image = augmented['image']
                ax1[x,y].imshow(aug_image)
                ax1[x,y].axis('off')
            
interact(interactive_patch_visualization, i=i_widget,x=x_widget,y=y_widget,z=z_widget,plot=True)

# +
## Scaling Augmentation
i_widget = widgets.IntSlider(min=0, max=100, step=1, value=0, continuous_update=False, description='Index From Pytable')
x_widget = widgets.FloatSlider(min=0, max=2, step=.01, value=1, continuous_update=False, description='Rescale Factor')

def interactive_patch_visualization(i,x):
    img, mask = get_item(i)
    img_t = augmentations.functional.scale(img, x)
    mask_t = augmentations.functional.scale(mask,x)
    fig, ax = plt.subplots(2,2, figsize=(15,15))
    ax[0,0].imshow(img)
    ax[0,1].imshow(mask)
    ax[1,0].imshow(img_t)
    ax[1,1].imshow(mask_t)
    
    ax[1,0].set_xlim(ax[0,0].get_xlim())
    ax[1,0].set_ylim(ax[0,0].get_ylim())
    
    ax[1,1].set_xlim(ax[0,0].get_xlim())
    ax[1,1].set_ylim(ax[0,1].get_ylim()) 
    
interact(interactive_patch_visualization, i=i_widget,x=x_widget)

# +
## Rotation Augmentation
i_widget = widgets.IntSlider(min=0, max=100, step=1, value=0, continuous_update=False, description='Index From Pytable')
x_widget = widgets.FloatSlider(min=0, max=360, step=1, value=0, continuous_update=False, description='Degrees')
cv2mode = widgets.Dropdown(
    options=[('BORDER_CONSTANT', 0), ('BORDER_REPLICATE', 1), ('BORDER_REFLECT', 2),('BORDER_WRAP',3),('BORDER_REFLECT_101',4)],
    value=0,
    description='Border Mode:',
)

def interactive_patch_visualization(i,x,selection):
    img, mask = get_item(i)
    img = augmentations.functional.rotate(img, x, border_mode=selection)
    mask = augmentations.functional.rotate(mask,x, border_mode=selection)
    fig, ax = plt.subplots(1,2, figsize=(15,15))
    ax[0].imshow(img)
    ax[1].imshow(mask)
    
interact(interactive_patch_visualization, i=i_widget,x=x_widget,selection=cv2mode)


# +
## Elastic Transformation

i_widget = widgets.IntSlider(min=0, max=100, step=1, value=0, continuous_update=False, description='Index From Pytable')
x_widget = widgets.IntSlider(min=0, max=150, step=1, value=1,continuous_update=False, description='Alpha')
y_widget = widgets.IntSlider(min=0, max=30, step=1, value=10,continuous_update=False, description='Sigma')
z_widget = widgets.IntSlider(min=0, max=1000, step=1, value=50,continuous_update=False,description='Alpha Afine')

cv2mode = widgets.Dropdown(
    options=[('BORDER_CONSTANT', 0), ('BORDER_REPLICATE', 1), ('BORDER_REFLECT', 2),('BORDER_WRAP',3),('BORDER_REFLECT_101',4)],
    value=0,
    description='Border Mode:',
)

def interactive_patch_visualization(i,x,y,z,selection,plot):
    img_or, mask = get_item(i)
    img = augmentations.functional.elastic_transform(img_or,x,y,z,border_mode=selection,approximate=True)
    mask = augmentations.functional.elastic_transform(mask,x,y,z,border_mode=selection,approximate=True)
    fig, ax = plt.subplots(1,2, figsize=(10,10))
    ax[0].imshow(img)
    ax[1].imshow(mask)
    if plot==True:
        augmen = core.composition.OneOf([ElasticTransform(always_apply=True,approximate=True,alpha=x, sigma=y, alpha_affine=z,border_mode=selection)])
        data = {"image":img_or, "mask":mask}
        fig1, ax1 = plt.subplots(3,3, figsize=(10,10))
        for x in range(0,3):
            for y in range(0,3):
                augmented = augmen(**data)
                aug_image = augmented['image']
                ax1[x,y].imshow(aug_image)
                ax1[x,y].axis('off')
            

    
interact(interactive_patch_visualization, i=i_widget,x=x_widget,y=y_widget,z=z_widget,selection=cv2mode,plot=True)

