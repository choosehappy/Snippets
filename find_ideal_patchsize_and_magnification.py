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
#     display_name: Python [conda env:Pytorch]
#     language: python
#     name: conda-env-Pytorch-py
# ---

# +
import skimage
import tables
import os,sys
import glob
import math
import random
import PIL
import numpy as np
import cv2
import sklearn.feature_extraction.image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# %matplotlib inline
# %matplotlib widget

# +
# Collect image and mask filenames
image_files=glob.glob('./Tubules/PAS/*.png')
files = [(x, x.replace("./Tubules/PAS/","./Tubules/mask/").replace(".png","_mask.png")) for x in image_files]

initial_magnification = 40 # Magnification of the original image

# + tags=[]
areas = [] # Area of each OOI
rect_max = [] # Vertical or Horizontal dimension of each OOI (whichever is larger)
centroids = [] # Centroid of each OOI
image_index = [] # Which image each OOI is on
patch_label = [] # Track mask value for each OOI
edge_patch_thresh = 256 # Ignore OOI too close to the edge of the slide image (0 to disable)

# Gather metrics from mask images
for i,(_,fname) in enumerate(files):
    img_mask = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)[:, :, 0].squeeze()
    img_mask_measure = skimage.measure.label(img_mask)
    rp = skimage.measure.regionprops(img_mask_measure)
    used_patches = 0
    for r in rp:
        if (r.centroid[0] >= edge_patch_thresh and r.centroid[1] >= edge_patch_thresh and 
            r.centroid[0] <= img_mask.shape[0] - edge_patch_thresh and r.centroid[1] <= img_mask.shape[1] - edge_patch_thresh):
            image_index.append(i)
            areas.append(r.area)
            rect_max.append(max([r.bbox[2]-r.bbox[0],r.bbox[3]-r.bbox[1]]))
            centroids.append(r.centroid)
            patch_label.append(img_mask[r.coords[0,0],r.coords[0,1]])
            used_patches += 1
    print(f"fname: {fname}, \tNumber of total patches: {len(rp)},\tNumber of usable patches:{used_patches}")

total_ooi = len(image_index)
areas = np.asarray(areas)
rect_max = np.asarray(rect_max)
centroids = np.asarray(centroids)

# +
# Find a magnification that works best (Magnification Test)

trial_magnifications = [30, 20, 10, 5] # What magnifications to test
padding = 10 # Number of pixels of context to each OOI
mask = False # Visualize with or without mask applied

# Create a matrix of OOI, left column is the original magnification, each column to the right is a different magnification

a = plt.figure(figsize=(15,30))
plt.tight_layout(pad = 0.5)
dim = (10, 1 + len(trial_magnifications))

def compare_plot(row, index, y_label = "patch "):
    half_patch_size = rect_max[index]//2
    dx = [int(centroids[index][0] - half_patch_size - padding),int(centroids[index][0] + half_patch_size + padding)]
    dy = [int(centroids[index][1] - half_patch_size - padding),int(centroids[index][1] + half_patch_size + padding)]
    offset = -1*min(dx[0],dx[1],dy[0],dy[1],0)
    dx[0] += offset
    dx[1] += offset
    dy[0] += offset
    dy[1] += offset
    ax = plt.subplot2grid(dim, (row, 0))
    image = cv2.cvtColor(cv2.imread(files[image_index[index]][0]),cv2.COLOR_BGR2RGB)
    image = cv2.copyMakeBorder(image, offset, offset, offset, offset, cv2.BORDER_CONSTANT)
    patch = image[dx[0]:dx[1], dy[0]:dy[1]]
    if mask:
        img_mask = cv2.cvtColor(cv2.imread(files[image_index[index]][1]), cv2.COLOR_BGR2RGB)
        img_mask = cv2.copyMakeBorder(img_mask, offset, offset, offset, offset, cv2.BORDER_CONSTANT)[dx[0]:dx[1],dy[0]:dy[1],0].squeeze()
        img_mask = img_mask == patch_label[index]
        patch = np.multiply(patch, img_mask[:,:,None])
    plt.imshow(patch)
    ax.set_ylabel(y_label + str(index))
    if row == 0:
        ax.set_title("Original | "+str(initial_magnification)+"x")
    for j, mag in enumerate(trial_magnifications):
        patch_size_resize = int((rect_max[index] + 2*padding)*mag/initial_magnification)
        patch_resize = cv2.resize(patch, (patch_size_resize, patch_size_resize), interpolation = cv2.INTER_NEAREST)
        ax = plt.subplot2grid(dim, (row, j + 1))
        plt.imshow(patch_resize)
        if row == 0:
            ax.set_title(str(mag)+"x")

# First 4 OOI are random
for i in range(0,4):
    compare_plot(i, random.randint(0, total_ooi - 1))
        
# 5th OOI is smallest 1% by area, 6th is smallest 5% by area, 7th is smallest 10% by area, 8th is smallest 15% by area
compare_plot(4, np.argmin(np.abs(np.array(areas)-np.percentile(areas, 1))), y_label = "Smallest | 1% | patch ")
compare_plot(5, np.argmin(np.abs(np.array(areas)-np.percentile(areas, 5))), y_label = "Smallest | 5% | patch ")
compare_plot(6, np.argmin(np.abs(np.array(areas)-np.percentile(areas, 10))), y_label = "Smallest | 10% | patch ")
compare_plot(7, np.argmin(np.abs(np.array(areas)-np.percentile(areas, 15))), y_label = "Smallest | 15% | patch ")

# 9th is average area, 10th is largest 5% by area 
compare_plot(8, np.argmin(np.abs(np.array(areas)-np.percentile(areas, 50))), y_label = "Average | 50% | patch ")
compare_plot(9, np.argmin(np.abs(np.array(areas)-np.percentile(areas, 95))), y_label = "Largest | 95% | patch ")

plt.show()

# +
# Choose a final magnification and update patch metrics
final_magnification = 20

areas_final = areas*(final_magnification/initial_magnification)**2
rect_final = rect_max*(final_magnification/initial_magnification)
resize = final_magnification/initial_magnification
centroids_final = centroids*(final_magnification/initial_magnification)

# +
# Find a patch size that works the best (Size Test)

mark_patchsizes = [(512,"yellow"),(256,"green"),(128,"orange"),(64,"red")] # What patch sizes to test
bins = 40 # Number of bins in histograms

# Visualize patch areas compared to the largest circle that can fit in each patch size
areas_total = areas_final[areas_final < 220000] # Filter outliers to make graphs readable
a = plt.figure(figsize=(14,5))
plt.subplot(1, 2, 1)
plt.hist(areas_total,bins=bins)
for ps,color in mark_patchsizes:
    plt.axvline(x=math.pi*((ps/2)**2), color=color)
plt.title("Areas")

# Visualize patch maximum vertical/horizontal dimensions compared to each patch size
rect_total = rect_final[rect_final < 600] # Filter outliers to make graphs readable
plt.subplot(1, 2, 2)
plt.hist(rect_total,bins=bins)
for ps,color in mark_patchsizes:
    plt.axvline(x=ps, color=color)
plt.title("Normal Axis Lengths")
plt.show()

# Print the number of patches that will be cut off at each patch size
#patchsizes = [x[0] for x in mark_patchsizes]
mark_patchsizes.sort()
for dim,_ in mark_patchsizes:
    i = len(rect_final[rect_final<=dim])
    print(f"{i} / {total_ooi}, {int(100*i/total_ooi)}% fit in {dim}px patch size")

# +
# Create a matrix of OOI that are cutoff at each patch size in mark_patchsizes, each column is a different patch size
# A patch cutoff at a larger patch size in the mark_patchsizes will not be included in smaller patch sizes

mask = True # Visualize with or without mask applied
padding = 64 # Visualize at larger patch size to show what is cutoff
example_count = 5 # Number of cutoff examples per patch size

mark_patchsizes.sort(reverse=1)
a = plt.figure(figsize=(15,20))
plt.tight_layout(pad = 0.5)
dim = (example_count,len(mark_patchsizes))
rect_cutoff = rect_final.copy()


for i,(ps,_) in enumerate(mark_patchsizes):
    half_patch_size = ps//2 + padding
    for j in range(0, example_count):
        image = None
        while np.shape(image) != (ps + 2*padding,ps + 2*padding,3) and len(rect_cutoff[rect_cutoff >= ps]) > 0:
            index = random.choice(np.where(rect_cutoff >= ps)[0])
            image = cv2.cvtColor(cv2.imread(files[image_index[index]][0]),cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (int(image.shape[1]*resize), int(image.shape[0]*resize)),interpolation = cv2.INTER_NEAREST)
            dx = (int(centroids_final[index][0] - half_patch_size),int(centroids_final[index][0] + half_patch_size))
            dy = (int(centroids_final[index][1] - half_patch_size),int(centroids_final[index][1] + half_patch_size))
            image = image[dx[0]:dx[1], dy[0]:dy[1]]
            rect_cutoff[index] = 0
        if len(rect_cutoff[rect_cutoff >= ps]) > 0:
            if mask:
                img_mask = cv2.cvtColor(cv2.imread(files[image_index[index]][1]), cv2.COLOR_BGR2RGB)
                img_mask = cv2.resize(img_mask, (int(img_mask.shape[1]*resize), int(img_mask.shape[0]*resize)),
                                      interpolation = cv2.INTER_NEAREST)[dx[0]:dx[1], dy[0]:dy[1], 0].squeeze()
                img_mask = img_mask == patch_label[index]
                image = np.multiply(image, img_mask[:,:,None])
            ax = plt.subplot2grid(dim, (j, i))
            if j == 0:
                ax.set_title(f"Cutoff by Patch Size {ps}px")
            plt.imshow(image)
            for ps2,color in mark_patchsizes:
                if ps2 <= ps:
                    ddx = half_patch_size - ps2//2
                    ddy = ps2
                    ax.add_patch(Rectangle((ddx,ddx),ddy,ddy,linewidth=2, edgecolor=color, facecolor="none"))
        else:
            print(f"Ran out of patches cutoff at {ps}px patchsize")
        
    rect_cutoff[rect_cutoff > ps] = 0

plt.show()
# -

a = plt.figure(figsize=(10,5))
half_patch_size = 128
#index = random.randint(0, len(files))
index = 118
image1 = cv2.cvtColor(cv2.imread(files[index][0]),cv2.COLOR_BGR2RGB)
resize = .25
image2 = cv2.resize(image1, (int(image.shape[1]*resize), int(image.shape[0]*resize)),interpolation = cv2.INTER_NEAREST)
dx = (int(centroids_final[index][0] - half_patch_size),int(centroids_final[index][0] + half_patch_size))
dy = (int(centroids_final[index][1] - half_patch_size),int(centroids_final[index][1] + half_patch_size))
patch1 = image1[dx[0]:dx[1], dy[0]:dy[1]]
#dx = (int(centroids_final[index][0]//2 - half_patch_size),int(centroids_final[index][0]//2 + half_patch_size))
#dy = (int(centroids_final[index][1]//2 - half_patch_size),int(centroids_final[index][1]//2 + half_patch_size))
patch2 = image2[dx[0]//4:dx[1]//4, dy[0]//4:dy[1]//4]
ax = plt.subplot2grid((1,2), (0, 0))
ax.set_title("256px by 256px (40x)")
plt.imshow(patch1)
ax = plt.subplot2grid((1,2), (0, 1))
ax.set_title("64px by 64px (10x)")
plt.imshow(patch2)

a = plt.figure(figsize=(10,5))
half_patch_size = 128
#index = random.randint(0, len(files))
index = 118
image1 = cv2.cvtColor(cv2.imread(files[index][0]),cv2.COLOR_BGR2RGB)
resize = .5
image2 = cv2.resize(image1, (int(image.shape[1]*resize), int(image.shape[0]*resize)),interpolation = cv2.INTER_NEAREST)
dx = (int(centroids_final[index][0] - half_patch_size),int(centroids_final[index][0] + half_patch_size))
dy = (int(centroids_final[index][1] - half_patch_size),int(centroids_final[index][1] + half_patch_size))
patch1 = image1[dx[0]:dx[1], dy[0]:dy[1]]
dx = (int(centroids_final[index][0]//2 - half_patch_size),int(centroids_final[index][0]//2 + half_patch_size))
dy = (int(centroids_final[index][1]//2 - half_patch_size),int(centroids_final[index][1]//2 + half_patch_size))
patch2 = image2[dx[0]:dx[1], dy[0]:dy[1]]
ax = plt.subplot2grid((1,2), (0, 0))
ax.set_title("40x")
plt.imshow(patch1)
ax = plt.subplot2grid((1,2), (0, 1))
ax.set_title("20x")
plt.imshow(patch2)


