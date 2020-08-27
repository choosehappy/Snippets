#open sourced with permission from Cedric Walker, 2020
import openslide
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

import numpy as np
import matplotlib.pyplot as plt
import cv2
import xmltodict
import glob


def parse_annot(annot, ds=1):
    dict_xml = xmltodict.parse(annot)
    leng_elem = len(dict_xml['annotation_meta_data']['destination']['annotations']['annotation'])
    coor_dict = {}
    os_y = int(int( dict_xml['annotation_meta_data']['source']['region_of_interest']['@offset_top'] )/ ds)
    os_x = int(int( dict_xml['annotation_meta_data']['source']['region_of_interest']['@offset_left'] )/ ds)
    t_n = dict_xml['annotation_meta_data']['source']['bounding_annotation']['@name']
    p =  dict_xml['annotation_meta_data']['source']['bounding_annotation']['p']
    dict_xml['annotation_meta_data']['source']['@zoom_level']
    x = [int(int(pi['@x'])/ds) + os_x for pi in p]
    y = [int(int(pi['@y'])/ds) + os_y for pi in p]
    coor_dict[t_n] = {'x': [x], 'y': [y]}


    p = dict_xml['annotation_meta_data']['source']['bounding_annotation']['p']
    for i in range(leng_elem):
        name = dict_xml['annotation_meta_data']['destination']['annotations']['annotation'][i]['@name']
        p = dict_xml['annotation_meta_data']['destination']['annotations']['annotation'][i]['p']
        x = [int(int(pi['@x'])/ds) + os_x for pi in p]
        y = [int(int(pi['@y'])/ds) + os_y for pi in p]
        if name in coor_dict.keys():
            xe = coor_dict[name]['x']
            ye = coor_dict[name]['y']
            xe.append(x)
            ye.append(y)
            coor_dict[name]['x'] = xe
            coor_dict[name]['y'] = ye
        else:
            coor_dict[name] = {'x':[x], 'y':[y]}
    return coor_dict, t_n


def coor_to_masks(slide, coor_dict):
    dict_img = {}
    for name in coor_dict.keys():
        x, y = slide.level_dimensions[slide.get_best_level_for_downsample(ds)]
        img = np.zeros((y, x))
        for i in range(len(coor_dict[name]['x'])):
            image = Image.new("L", (x,y))
            draw = ImageDraw.Draw(image)
            draw.polygon(tuple([*zip(coor_dict[name]['x'][i], coor_dict[name]['y'][i])]), fill = 255)
            img = img + np.array(image)
        dict_img[name] = img
    return dict_img


def correct_hierarchical_masks(dict_img, wsi_name, output, t_n):
    for i in dict_img.keys():
        img = np.asarray(dict_img[i].copy(),dtype=np.uint8)
        for j in dict_img.keys():
            if i == j:
                continue
            a , scan = cv2.connectedComponents(np.asarray(dict_img[j], dtype=np.uint8))
            for v in range(1, a-1):
                if np.array_equal(np.logical_and(img, scan == v), np.asarray(scan == v, dtype =np.bool)):
                    img = img.copy() - ((scan==v)*255).copy()

        cv2.imwrite(f'{output}/{wsi}_{i}_mask.png', np.asarray(img, dtype=np.uint8))

annot_path = '/data'
annot_path = glob.glob(f'{annot_path}/*tumeur*')
wsi_path = '/data/hug_brca1_lou'
output = './Masks'
ds = 16
import re
annot = annot_path[0]
for annot in annot_path:
    wsi = re.search(string=annot, pattern='BRCA\d\d_HE').group()
    slide = openslide.open_slide(f'{wsi_path}/{wsi}.mrxs')
    with open(annot, 'r') as file:
        all_of_it = file.read()
    coor_dict, t_n = parse_annot(all_of_it, ds)
    dict_img = coor_to_masks(slide, coor_dict)
    correct_hierarchical_masks(dict_img, wsi, output, t_n)




i = next(iter(dict_img.keys()))
i = 'c tum'
img = np.asarray(dict_img[i].copy(),dtype=np.uint8)
plt.imshow(img)
j = 'stroma fibr'
a , scan = cv2.connectedComponents(np.asarray(dict_img[j], dtype=np.uint8))
plt.imshow(scan)
for v in range(1, a-1):
    if np.array_equal(np.logical_and(img, scan == v), np.asarray(scan == v, dtype =np.bool)):
        img = img.copy() - ((scan==v)*255).copy()
