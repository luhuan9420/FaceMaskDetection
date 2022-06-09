import pandas as pd 
import glob 
from xml.etree import ElementTree
import os
import cv2
import numpy as np


# preprocess data
annotations_dir = 'data/annotations/'
images_dir = 'data/images/'

dataset = {
    'file':[], 
    'label':[], 
    'width':[], 
    'height':[],
    'xmin':[],
    'ymin':[],
    'xmax': [],
    'ymax': [],
    }

for file in glob.glob(annotations_dir+"/*.xml"):
    tree = ElementTree.parse(file)

    for element in tree.iter():
        if 'filename' in element.tag:
            filename = element.text
        if 'size' in element.tag:
            for attr in list(element):
                if 'width' in attr.tag:
                    width = int(round(float(attr.text)))
                if 'height' in attr.tag:
                    height = int(round(float(attr.text)))
        if 'object' in element.tag:
            for attr in list(element):
                if 'name' in attr.tag:
                    name = attr.text
                    dataset['label'] += [name]
                    dataset['width'] += [width]
                    dataset['height'] += [height]
                    dataset['file'] += [filename[0:-4]] 
                if 'bndbox' in attr.tag:
                    for dim in list(attr):
                        if 'xmin' in dim.tag:
                            xmin = int(round(float(dim.text)))
                            dataset['xmin'] += [xmin]
                        if 'ymin' in dim.tag:
                            ymin = int(round(float(dim.text)))
                            dataset['ymin'] += [ymin]
                        if 'xmax' in dim.tag:
                            xmax = int(round(float(dim.text)))
                            dataset['xmax'] += [xmax]
                        if 'ymax' in dim.tag:
                            ymax = int(round(float(dim.text)))
                            dataset['ymax'] += [ymax]
df = pd.DataFrame(dataset)
df['annotation_file'] = df['file'] + '.xml'
df['image_file'] = df['file'] + '.png'

new_dir = 'data/cropped_images'
os.mkdir(new_dir)
df['cropped_image_file'] = df['file']

for i in range(len(df)):
    image_filepath = images_dir + df['image_file'].iloc[i]
    image = cv2.imread(image_filepath)
    label = df['label'].iloc[i]
    df['cropped_image_file'].iloc[i] = df['cropped_image_file'].iloc[i][-1] + '-' + str(i) + '-' + label +'.png'
    cropped_image_filename = df['cropped_image_file'].iloc[i]

    xmin = df['xmin'].iloc[i]
    ymin = df['ymin'].iloc[i]
    xmax = df['xmax'].iloc[i]
    ymax = df['ymax'].iloc[i]
    
    cropped_image = image[ymin:ymax, xmin:xmax]

    cropped_image_directory = os.path.join(new_dir, cropped_image_filename) 
    cv2.imwrite(cropped_image_directory, cropped_image)
