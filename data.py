import numpy as np
from collections import defaultdict
from xml.etree import ElementTree
import glob
import pandas as pd

annotations_path = './data/annotations'
images_path = './data/images'

label_count = defaultdict(int)

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

for file in glob.glob(annotations_path+"/*.xml"):
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
                    label_count[name] += 1
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

print(label_count)

df = pd.DataFrame(dataset)
print(df.head())

