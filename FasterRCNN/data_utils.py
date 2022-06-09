from xml.etree import ElementTree as ET
from collections import defaultdict
from glob import glob
import pandas as pd
import random
import os
import torch
from torch.utils.data import  Dataset
from torchvision import transforms
import cv2 as cv


def read_annot(annotations_path='./data/annotations'):
    """ Read annotaion from xml file and return DataFrame

    Args:
        annotations_path (str, optional): Folder path of annotation files. Defaults to './data/annotations'.

    Returns:
        df: Dataframe containing annotation informations
    """
    annotations_path = './data/annotations'
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
    label_count = defaultdict(int)

    for xml in glob(annotations_path + '/*.xml'):
        tree = ET.parse(xml)
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
                        dataset['file'] += [filename[:-4]]
                        dataset['width'] += [width]
                        dataset['height'] += [height]
                        dataset['label'] += [name]
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                xmin = int(round(float(dim.text)))
                                dataset['xmin'] += [xmin]
                            if 'xmax' in dim.tag:
                                xmax = int(round(float(dim.text)))
                                dataset['xmax'] += [xmax]
                            if 'ymin' in dim.tag:
                                ymin = int(round(float(dim.text)))
                                dataset['ymin'] += [ymin]
                            if 'ymax' in dim.tag:
                                ymax = int(round(float(dim.text)))
                                dataset['ymax'] += [ymax]                       
    df = pd.DataFrame(dataset)
    df = df.sort_values('file', ignore_index=True)

    return df


def get_image(images_path, image_name):
    """ Get image from path by image name

    Args:
        images_path (path): images directory.
        image_name (str): image name.

    Returns:
        img(nd.arrray): image (RGB)
    """
    img_name = os.path.join(images_path, image_name + '.png')
    img = cv.imread(img_name)   
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def get_boxes(image_name, df):
    """ Get bounding box from DataFrame element

    Args:
        image_name (str): image name.
        df (pd.DataFrame): dataframe.

    Returns:
        boxes (list): list of bounding boxes. [[x_min, y_min, x_max, y_max], ...]
    """   
    target = df.loc[df['file'] == image_name]
    boxes = []
    for idx, row in target.iterrows():
        bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
        boxes.append(bbox)
    return boxes


def get_labels(image_name, df):
    """ Get labels from DataFrame element

    Args:
        image_name (str): image name.
        df (pd.DataFrame): dataframe.

    Returns:
        labels (list): list of labels.
    """ 
    target = df.loc[df['file'] == image_name]
    labels = []
    for idx, row in target.iterrows():
        if row['label'] == 'with_mask':
            labels += [0]
        elif row['label'] == 'without_mask':
            labels += [2]
        else:
            labels += [1]
        
    return labels

def sorted_targets(boxes, labels):
    """ Sort targets

    Args:
        boxes: list of bounding boxes. [[x_min, y_min, x_max, y_max], ...]
        labels: list of labels.

    Returns:
        sorted_boxes: sorted list of bounding boxes. [[x_min, y_min, x_max, y_max], ...]
        sorted_labels: sorted list of labels.
    """
    sorted_boxes, sorted_labels = (list(t) for t in zip(*sorted(zip(boxes, labels))))
    return sorted_boxes, sorted_labels


class image_dataset(Dataset):
    def __init__(self, image_list, image_path, xml_path, device='cpu'):
        """ Init image dataset object

        Args:
            image_list (list): _description_
            image_path (str): path to images directory
            xml_path (str):  path to annotations directory
            device (str, optional): cuda or cpu. Defaults to 'cpu'.
        """
        self.image_list = image_list
        self.image_path = image_path
        self.xml_path = xml_path
        self.device = device
       
    def __getitem__(self, idx):
        """ 
        load image 
        """
        img_name = self.image_list[idx][:-4]
        img = get_image(self.image_path, img_name)
        img = transforms.ToTensor()(img)

        df = read_annot(self.xml_path)

        bbox = get_boxes(img_name, df)
        label = get_labels(img_name, df)
        boxes = torch.as_tensor(bbox, dtype=torch.float32)
        labels = torch.as_tensor(label, dtype=torch.int64)

        area = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes)), dtype=torch.int64)
        
        target = {}
        
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_idx'] = torch.tensor([idx])
        target['area'] = area
        target['iscrowed'] = iscrowd

        return img, target
                    
    def __len__(self):
        return len(self.image_list)


def collate_fn(batch):
    """ 
    collate_fn for DataLoader 
    """
    return list(zip(*batch))


def pick_random_img(image_list, images_path):
    """ Pick a random img from images list

    Args:
        image_list (list): list of images
        images_path (str): path to images directory

    Returns:
        img: picked image
        img_list[idx]
    """
    idx = random.randint(1,len(image_list))
    img_name = os.path.join(images_path, image_list[idx])
    img = cv.imread(img_name)   
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img, image_list[idx]