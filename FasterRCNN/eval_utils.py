import cv2 as cv
import matplotlib.pyplot as plt
import torch
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
from tqdm import tqdm
import wandb
from sklearn.metrics import ConfusionMatrixDisplay


def show_boxes(image, boxes, labels=None, title='', fname='Unknown', save=False):
    """ Plot bounding boxes with its label onto image

    Args:
        image (torch.tensor): _description_
        boxes (list): bounding boxes.
        labels (list, optional): labels.
        title (str, optional): figure title. Defaults to ''.
        fname (str, optional): filename to save. Defaults to 'Unknown'.
        save (bool, optional): save file or not. Defaults to False.

    Returns:
        img: image with bounding boxes and labels
    """
    image = image.permute(1,2,0).numpy()

    img = image.copy()
    if np.all(labels==None):
        print('test mode!')
        for bbox in boxes:
            bbox = [int(x) for x in bbox]
            cv.rectangle(img, (bbox[0],bbox[1]),(bbox[2],bbox[3]),color=(0,0,255))

    else:
        for bbox,label in zip(boxes,labels):
            bbox = [int(x) for x in bbox]
            if label == 0:
                color = (0,0,255) # blue
                text = 'with_mask'
                text_length = 60
            elif label == 2:
                color = (0,225,0) # green
                text = 'without_mask'
                text_length = 80
            else:
                color = (255,0,0)
                text = 'masked_incorrect'
                text_length = 90

            img = cv.rectangle(img, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color=color)
            img = cv.rectangle(img,(bbox[0], bbox[1]-8), (bbox[0]+text_length, bbox[1]), color=color, thickness=-1)
            img = cv.putText(img, text, (bbox[0], bbox[1]), cv.FONT_HERSHEY_PLAIN , 0.7, (255,255,255))

    plt.imshow(img)
    plt.title(title)
    if save:
        plt.savefig('./results/' + fname + '.png')
        print(fname + '.png saved!')
    plt.show()
    
    return img


def evaluate(model, dataloader, device, use_wandb=False):
    """ Evaluate model with validation dataset

    Args:
        model (nn Model): model object.
        dataloader (DataLoader): validation data loader
        device (str): cuda or cpu.
        use_wandb (bool, optional): use WandB or not. Defaults to False.

    Returns:
        mean_mAP: mean mAP over validation dataset.
        mean_mAR: mean mAR over validation dataset.
    """
    targets_metric = []
    preds_metric = []
    for images, targets in tqdm(dataloader):
        # move data to GPU
        images = list(image.to(device) for image in images)
        targets = [{k:v.to(device) for k, v in t.items()} for t in targets]

        model.eval()
        mAP_list = []
        mAR_list = []
            
        with torch.no_grad():
            preds = model(images, targets)

        for i, image in enumerate(images):
            target = targets[i]
            pred = preds[i]
            img_size = image.shape[1:]

            targets_metric.append({'boxes': target['boxes'],
                              'labels': target['labels']})

            preds_metric.append({'boxes': pred['boxes'],
                            'scores': pred['scores'],
                            'labels': pred['labels']})

        metric = MeanAveragePrecision().to(device)

        metric.update(preds_metric, targets_metric)
        mAP = metric.compute()
        mean_mAP = torch.mean(mAP['map']).item()
        mean_mAR = torch.mean(mAP['mar_10']).item()
        mAP_list.append(mean_mAP)
        mAR_list.append(mean_mAR)

    mean_mAP = sum(mAP_list) / len(mAP_list)
    mean_mAR = sum(mAR_list) / len(mAR_list)

    return mean_mAP, mean_mAR


def plot_training_loss(loss_list, title='Training Loss', fname='loss', save=False):
    """ Plot training loss over epochs

    Args:
        loss_list (list): list of training losses
        title (str, optional): plot title. Defaults to 'Training Loss'.
        fname (str, optional):filename to save. Defaults to 'loss'.
        save (bool, optional): save file or not. Defaults to False.
    """
    plt.plot(loss_list)
    plt.xlabel('Training Loss')
    plt.xlabel('Epoch')
    plt.title(title)
    if save:
        plt.savefig('./results/' + fname + '.png')
        print(fname + '.png saved!')
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


def plot_mAP(mAP_list, title='mAP@[.5:.95]', fname='mAP', save=False):
    """ Plot mean-average-precision

    Args:
        mAP_list (list): list of mAP.
        title (str, optional): plot title. Defaults to 'mAP@[.5:.95]'.
        fname (str, optional): filename to save. Defaults to 'mAP'.
        save (bool, optional): save file or not. Defaults to False.
    """
    plt.plot(mAP_list)
    plt.xlabel('mAP@[.5:.95]')
    plt.xlabel('Epoch')
    plt.title(title)
    if save:
        plt.savefig('./results/' + fname + '.png')
        print(fname + '.png saved!')
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()


def plot_mAR(mAR_list, title='mAR', fname='mAR', save=False):
    """ Plot mean-average-recall

    Args:
        mAR_list (_type_): list of mAR.
        title (str, optional): plot title. Defaults to 'mAP@[.5:.95]'.
        fname (str, optional): filename to save. Defaults to 'mAP'.
        save (bool, optional): save file or not. Defaults to False.
    """
    plt.plot(mAR_list)
    plt.xlabel('mAR')
    plt.xlabel('Epoch')
    plt.title(title)
    if save:
        plt.savefig('./results/' + fname + '.png')
        print(fname + '.png saved!')
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    

def plot_cm_pr(model, dataloader, device, logger, output_path):
    """ Plot confusion matrix

    Args:
        model: torch.nn model
        dataloader: (eval) dataloader
        device: cuda or cpu
        logger: info logger
        output_path: path to save plots
    """
    target_labels = []
    pred_labels = []
    pred_scores = []

    class_names=['background FN', 'without_mask', 'masked_incorrectly', 'with_mask']
    
    for images, targets in tqdm(dataloader):    
        # move data to GPU
        images = list(image.to(device) for image in images)
        targets = [{k:v.to(device) for k, v in t.items()} for t in targets]

        model.eval()

        with torch.no_grad():
            preds = model(images, targets)

        for i, image in enumerate(images):
            target = targets[i]
            pred = preds[i]
            img_size = image.shape[1:]

            target_label = target['labels']
            pred_label = pred['labels']
            pred_score = pred['scores']

            if len(pred_label) < len(target_label):
                for i in range(len(target_label) - len(pred_label)):
                    pred_label = torch.cat((pred_label, torch.tensor([3]).to(device)))
            elif len(pred_label) > len(target_label):
                for i in range(len(pred_label) - len(target_label)):
                    target_label = torch.cat((target_label, torch.tensor([3]).to(device)))
            target_label = target_label.cpu().tolist()
            pred_label = pred_label.cpu().tolist()
            pred_score = pred_score.cpu().tolist()
            target_labels += target_label
            pred_labels += pred_label
            pred_scores += pred_score

    # wandb log comfusion matrix     
    wandb.log({"cm": wandb.sklearn.plot_confusion_matrix(y_true=target_labels, 
                y_pred=pred_labels, labels=class_names, normalize='pred')})


    # sklearn
    ConfusionMatrixDisplay.from_predictions(y_true=target_labels, y_pred=pred_labels, labels=[0, 1, 2, 3], normalize='true',
                                  display_labels=['with_mask', 'masked_incorrectly', 'without_mask', 'background'], xticks_rotation='vertical', cmap='Blues')
    
    plt.show()
    plt.savefig('{}confusion_mat.png'.format(output_path), bbox_inches='tight')
    cm_fig = plt.figure()
