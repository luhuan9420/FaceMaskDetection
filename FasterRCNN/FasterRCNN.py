from data_utils import *
from eval_utils import *
import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset 
from torchvision import models
import numpy as np
from torchvision.ops import box_iou
import wandb


def train_model(model, optimizer, train_loader, test_loader, logger, output_path, device='cpu', lr_scheduler=None, num_epochs=10, use_wandb=False):
    """ Main function for training model 
    
    Args:
        model (nn Model): model.
        optimizer (nn.optim): optimizer
        train_loader (DataLoader): training set dataloader
        test_loader (DataLoader): test set dataloader
        logger: info logger
        output_path (str): path for saving results
        device (str, optional): cuda or cpu. Defaults to 'cpu'.
        num_epochs (int, optional): number of training epochs. Defaults to 10.
        use_wandb (bool, optional): use WandB or not. Defaults to False.

    Returns:
        loss_list(list): list of loss
        mAP_list(list): list of mAP
        mAR_list(list): list of mAR
    """
    loss_list = []
    mAP_list = []
    mAR_list = []

    for epoch in range(num_epochs):
        logger.info('Starting training....{}/{}'.format(epoch+1, num_epochs))
        loss_sub_list = []

        start = time.time()
        for images, targets in tqdm(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k:v.to(device) for k, v in t.items()} for t in targets]

            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            loss_sub_list.append(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        end = time.time()

        epoch_loss = np.mean(loss_sub_list)

        loss_list.append(epoch_loss)
        logger.info('Epoch loss: {:.3f} , time used: ({:.1f}s)'.format(epoch_loss, end-start))
        logger.info('Evaluating....')
        mAP, mAR = evaluate(model, test_loader, device, use_wandb=use_wandb)
        logger.info('mAP@.5:.95 = {} / mAR = {}'.format(mAP, mAR)) 
        mAP_list.append(mAP)
        mAR_list.append(mAR)

        if use_wandb:
            # wandb log
            wandb_dict = {
                'training_loss': epoch_loss,
                'mAP': mAP,
                'mAR': mAR,
                'epoch': epoch
            }
            wandb.log(wandb_dict)

    if use_wandb:
        plot_cm_pr(model, test_loader, device, logger, output_path)   

    return loss_list, mAP_list, mAR_list


def test_random_image():
    ''' test functions with a random image '''
    annotations_path = './data/annotations'
    images_path = './data/images'

    print('=== Read annotations ===')
    df = read_annot(annotations_path)
    print('There are total {} images.'.format(df.shape[0]))
    print(df.head())

    print('=== Test on 1 random image ===')
    image_list = os.listdir(images_path)
    image_name = image_list[random.randint(0, len(image_list))] # random select an image
    img = get_image(images_path, image_name[:-4])
    boxes = get_boxes(image_name[:-4], df)
    labels = get_labels(image_name[:-4], df)
    boxes, labels = sorted_targets(boxes, labels)
    print('boxes:', boxes)
    print('labels:', labels)
    show_boxes(img, boxes, labels)

    
def test_eval():
    """ Test evaluatation function """
    # data paths
    annotations_path = './data/annotations'
    images_path = './data/images'
    
    # Train-test split parameters
    TEST_SIZE = 0.1
    BATCH_SIZE = 2
    SEED = 42

    image_list = os.listdir(images_path)
    mask_dataset = image_dataset(image_list, images_path, annotations_path)
    # train-test split
    train_indices, test_indices = train_test_split(
        range(len(mask_dataset)),
        test_size=TEST_SIZE,
        random_state=SEED)
    # generate subset based on indices
    train_split = Subset(mask_dataset, train_indices)
    test_split = Subset(mask_dataset, test_indices)
    # create batches
    train_loader = DataLoader(train_split, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True,
                            num_workers=2,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_split, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True,
                            num_workers=2,
                            collate_fn=collate_fn)

    # GPU or CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:', device)

    # Setting up the model
    num_classes = 3
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model = model.to(device)
    mAP = evaluate(model, test_loader, device)
    print('mAP:', mAP)



def test_random_pred(model_path):
    """ Test prediction on a random image """
    annotations_path = './data/annotations'
    images_path = './data/images'
    
    # GPU or CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:', device)

    # Train-test split parameters
    TEST_SIZE = 0.1
    BATCH_SIZE = 2
    SEED = 42

    num_classes = 3

    image_list = os.listdir(images_path)
    mask_dataset = image_dataset(image_list, images_path, annotations_path)
    # train-test split
    train_indices, test_indices = train_test_split(
        range(len(mask_dataset)),
        test_size=TEST_SIZE,
        random_state=SEED)

    # generate subset based on indices
    train_split = Subset(mask_dataset, train_indices)
    test_split = Subset(mask_dataset, test_indices)

    # create batches
    train_loader = DataLoader(train_split, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True,
                            num_workers=2,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_split, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True,
                            num_workers=2,
                            collate_fn=collate_fn)

    # set up model
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.to(device)

    model.eval()

    # pick one random img
    for images, targets in test_loader:
        # print(targets[0])
        img = images[0].to(device)
        target = {k:v.to(device) for k, v in targets[0].items()}
        # print(target)

        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device))[0]
        
        gt_boxes = target['boxes'].cpu().numpy()
        gt_labels = target['labels'].cpu().numpy()
        print(gt_boxes)
        print(gt_labels)

        pred_boxes = pred['boxes'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()
        print(pred_boxes)
        print(pred_labels)

        img_gt = show_boxes(img, gt_boxes, gt_labels, title='Ground Truth', fname=model_path[9:-3]+'_gt', save=True)
        img_pred = show_boxes(img, pred_boxes, pred_labels, title='Prediction', fname=model_path[9:-3]+'_pred', save=True)
        
        break


def test_random_image():
    ''' test functions with a random image '''
    annotations_path = './data/annotations'
    images_path = './data/images'

    print('=== Read annotations ===')
    df = read_annot(annotations_path)
    print('There are total {} images.'.format(df.shape[0]))
    print(df.head())

    print('=== Test on 1 random image ===')
    image_list = os.listdir(images_path)
    image_name = image_list[random.randint(0, len(image_list))] # random select an image
    img = get_image(images_path, image_name[:-4])
    boxes = get_boxes(image_name[:-4], df)
    labels = get_labels(image_name[:-4], df)
    boxes, labels = sorted_targets(boxes, labels)
    print('boxes:', boxes)
    print('labels:', labels)
    show_boxes(img, boxes, labels)


def single_img_pred(model_path, image_name):
    """ Predict on single image """
    annotations_path = './data/annotations'
    images_path = './data/images'
    df = read_annot(annotations_path)
    
    # GPU or CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Device:', device)

    # Train-test split parameters
    TEST_SIZE = 0.1
    BATCH_SIZE = 2
    SEED = 42

    num_classes = 3

    image_list = os.listdir(images_path)
    mask_dataset = image_dataset(image_list, images_path, annotations_path)
    # train-test split
    train_indices, test_indices = train_test_split(
        range(len(mask_dataset)),
        test_size=TEST_SIZE,
        random_state=SEED)
    # generate subset based on indices
    train_split = Subset(mask_dataset, train_indices)
    test_split = Subset(mask_dataset, test_indices)
    # create batches
    train_loader = DataLoader(train_split, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True,
                            num_workers=2,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_split, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True,
                            num_workers=2,
                            collate_fn=collate_fn)

    # set up model
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.to(device)

    model.eval()

    img = get_image(images_path, image_name)
    img = transforms.ToTensor()(img)
    img = img.to(device)

    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device))[0]
    
    gt_boxes = get_boxes(image_name, df)
    gt_labels = get_labels(image_name, df)
    print(gt_boxes)
    print(gt_labels)

    pred_boxes = pred['boxes'].cpu().numpy()
    pred_labels = pred['labels'].cpu().numpy()
    print(pred_boxes)
    print(pred_labels)

    img_gt = show_boxes(img, gt_boxes, gt_labels, title='Ground Truth', fname='img'+image_name[12:]+'_gt', save=True)
    img_pred = show_boxes(img, pred_boxes, pred_labels, title='Prediction', fname=model_path[9:-3]+'ep_img'+image_name[12:]+'_pred', save=True)
        