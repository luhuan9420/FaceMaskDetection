import torch
import matplotlib
import argparse
import logging
import logging.handlers
import os
import sys
import numpy as np
import wandb
from FasterRCNN import *
from data_utils import *
from eval_utils import *
from config import cfg
from focal_loss.focal_loss import FocalLoss

# Set argpars
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', dest='num_epochs', type=int, default=10)
parser.add_argument('--lr', dest='learning_rate', type=int, default=0.01)
parser.add_argument('--focal_loss', dest='use_focal', action='store_true')
parser.add_argument('--no_focal_loss', dest='use_focal', action='store_false')
parser.add_argument('--wandb', dest='use_wandb', action='store_true')
parser.add_argument('--no_wandb', dest='use_wandb', action='store_false')
parser.add_argument('--save', dest='save_file', action='store_true')
parser.add_argument('--run_name', dest = 'run_name', type=str, default=None)
parser.set_defaults(use_focal=False)
parser.set_defaults(use_wandb=False)
parser.set_defaults(save_file=False)

# Example terminal command to run:
# python main.py --epochs 50 --focal_loss --wandb --run_name testname --save

def main(config):
    """ Main experiment program

    Args:
        config (cfg): experiment configuration.
    """
    use_focal = config.use_focal
    use_wandb = config.use_wandb
    save_file = config.save_file
    run_name = config.run_name

    if use_wandb:
        run = wandb.init(project="Faster-RCNN face mask detection", entity="clc70")

        wandb.define_metric("epoch")
        wandb.define_metric("training_loss", step_metric="epoch")
        wandb.define_metric("mAP", step_metric="epoch")
        wandb.define_metric("mAR", step_metric="epoch")


        if run_name:
            wandb.run.name = run_name
            wandb.run.save()



    # data paths
    annotations_path = config.annotations_path
    images_path = config.images_path
    log_path = config.log_path
    output_path = config.output_path
    
    # Set logger
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        open(log_path, 'w').close()
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', 
                                '%m-%d-%Y %H:%M:%S')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    # GPU or CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info('Device: {}'.format(device))
    
    # Train-test split parameters
    TEST_SIZE = 0.1
    BATCH_SIZE = 2
    SEED = 42
    # Training parameters
    lr = config.learning_rate
    momentum = 0.9
    weight_decay = 0.0005
    num_epochs = config.num_epochs

    image_list = os.listdir(images_path)
    mask_dataset = image_dataset(image_list, images_path, annotations_path)

    # generate indices: instead of the actual data we pass in integers instead
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


    # Setting up the model
    num_classes = 3

    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    if use_focal:
        model.roi_heads.mask_rcnn_loss = FocalLoss(alpha=2, gamma=5)
    logger.info('use_focal: {}'.format(use_focal))

    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr, momentum, weight_decay)

    # Main training function
    loss_list, mAP_list, mAR_list = train_model(model, optimizer, train_loader, test_loader, logger, output_path,
                device=device, num_epochs=num_epochs, use_wandb=use_wandb)

    if use_wandb:
        run.finish()
    
    if save_file:
    # Store results
        np.save(config.train_loss_output, np.array(loss_list))
        logger.info('train_loss.npy saved!')
        np.save(config.mAP_output, np.array(mAP_list))
        logger.info('mAP.npy saved!')
        np.save(config.mAR_output, np.array(mAR_list))
        logger.info('mAR.npy saved!')

        # os.makedirs(output_path)
        torch.save(model.state_dict(), os.path.join(config.output_path, 'FasterRCNN_{}.pt'.format(str(num_epochs))))
        logger.info('FasterRCNN_{}.pt saved!'.format(str(num_epochs)))

    plot_training_loss(loss_list, title='Training Loss', fname='loss_{}epochs'.format(str(num_epochs)), save=True)
    plot_mAP(mAP_list, title='mAP@[.5:.95] ({} Epochs)'.format(str(num_epochs)), fname='mAP_{}epochs'.format(str(num_epochs)), save=True)
    plot_mAR(mAR_list, title='mAR({} Epochs)'.format(str(num_epochs)), fname='mAR_{}epochs'.format(str(num_epochs)), save=True)


if __name__ == '__main__':
    args = parser.parse_args()
    config = cfg(args.num_epochs, args.learning_rate, args.use_focal, args.use_wandb, args.save_file, args.run_name)
    main(config)