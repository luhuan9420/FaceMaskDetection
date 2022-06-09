# FaceMaskDetection
UCSE ECE228 Final Project - Face Mask Classification and Detection using Neural Network

## Dependencies
- python==3.9+

- focal_loss_torch
- glob
- matplotlib==3.4.2
- numpy==1.19.5
- opencv-python==4.5.5.64
- pandas==1.4.1
- Pytorch==1.9.0
- scikit-learn
- torchmetrics--0.8.2
- torchvision==0.10.0
- tqdm==4.61.2
- wandb==0.12.17
- xml.etree


## Data Preparation

## Usage
Make sure that the directories are arranged as follows:

```bash
. FaceMaskDetection
├── data
│   ├── annotations
│   └── images
├── CNN
├── FasterRCNN
│   ├── config.py
│   ├── data_utils.py
│   ├── eval_utils.py
│   ├── FasterRCNN.py
│   ├── main.py
│   └── results
├── YOLOv5
└── README.md
```


### CNN


### Faster R-CNN
To use the WandB API for online experiment tracking, please register a WandB account first.

The general form for running your Faster R-CNN implementation in your terminal is as follows:

```console
python FasterRCNN/main.py --run_name <RUN_NAME>
```
This runs the experiment with default setting as follows:
- 10 epochs
- learning rate = 0.01
- no focal loss
- not using WandB API
- random run name on WandB
- not saving result files


To run the experiments with custom setting, you can use command line arguments as follows:

```console
--epochs <NUM_EPOCHS>
--lr <LEARNING_RATE>
--focal_loss
--wandb
--run_name <RUN_NAME>
--save 
```

For example, you can run the experiment with 50 epochs training with focal loss, using wandb, and saving result files with command as follows:

```console
python FasterRCNN/main.py --epochs 50 --focal_loss --wandb --run_name <RUN_NAME> --save 
```

### YOLOv5
