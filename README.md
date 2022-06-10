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
- torch==1.5
- torchmetrics--0.8.2
- torchvision==0.10.0
- tqdm==4.61.2
- wandb==0.12.17
- xml.etree
- scipy==1.4.1
- cudatoolkit==10.2.89
- pycocotools
- pillow
- tensorboard
- pyyaml


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
├── yolo
│   ├── images
│   │   ├── test
│   │   ├── train
|   |   └── val
│   ├── label_yolo
│   ├── labels
│   │   ├── test
│   │   ├── train
|   |   └── val
│   ├── models
│   ├── FaceMask.yaml
│   ├── hyp.facemask.yaml
│   ├── hyp.facemaskFocal.yaml
│   ├── loss.py
│   ├── split_data.py
│   └── xml_to_yolo.py
├── yolov5
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
Please follow the following steps correctly in order to run the YOLOv5 model

First, clone the YOLOv5 repo to this repository

```console
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

Then we need to replace and add some files to the yolov5 directory
- Replace **loss.py** with the one in the yolov5
- Add **FaceMask.yaml** to yolov5/data
- Add **hyp.facemask.yaml** and **hyp.facemaskFocal.yaml** to yolov5/data/hyps
- Add every file inside **models** (3 files total) to yolov5/models.

The next two steps in the block below is to show how to setup the data for yolov5 model. In the repository, this two steps have already been done and no further action needs to be done.

-------------------------------------------------------------
In **yolo** directory (not yolov5), run `xml_to_yolo.py` in the terminal to convert the XML anntations into the YOLOv5 format. 

```
python xml_to_yolo.py
```

Split the data into train, validation, and test sets with a ratio of 8:1:1
```
python split_data.py
```
-------------------------------------------------------------
To train the model, run the following command

```
cd yolov5
python train.py --img <image size> --batch <batch size> --epochs <# of epochs> --data FaceMask.yaml --cfg <model.yaml> --hyp hyp.facemask.yaml --name <project name>
```

One of the command I use for my models is

```
python train.py --img 682 --batch 16 --epochs 50 --data FaceMask.yaml --cfg yolov5s.yaml --hyp hyp.facemask.yaml --name facemask1
```

To run inference for test set, run the following command

```
python detect.py --source ../yolo/images/test/ --weights <model weight> --conf <confidence threshold> --save-txt --name <project name>
```

If I want to use the weight from the model facemask1, I can run a command like this

``` 
python detect.py --source ../yolo/images/test/ --weights runs/train/facemask1/weights/best.pt --conf 0.2 --save-txt --name facemask1
```