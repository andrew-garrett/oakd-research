#################################################
#################### IMPORTS ####################
#################################################

import os

from ultralytics import YOLO

import wandb

############################################################
#################### CUSTOM MODEL CLASS ####################
############################################################


class YOLOV8:
    def __init__(self, model_cfg) -> None:
        self.model_cfg = model_cfg
        model_version = self.model_cfg["name"].split("_")[0].lower()
        self.project = f"{os.getcwd()}/runs/{self.model_cfg['task']}/{self.model_cfg['dataset_name']}/{self.model_cfg['model_arch']}/{self.model_cfg['name'].split('_')[0]}"
        self.model = YOLO(f"{self.project}/{model_version}.pt") # load a pretrained model (recommended for training)
        # self.model = YOLO(f"{self.project}/{self.model_cfg['name']}/weights/best.pt") # load a custom trained model (recommended for training)
        # self.model = YOLO(f"{self.project}/{model_version}.yaml") # load a fresh model
    
    def train(self):
        self.model.train(
            data=f"{os.getcwd()}/datasets/{self.model_cfg['task']}/{self.model_cfg['dataset_name']}/data.yaml",    # path to data file, i.e. coco128.yaml
            epochs=self.model_cfg["epochs"],                    # number of epochs to train for
            patience=int(0.25*self.model_cfg["epochs"]),        # epochs to wait for no observable improvement for early stopping of training
            batch=self.model_cfg["batch_size"],                 # number of images per batch (-1 for AutoBatch)
            imgsz=self.model_cfg["imgsz"],                      # size of input images as integer or w,h
            save=True,                                          # save train checkpoints and predict results
            save_period=0,                                      # Save checkpoint every x epochs (disabled if < 1)
            cache=False,                                        # True/ram, disk or False. Use cache for data loading
            device=None,                                        # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
            workers=8,                                          # number of worker threads for data loading (per RANK if DDP)
            project=self.project,                               # project name
            name=self.model_cfg["name"],                        # experiment name
            exist_ok=False,                                     # whether to overwrite existing experiment
            pretrained=False,                                   # whether to use a pretrained model
            optimizer=self.model_cfg["optimizer"],              # optimizer to use, choices=['SGD', 'Adam', 'AdamW', 'RMSProp']
            verbose=False,                                      # whether to print verbose output
            seed=0,                                             # random seed for reproducibility
            deterministic=True,                                 # whether to enable deterministic mode
            single_cls=False,                                   # train multi-class data as single-class
            rect=False,                                         # rectangular training with each batch collated for minimum padding
            cos_lr=False,                                       # use cosine learning rate scheduler
            close_mosaic=0,                                     # (int) disable mosaic augmentation for final epochs
            resume=False,                                       # resume training from last checkpoint
            amp=True,                                           # Automatic Mixed Precision (AMP) training, choices=[True, False]
            lr0=self.model_cfg["lr"],                           # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
            lrf=0.01,                                           # final learning rate (lr0 * lrf)
            momentum=self.model_cfg["momentum"],                # SGD momentum/Adam beta1
            weight_decay=self.model_cfg["weight_decay"],        # optimizer weight decay 5e-4
            warmup_epochs=0.02*self.model_cfg["epochs"],        # warmup epochs (fractions ok)
            warmup_momentum=0.9*self.model_cfg["momentum"],     # warmup initial momentum
            warmup_bias_lr=10.*self.model_cfg["lr"],            # warmup initial bias lr
            box=7.5,                                            # box loss gain
            cls=0.5,                                            # cls loss gain (scale with pixels)
            dfl=1.5,                                            # dfl loss gain
            pose=12.0,                                          # pose loss gain (pose-only)
            kobj=2.0,                                           # keypoint obj loss gain (pose-only)
            label_smoothing=0.0,                                # label smoothing (fraction)
            nbs=64,                                             # nominal batch size
            overlap_mask=True,                                  # masks should overlap during training (segment train only)
            mask_ratio=4,                                       # mask downsample ratio (segment train only)
            dropout=0.0,                                        # use dropout regularization (classify train only)
            val=True,                                           # validate/test during training,
        )

    def validate(self):
        self.model.val(
            # data=f"{os.getcwd()}/datasets/{self.model_cfg['task']}/{self.model_cfg['dataset_name']}/data.yaml",    # path to data file, i.e. coco128.yaml
            # imgsz=self.model_cfg["imgsz"],                     # image size as scalar or (h, w) list, i.e. (640, 480)
            batch=self.model_cfg["test_batch_size"],            # number of images per batch (-1 for AutoBatch)
            save_json=False,                                    # save results to JSON file
            save_hybrid=False,                                  # save hybrid version of labels (labels + additional predictions)
            conf=0.001,                                         # object confidence threshold for detection
            iou=0.6,                                            # intersection over union (IoU) threshold for NMS
            max_det=300,                                        # maximum number of detections per image
            half=True,                                          # use half precision (FP16)
            device=None,                                        # device to run on, i.e. cuda device=0/1/2/3 or device=cpu
            dnn=False,                                          # use OpenCV DNN for ONNX inference
            plots=True,                                         # show plots during training
            rect=False,                                         # rectangular val with each batch collated for minimum padding
            split="val"                                        # dataset split to use for validation, i.e. 'val', 'test' or 'train'
        )
        # Log validation data
        run_dir = f"{os.getcwd()}/runs/{self.model_cfg['task']}/{self.model_cfg['dataset_name']}/{self.model_cfg['model_arch']}/{cfg_wandb['name'].split('_')[0]}/{cfg_wandb['name']}/"
        for val_data in os.listdir(run_dir):
            if "val_batch" in val_data:
                eval_im = wandb.Image(os.path.join(run_dir, val_data))
                wandb.log(
                    {
                        val_data.split(".")[0]: eval_im
                    }
                )