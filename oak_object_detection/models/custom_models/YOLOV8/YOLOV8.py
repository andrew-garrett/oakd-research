import os
from ultralytics import YOLO


class YOLOV8:
    def __init__(self, model_cfg) -> None:
        self.model_cfg = model_cfg
        model_version = self.model_cfg["name"].split("_")[0].lower()
        self.project = f"{os.getcwd()}/trained_models/{self.model_cfg['task']}/{self.model_cfg['dataset_name']}/{self.model_cfg['model_arch']}/{self.model_cfg['name'].split('_')[0]}"
        self.model = YOLO(f"{self.project}/{model_version}.pt") # load a pretrained model (recommended for training)
        # self.model = YOLO(f"{self.project}/{model_version}.yaml") # load a fresh model
    
    def train(self):
        self.model.train(
            data=f"{os.getcwd()}/datasets/{self.model_cfg['task']}/{self.model_cfg['dataset_name']}/data.yaml",    # path to data file, i.e. coco128.yaml
            epochs=self.model_cfg["epochs"],                    # number of epochs to train for
            patience=int(0.5*self.model_cfg["epochs"]),         # epochs to wait for no observable improvement for early stopping of training
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
            weight_decay=self.model_cfg["weight_decay"],    	# optimizer weight decay 5e-4
            warmup_epochs=0.05*self.model_cfg["epochs"],        # warmup epochs (fractions ok)
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