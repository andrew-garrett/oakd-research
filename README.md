# oakd-research

This is a repository for creating computer vision (traditional and deep learning) based applications with the OpenCV AI Kit RGBD Camera (OAK-D Cam).

Each branch is a different application or system which utilize the camera in different ways.  My goal here is to explore topics in vision and edge deployment of vision systems.  This will hopefully educate not only myself, but other young minds interested in the beautiful fields of computer vision and perception.

Some of the topics I will explore include:

- Object Detection
- Segmentation
- Image Processing Pipelines
- Scene Understanding
- SLAM and Pose Estimation
- 3D Reconstruction

# Usage

```
python main.py
```

```
usage: main.py [-h] [--task {cls,objdet,semseg,instseg}]

Runner for various training and logging vision pipelines

optional arguments:
  -h, --help            show this help message and exit
  --task {cls,objdet,semseg,instseg}
                        type of training task for network selection (default: instseg)
```

The model_cfg.json in the "./oak_{task}" dictates the behavior of main.py, where user can select different models, 
training hyperparameters, and datasets