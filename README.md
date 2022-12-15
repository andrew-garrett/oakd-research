# oak-d-research

## Description

This repository is to serve as a platform for plug-and-play use of the OAK-D Camera for computer vision research and applications.  Using the depthAI python package, we bring a generalizable and extensible pipeline for grabbing frames from the OAK-D Camera, as well as testing the on-board compute functionality.  With the OpenCV python package, we bring similar level of customizability to off-board image processing and deep computer vision pipelines.  This repository is open-source, so please feel free to fork and create.

## Installation

Simply use git to create a local copy of this repo:

`git clone https://gitlab.com/andrewgarrett/oak-d-research.git`

This project uses [Docker](https://www.docker.com) and [docker-compose](https://docs.docker.com/compose/install/) to access the OAK-D Camera and perform computer vision.  The project also requires the user to install the [awscli](https://github.com/aws/aws-cli) and configure their credentials.

## Usage

The config files in the [oak_d/configs/](./oak_d/configs/) directory control the behavior of the application.  The config files with the prefix `oak_config` control the parameters of capture and processing, where the suffix specifies the type of pipeline to run.  The other config files are the parameterizations for the various OAK Neural Networks that are avaiable.

- OAKPipeline (in `OAKPipeline.py`)
  - Captures frames from OAK-D Camera data sources configured in the `oak_config.XXX.json` file. ([depthai](https://docs.luxonis.com/en/latest/), [cv2](https://opencv.org/), [np](https://numpy.org/))
- ProcessingPipeline (in `processingPipeline.py`)
  - Pipeline Architecture which applies image processing and deep computer vision operations to streamed or saved images ([cv2](https://opencv.org/), [np](https://numpy.org/))
  - DisplayPipeline (in `displayPipeline.py`)
    - Displays frames in a simple and navigable GUI ([cv2](https://opencv.org/))
      - In the `.env` file, set `PIPELINE"demo"`, then run `sudo ./build_and_run.sh`
  - FacialLandmarkPipeline (in `facialLandmarkPipeline.py`)
    - Displays frames with Facial Landmark Detection outputs overlaid
      - In the `.env` file, set `PIPELINE="facial_landmarks"`, then run `sudo ./build_and_run.sh`
  - DataCollectionPipeline (in `dataCollectionPipeline.py`)
    - Saves frames locally and uploads them to S3 Bucket whose name is specified in the .env file ([boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html))
      - In the `.env` file, set `PIPELINE="data_collection"`, then run `sudo ./build_and_run.sh`


## Support

For bugs and support, please do so on this repository or send an email to andrewgarrett2@gmail.com.

## Roadmap

#### Existing Features

- On-board
  - JSON configuration and customizability for parameters such as:
    - Enabling and Disabling of various capture streams
    - Capture Resolution
    - Capture FPS
    - Processing Operations
  - RGB image and Stereo Depth map acquisition
    - RGB/Depth Alignment
  - AprilTag Detection
  - MobileNet SSD & YOLO Object Detection
  - MobileNet SSD Spatial Object Detection
- Off-board
  - Processing operations such as:
    - Resizing
    - Normalization
    - Color Mapping
- Display saved and livestream frames

#### Future Features

- [ ] On-board
  - [ ] Deep Computer Vision Model Pipeline
    - [x] Object Detection
    - [ ] Segmentation
      - [ ] Semantic Segmentation
      - [ ] Instance Segmentation
    - [ ] SpatialAI
      - [x] Spatial Object Detection
      - [ ] Spatial Segmentation
    - [ ] Custom Neural Networks
  - [ ] JSON customizability for other parameters such as:
    - [ ] Deep Computer Vision Model Parameters
  - [ ] Custom image-processing operations for deep model preparation such as:
    - [ ] Resizing
    - [ ] Cropping
    - [ ] AprilTag Pose-estimation
- [ ] Off-board
  - [x] Dataset Curation
  - [ ] Deep Computer Vision Model Pipeline
    - [ ] Object Detection
    - [ ] Segementation
    - [ ] SpatialAI
  - [ ] Deep Computer Vision Model Training (python notebook)
  - [ ] Deep Computer Vision Model Selection

## Contributing

If you are interested in contributing, please feel free to fork this repository or make direct contributions.

## Authors and acknowledgment

Andrew Garrett

## Project status

In Development.
