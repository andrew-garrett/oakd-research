# oak-d-research

## Description

This repository is to serve as a platform for plug-and-play use of the OAK-D Camera for computer vision research and applications.  Using the depthAI python package, we bring a generalizable and extensible pipeline for grabbing frames from the OAK-D Camera, as well as testing the on-board compute functionality.  With the OpenCV python package, we bring similar level of customizability to off-board image processing and deep computer vision pipelines.  This repository is open-source, so please feel free to add to it and change in any way.

## Installation

Simply use git to create a local copy of this repo:

`git clone https://gitlab.com/andrewgarrett/oak-d-research.git`

Then install the dependencies.  We recommend using pip in a virtual environment, such as venv or conda.

`pip install -r requirements.txt`

## Usage

The `oak_config.json` file serves as the control mechanism for OAK-D parameters.  There are currently three main pipelines:
- OAKPipeline (in `OAKPipeline.py`)
  - Collects frames from OAK-D Camera, with the option to save intermittently ([depthai](https://docs.luxonis.com/en/latest/), [cv2](https://opencv.org/), [np](https://numpy.org/))
- ProcessingPipeline (in `processingPipeline.py`)
  - Applies image processing and deep computer vision operations to streamed or saved images ([cv2](https://opencv.org/), [np](https://numpy.org/))
- DisplayPipeline (in `displayPipeline.py`)
  - Displays frames in a simple and navigable GUI ([cv2](https://opencv.org/))

## Example

```
from OAKPipeline import OAKPipeline
from processingPipelines import ProcessingPipeline
from displayPipeline import DisplayPipeline

...

if __name__ == "__main__":
    oak_cam = OAKPipeline() # initialize a camera pipeline object
    oak_cam.startDevice() # start streaming
    oak_processor = ProcessingPipeline() # initialize a processing pipeline object
    oak_display = DisplayPipeline() # initialize a display pipeline object
    while oak_cam.isOpened():
        oak_cam.read()
        oak_processor.processPayload(oak_cam.frame_dict)
        oak_display.show(oak_cam.frame_dict)
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
```

## Support

For bugs and support, please do so on this repository or send an email to andrewgarrett@rivian.com.

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
