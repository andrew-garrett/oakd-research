#####################################################################
############################## IMPORTS ##############################
#####################################################################


import json
import os
import cv2
from time import time


#################################################################################
############################## PROCESSING PIPELINE ##############################
#################################################################################

"""
TODO:
    - Migrate individual pipeline functions into a utils.py function collection
    - Create ProcessingPipeline parent class that other focused pipeline classes can inherit
        - AprilTag Pose-Estimation Pipeline (depth fusion?)
        - Person Detection
        - Floor Plane Detection
        - 3D Person Detection
        - TSDF Fusion
        - 
"""


class ProcessingPipeline:
    """
    Class to apply image processing operations with OpenCV.
    """

    def __init__(self):
        self.readJSON()

    def readJSON(self):
        with open("./oak_d/configs/oak_config.json", "r") as f:
            params = json.load(f)
            oak_params = params["oakPipeline"]
            self.__params = params
            self.__fps = oak_params["fps"]
            self.__useRGB = oak_params["rgb"]["useRGB"]
            self.__useDepth = oak_params["depth"]["useDepth"]
            self.__useApril = oak_params["processing"]["april"]["useApril"]
            self.__useNN = oak_params["processing"]["nn"]["useNN"]
            self.__displayResolution = tuple(oak_params["display"]["resolution"])
        if self.__useNN is not None and len(self.__useNN) > 0:
            if (
                self.__useNN == "mobilenet_ssd"
                or self.__useNN == "mobilenet_spatial_ssd"
            ):
                with open("./data/label_maps/voc_20cl.txt", "r") as f:
                    self.label_mapping = f.readlines()
            elif self.__useNN == "yolo" or self.__useNN == "tiny_yolo":
                with open("./data/label_maps/coco_80cl.txt", "r") as f:
                    self.label_mapping = f.readlines()

    def processDepth(self, depth_im):
        return depth_im

    def processRGB(self, rgb_im):
        return rgb_im

    def processPayload(self, frame_dict):
        return

    def saveData(self):
        rawROOT = "./data/raw/"
        processingROOT = "./data/processed/"
        try:
            os.mkdir(processingROOT)
        except FileExistsError:
            pass
        raw_im_list = os.listdir(rawROOT)
        for im_name in raw_im_list:
            processed_im = cv2.imread(rawROOT + im_name)
            if im_name[0] == "R":  # RGB images
                processed_im = self.processRGB(processed_im)
            elif im_name[0] == "D":  # DEPTH images
                processed_im = self.processDepth(processed_im)
            else:
                processed_im = None
            if processed_im is not None:
                cv2.imwrite(processingROOT + im_name, processed_im)

    def clearData(self, raw=False, processed=False):
        if raw:
            rawROOT = "./data/raw/"
            rm_files = list(os.listdir(rawROOT))
            for r_f in rm_files:
                os.remove(rawROOT + r_f)
        if processed:
            processedROOT = "./data/processed/"
            rm_files = list(os.listdir(processedROOT))
            for r_f in rm_files:
                os.remove(processedROOT + r_f)


####################################################################
############################## RUNNER ##############################
####################################################################
'''
if __name__ == "__main__":
    oak_cam = OAKPipeline()
    oak_cam.startDevice()
    print("Device Started")
    oak_processor = ProcessingPipeline()
    print("Processor Started")
    t0 = time()
    counter = 1
    while oak_cam.isOpened() and (time() - t0 < 10.0):
        oak_cam.read()
        oak_processor.processPayload(oak_cam.frame_dict)
        if counter % 100 == 0:
            dt = time() - t0
            print("Time Elapsed: ", time() - t0)
            print("Average FPS: ", counter / dt)
        counter += 1
'''
