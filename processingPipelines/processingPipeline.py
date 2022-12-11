#####################################################################
############################## IMPORTS ##############################
#####################################################################


import json
import os
import cv2


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
        with open("./oak_d/configs/oak_config.json", 'r') as f:
            params = json.load(f)
            oak_params = params["oakPipeline"]
            self.__params = params
            self.__useRGB = oak_params["rgb"]["useRGB"]
            self.__useDepth = oak_params["depth"]["useDepth"]
            self.__useApril = oak_params["processing"]["april"]["useApril"]
            self.__useOAKNN = oak_params["processing"]["nn"]["useNN"]
            self.__displayResolution = tuple(oak_params["display"]["resolution"])


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
        raw_im_list = list(os.listdir(rawROOT))
        for im_name in raw_im_list:
            processed_im = cv2.imread(rawROOT + im_name)
            if im_name[0] == "R": # RGB images
                processed_im = self.processRGB(processed_im)
            elif im_name[0] == "D": # DEPTH images
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

if __name__ == "__main__":
    oak_processor = ProcessingPipeline()
    # oak_processor.saveData()
    # oak_processor.clearData(processed=True)
    # import blobconverter
    # nn_work = blobconverter.from_zoo(name="mobilenet-v2", shaves=8)
    # print("Loaded")