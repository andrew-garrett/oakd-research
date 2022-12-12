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

	def __init__(self, cfg_fname):
		self.cfg_fname = cfg_fname
		self.readJSON()

	def readJSON(self):
		with open(self.cfg_fname, "r") as f:
			params = json.load(f)
			params = params["oakPipeline"]
			self.params = params
			self.fps = params["fps"]
			self.useRGB = params["rgb"]["useRGB"]
			self.useDepth = params["depth"]["useDepth"]
			self.useApril = params["processing"]["april"]["useApril"]
			self.useNN = params["processing"]["nn"]["useNN"]
			self.displayResolution = tuple(params["display"]["resolution"])
		if self.useNN is not None and len(self.useNN) > 0:
			if (
				self.useNN == "mobilenet_ssd"
				or self.useNN == "mobilenet_spatial_ssd"
			):
				with open("./data/label_maps/voc_20cl.txt", "r") as f:
					self.label_mapping = f.readlines()
				if self.label_mapping is None:
					with open("../data/label_maps/voc_20cl.txt", "r") as f:
						self.label_mapping = f.readlines()
			elif self.useNN == "yolo" or self.useNN == "tiny_yolo":
				with open("./data/label_maps/coco_80cl.txt", "r") as f:
					self.label_mapping = f.readlines()
				if self.label_mapping is None:
					with open("../data/label_maps/coco_80cl.txt", "r") as f:
						self.label_mapping = f.readlines()

	def processDepth(self, depth_im):
		return depth_im

	def processRGB(self, rgb_im):
		return rgb_im

	def processPayload(self, frame_dict):
		return


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
