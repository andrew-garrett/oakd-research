#####################################################################
############################## IMPORTS ##############################
#####################################################################


import json
import os, shutil
import cv2
from threading import Thread
from time import time, sleep

from oak_d.OAKPipeline import OAKPipeline


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

	def __init__(self, cfg_fname, LOGGER):
		self.cfg_fname = cfg_fname
		self.LOGGER = LOGGER
		self.readJSON()

	def readJSON(self):
		with open(self.cfg_fname, "r") as f:
			params = json.load(f)
			oak_params = params["oakPipeline"]
			self.params = params
			self.fps = oak_params["fps"]
			self.useRGB = oak_params["rgb"]["useRGB"]
			self.useDepth = oak_params["depth"]["useDepth"]
			self.useApril = oak_params["processing"]["april"]["useApril"]
			self.useNN = oak_params["processing"]["nn"]["useNN"]
			self.displayResolution = tuple(oak_params["display"]["resolution"])
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
			elif self.useNN == "facial_landmarks":
				self.label_mapping = ["", "Person"]

	def processDepth(self, depth_im):
		return depth_im

	def processRGB(self, rgb_im):
		return rgb_im

	def processPayload(self, frame_dict):
		# self.LOGGER.debug(", ".join(frame_dict.keys()))
		return

	def start(self):
		# Define and start OAKPipeline Capture Thread
		self.oak_cam = OAKPipeline(self.cfg_fname, self.LOGGER)
		self.oak_capture_thread = Thread(target=self.oak_cam.startDevice, daemon=True)
		self.oak_capture_thread.start()
		self.LOGGER.info("Started OAK Capture Thread")
		self.running = True
		while not self.oak_cam.isOpened():
			continue
			sleep(0.5)

	def main(self):
		counter = 1
		t0 = time()
		while self.oak_cam.isOpened():		
			current_frame_dict = self.oak_cam.frame_dict
			self.processPayload(current_frame_dict)
			# self.LOGGER.info("Payload Processed")
			if time() - t0 >= 30:
				self.running = False
				break
			if counter % 100 == 0:
				dt = time() - t0
				# self.LOGGER.debug(f"Average FPS: {counter / dt}")
			counter += 1

	def stop(self):
		self.LOGGER.info("Stopping Pipeline")
		self.oak_capture_thread.join()
