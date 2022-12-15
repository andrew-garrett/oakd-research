#####################################################################
############################## IMPORTS ##############################
#####################################################################

import json
import os, sys
import cv2
import numpy as np
from threading import Thread
from time import time, sleep

from oak_d.OAKPipeline import OAKPipeline
from processingPipelines.displayPipeline import DisplayPipeline


##############################################################################
############################## DISPLAY PIPELINE ##############################
##############################################################################

"""
TODO:
	- Explore more efficient ways of viewing data (tkinter)
		- Dropdown to select frame to view
"""


class FacialLandmarkPipeline(DisplayPipeline):
	"""
	Class to manage displaying OAK-D extracted data.  Designed to read openCV frames
	and display them.
	"""

	def __init__(self, cfg_fname, LOGGER):
		super().__init__(cfg_fname, LOGGER)

	def processOAKDetections(self, inference_payload, nn_im):
		if "detections" in inference_payload.keys():
			detections = inference_payload["detections"]
		else:
			detections = None
		if detections is not None and len(detections) > 0 and nn_im is not None:
			n, m, _ = nn_im.shape
			for detection in detections:
				bbox, confidence, class_ind = (
					[
						(detection.xmin, detection.ymin),
						(detection.xmax, detection.ymax),
					],
					detection.confidence,
					detection.label,
				)
				ij_tl = (int(bbox[0][0] * m), int(bbox[0][1] * n))
				ij_br = (int(bbox[1][0] * m), int(bbox[1][1] * n))
				label_conf = (
					self.label_mapping[class_ind].rstrip()
					+ ", "
					+ str(round(confidence, 3))
				)
				if hasattr(detection, "spatialCoordinates"):
					label_conf += (
						", (" + str(int(detection.spatialCoordinates.x)) + ", "
					)
					label_conf += str(int(detection.spatialCoordinates.y)) + ", "
					label_conf += str(int(detection.spatialCoordinates.z)) + ")"
				
				self.LOGGER.debug(f"OAKNN Detection Found: {label_conf}")
				if self.useNN in inference_payload.keys():
					ij_center = ij_tl[0] + int((ij_br[0] - ij_tl[0])/2), ij_tl[1] + int((ij_br[1] - ij_tl[1])/2)
					#det_w, det_h = 1.1*
					landmarks = np.array(inference_payload[self.useNN])
					for i in range(0, landmarks.shape[0], 2):
						landmark_coord = (
							ij_center[0] + int((1.1*(landmarks[i]-0.5))*abs(ij_br[0] - ij_tl[0])), 
							ij_center[1] + int((1.04*(landmarks[i+1]-0.5))*abs(ij_br[1] - ij_tl[1])),
						)
		return nn_im

	def processPayload(self, frame_dict):
		if self.framecounter == int(5 * self.fps):
			self.avgFPS = round(self.framecounter / (time() - self.t0), 2)
			self.framecounter = 0
			self.t0 = time()
		show_im = None
		if (
			self.currentView == "nn"
			and self.useNN is not None
			and len(self.useNN) > 0
		):
			show_im = self.drawOAKInferences(frame_dict["nn"], frame_dict["rgb"])
		if self.currentView == "aprilTag" and self.useApril:
			show_im = self.drawAprilTagDetection(frame_dict["april"])
		if self.currentView == "rgb" and self.useRGB:
			if frame_dict["rgb"] is not None:
				show_im = frame_dict["rgb"]
		if self.currentView == "depth" and self.useDepth:
			show_im = self.drawDepthMap(frame_dict["depth"])
		self.drawGUI(show_im)
		self.framecounter += 1

	def start(self):
		super().start()
		self.LOGGER.info("Starting Display Pipeline")
		self.main()
		

	def main(self):
		counter = 1
		t0 = time()
		while self.oak_cam.isOpened():
			current_frame_dict = self.oak_cam.frame_dict
			self.processPayload(current_frame_dict)
			self.LOGGER.debug("Payload Processed")
			if counter % 100 == 0:
				dt = time() - t0
				self.LOGGER.debug(f"Average FPS: {counter / dt}")
			counter += 1
			waitkey = cv2.waitKey(5)
			if waitkey == ord('q'):
				self.running = False
				break

	def stop(self):
		cv2.destroyAllWindows()
		super().stop()
