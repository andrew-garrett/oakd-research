#####################################################################
############################## IMPORTS ##############################
#####################################################################

import json
import os, sys
from time import time
import cv2
import numpy as np

from processingPipelines.processingPipeline import ProcessingPipeline


##############################################################################
############################## DISPLAY PIPELINE ##############################
##############################################################################

"""
TODO:
	- Explore more efficient ways of viewing data (tkinter)
		- Dropdown to select frame to view
"""


class DisplayPipeline(ProcessingPipeline):
	"""
	Class to manage displaying OAK-D extracted data.  Designed to read openCV frames
	and display them.
	"""

	def __init__(self, cfg_fname):
		super().__init__(cfg_fname)
		
		availableViews = []
		if self.useNN is not None and len(self.useNN) > 0:
			availableViews.append("nn")
		if self.useApril:
			availableViews.append("aprilTag")
		if self.useRGB:
			availableViews.append("rgb")
		if self.useDepth:
			availableViews.append("depth")
		self.availableViews = tuple(availableViews)
		self.windowName = "Display"
		self.currentView = self.availableViews[0]  # any of the available views
		cv2.namedWindow(self.windowName, cv2.WND_PROP_FULLSCREEN)
		cv2.setWindowProperty(
			self.windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
		)
		self.buttonWidth = int(0.25 * self.displayResolution[0])
		self.makeButtons()
		cv2.setMouseCallback(self.windowName, self.mouseCallback)
		self.t0 = time()
		self.framecounter = 1
		self.avgFPS = 0

	def makeButtons(self):
		num_views = len(self.availableViews)
		self.buttons = {}
		for i, b_name in enumerate(self.availableViews):
			ij_tl = (0, int(0.3 * ((i) / num_views) * self.displayResolution[1]))
			ij_br = (
				self.buttonWidth,
				int(0.3 * ((i + 1) / num_views) * self.displayResolution[1]),
			)
			self.buttons[b_name] = {}
			self.buttons[b_name]["container"] = (ij_tl, ij_br)
			self.buttons[b_name]["text"] = (
				ij_tl[0] + int(0.1 * self.buttonWidth),
				ij_br[1] - int((0.1 / num_views) * self.displayResolution[1]),
			)

	def drawDepthMap(self, depth_im):
		if depth_im is not None:
			depth_im = cv2.normalize(depth_im, None, 255, 0, cv2.NORM_INF)
			depth_im = cv2.equalizeHist(depth_im)
			depth_im = cv2.applyColorMap(depth_im, cv2.COLORMAP_HOT)
			return depth_im

	def drawAprilTagDetection(self, april):
		if april is not None:
			april_im = april["april_im"]
			april_im = cv2.cvtColor(april_im, cv2.COLOR_GRAY2RGB)
			for tag in april["tag_data"]:
				bbox = [tag.topLeft, tag.topRight, tag.bottomRight, tag.bottomLeft]
				pt_i = bbox[0]
				for pt in bbox[1:]:
					april_im = cv2.line(
						april_im,
						(int(pt_i.x), int(pt_i.y)),
						(int(pt.x), int(pt.y)),
						(0, 255, 0),
						3,
					)
					pt_i = pt
				april_im = cv2.line(
					april_im,
					(int(pt_i.x), int(pt_i.y)),
					(int(bbox[0].x), int(bbox[0].y)),
					(0, 255, 0),
					3,
				)
			return april_im

	def drawOAKDetections(self, detections, nn_im):
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
				nn_im = cv2.rectangle(nn_im, ij_tl, ij_br, (0, 255, 0), 3)
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
				nn_im = cv2.putText(
					nn_im,
					label_conf,
					(ij_tl[0] - 15, ij_tl[1] - 15),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.5,
					(0, 255, 0),
					2,
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
			show_im = self.drawOAKDetections(frame_dict["nn"], frame_dict["rgb"])
		if self.currentView == "aprilTag" and self.useApril:
			show_im = self.drawAprilTagDetection(frame_dict["april"])
		if self.currentView == "rgb" and self.useRGB:
			if frame_dict["rgb"] is not None:
				show_im = frame_dict["rgb"]
		if self.currentView == "depth" and self.useDepth:
			show_im = self.drawDepthMap(frame_dict["depth"])
		self.drawGUI(show_im)
		self.framecounter += 1

	def drawGUI(self, show_im):
		if show_im is not None:
			show_im = cv2.resize(show_im, self.displayResolution)
			if self.avgFPS > 0:
				show_im = cv2.putText(
					show_im,
					"FPS: " + str(self.avgFPS),
					(int(0.05 * show_im.shape[1]), int(0.95 * show_im.shape[0])),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.75,
					(0, 255, 0),
					2,
				)
			show_im = cv2.putText(
				show_im,
				self.currentView,
				(int(0.75 * show_im.shape[1]), int(0.95 * show_im.shape[0])),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.75,
				(0, 255, 0),
				2,
			)
			for (b_name, button) in self.buttons.items():
				if self.currentView == b_name:
					color = (0, 255, 0)
				else:
					color = (255, 255, 0)
				show_im = cv2.rectangle(
					show_im, button["container"][0], button["container"][1], color, -1
				)
				show_im = cv2.putText(
					show_im,
					b_name,
					button["text"],
					cv2.FONT_HERSHEY_SIMPLEX,
					0.75,
					(0, 0, 0),
					2,
				)
			cv2.imshow(self.windowName, show_im)

	def chooseFrame(self, eventKey):
		numViews = len(self.availableViews)
		if eventKey == ord("0"):
			if numViews > 0:
				self.currentView = self.availableViews[0]
		elif eventKey == ord("1"):
			if numViews > 1:
				self.currentView = self.availableViews[1]
		elif eventKey == ord("2"):
			if numViews > 2:
				self.currentView = self.availableViews[2]
		elif eventKey == ord("3"):
			if numViews > 3:
				self.currentView = self.availableViews[3]

	def mouseCallback(self, event, x, y, flags, params):
		if event == cv2.EVENT_LBUTTONDOWN:
			for (b_name, button) in self.buttons.items():
				if (
					x > button["container"][0][0]
					and x <= button["container"][1][0]
					and y > button["container"][0][1]
					and y <= button["container"][1][1]
				):
					self.currentView = b_name
					break


####################################################################
############################## RUNNER ##############################
####################################################################
if __name__ == "__main__":
	pass
'''
if __name__ == "__main__":
	oak_cam = OAKPipeline()
	oak_cam.startDevice()
	print("Device Started")
	oak_display = DisplayPipeline()
	print("Display Started")
	while oak_cam.isOpened():
		oak_cam.read()
		key = cv2.waitKey(1)
		oak_display.chooseFrame(key)
		oak_display.show(oak_cam.frame_dict)
		if key == ord("q"):
			break
	cv2.destroyAllWindows()
'''
