#####################################################################
############################## IMPORTS ##############################
#####################################################################

import json
import depthai as dai
import blobconverter
from threading import Thread


##########################################################################
############################## OAK PIPELINE ##############################
##########################################################################

"""
TODO:
	- Implement Custom NN Node creation
	- Measure runtime and iterate design
	- Align/sync depth capture with rgb capture
"""


class OAKPipeline:
	"""
	Class to manage data from OAK-D Camera.  Creates a stream pipeline that is
	configured by config.json.  Currently has functionality for collecting rgb and
	stereo depth images, as on-board apriltag detection data.
	"""

	def __init__(self, cfg_fname, LOGGER):
		self.cfg_fname = cfg_fname
		self.LOGGER = LOGGER
		self.__pipeline = dai.Pipeline()
		self.readJSON()  # read config for populating parameters
		self.frame_dict = {}
		self.__streaming = False

		self.LOGGER.debug(f"Initializing OAK Nodes according to {self.cfg_fname}")
		if self.__useRGB:
			self.initRGBNode()  # rgb node
			self.frame_dict["rgb"] = None
		if self.__useDepth:
			self.initDepthNode()  # full stereo node
			self.frame_dict["depth"] = None
		if self.__useApril:
			self.initAprilTagNode()  # aprilTag detection node
			self.frame_dict["april"] = None
		if self.__useNN is not None and len(self.__useNN) > 0:
			if (
				self.__useNN == "mobilenet_ssd"
				or self.__useNN == "mobilenet_spatial_ssd"
			):
				self.initMobileNetNode()  # mobilenet detection node
			elif self.__useNN == "yolo" or self.__useNN == "tiny_yolo":
				self.initYOLONode()  # yolo detection node
			self.frame_dict["nn"] = None

	def readJSON(self):
		with open(self.cfg_fname, "r") as f:
			params = json.load(f)
			params = params["oakPipeline"]
			self.__params = params
			self.__fps = params["fps"]
			# Basic RGB and Depth Streams
			self.__useRGB = params["rgb"]["useRGB"]
			self.__useDepth = params["depth"]["useDepth"]
			# On-Board Processing Streams
			self.__useApril = params["processing"]["april"]["useApril"]
			self.__useNN = params["processing"]["nn"]["useNN"]
		if self.__useNN is not None and len(self.__useNN) > 0:
			with open(
				"./oak_d/configs/" + self.__useNN.lower() + "_config.json", "r"
			) as f:
				nn_params = json.load(f)
				self.__params["processing"]["nn"] = nn_params

	def initRGBNode(self):
		"""
		Initialize RGB Camera
		"""
		rgbResolution = getattr(
			dai.ColorCameraProperties.SensorResolution,
			self.__params["rgb"]["resolution"],
		)
		if self.__params["rgb"]["resolution"] == "THE_1080_P":
			rgbPreviewResolutionPx = (1920, 1080)  # (1920, 1080) #
		# rgb camera
		self.cam_rgb = self.__pipeline.create(dai.node.ColorCamera)
		self.cam_rgb.setFps(self.__params["fps"])
		self.cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
		self.cam_rgb.setPreviewSize(rgbPreviewResolutionPx)
		self.cam_rgb.setResolution(rgbResolution)
		self.cam_rgb.setInterleaved(False)
		self.cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
		self.xout_rgb = self.__pipeline.createXLinkOut()
		self.xout_rgb.setStreamName("rgb")
		self.cam_rgb.preview.link(self.xout_rgb.input)

	def initDepthNode(self):
		"""
		Initialize Depth Node
		"""
		depthResolution = getattr(
			dai.MonoCameraProperties.SensorResolution,
			self.__params["depth"]["resolution"],
		)
		# left camera
		self.cam_left = self.__pipeline.create(dai.node.MonoCamera)
		self.cam_left.setFps(self.__params["fps"])
		self.cam_left.setResolution(depthResolution)
		self.cam_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
		self.xout_left = self.__pipeline.createXLinkOut()
		self.xout_left.setStreamName("left")
		# right camera
		self.cam_right = self.__pipeline.create(dai.node.MonoCamera)
		self.cam_right.setFps(self.__fps)
		self.cam_right.setResolution(depthResolution)
		self.cam_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
		self.xout_right = self.__pipeline.createXLinkOut()
		self.xout_right.setStreamName("right")
		# full stereo camera
		self.cam_stereo = self.__pipeline.create(dai.node.StereoDepth)
		self.cam_stereo.setLeftRightCheck(True)
		# self.cam_stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
		self.cam_stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
		self.xout_stereo = self.__pipeline.createXLinkOut()
		self.xout_stereo.setStreamName("depth")
		# linking left and right to stereo
		self.cam_left.out.link(self.cam_stereo.left)
		self.cam_right.out.link(self.cam_stereo.right)
		self.cam_stereo.disparity.link(self.xout_stereo.input)

	def initAprilTagNode(self):
		"""
		Initialize AprilTag Detection Node
		"""
		aprilResolution = self.__params["processing"]["april"]["resolution"]
		aprilFamily = getattr(
			dai.AprilTagConfig.Family, self.__params["processing"]["april"]["tagFamily"]
		)
		# apriltag detection
		self.aprilTag = self.__pipeline.create(dai.node.AprilTag)
		self.manip = self.__pipeline.create(dai.node.ImageManip)
		self.xout_AprilTag = self.__pipeline.create(dai.node.XLinkOut)
		self.xout_AprilTagImage = self.__pipeline.create(dai.node.XLinkOut)
		self.xout_AprilTag.setStreamName("aprilTagData")
		self.xout_AprilTagImage.setStreamName("aprilTagImage")
		self.manip.initialConfig.setResize(aprilResolution[0], aprilResolution[1])
		self.manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
		self.aprilTag.initialConfig.setFamily(aprilFamily)
		# Linking
		self.aprilTag.passthroughInputImage.link(self.xout_AprilTagImage.input)
		self.cam_rgb.video.link(self.manip.inputImage)
		self.manip.out.link(self.aprilTag.inputImage)
		self.aprilTag.out.link(self.xout_AprilTag.input)
		# configure aprilTag Detector
		aprilTagConfig = self.aprilTag.initialConfig.get()
		for (key, value) in self.__params["processing"]["april"]["config"].items():
			if key == "quadThresholds":
				for (sub_key, sub_value) in value.items():
					setattr(aprilTagConfig.quadThresholds, sub_key, sub_value)
			else:
				setattr(aprilTagConfig, key, value)
		self.aprilTag.initialConfig.set(aprilTagConfig)

	def initMobileNetNode(self):
		"""
		Initialize MobileNet Detection Network Node
		Parameterized by a depth flag (MobileNetDetectionNetwork vs MobileNetSpatialDetectionNetwork)
		"""
		input_dim = self.__params["processing"]["nn"]["resolution"]
		nnBlob = blobconverter.from_zoo(
			name=self.__params["processing"]["nn"]["nnBlob"], shaves=6
		)
		# dai mobilenet node
		depth_switch = (
			"useDepth" in self.__params["processing"]["nn"].keys()
			and self.__params["processing"]["nn"]["useDepth"]
		)
		if depth_switch:
			if not self.__useDepth:
				self.initDepthNode()
			self.mobilenet = self.__pipeline.create(
				dai.node.MobileNetSpatialDetectionNetwork
			)
		else:
			self.mobilenet = self.__pipeline.create(dai.node.MobileNetDetectionNetwork)
		self.xout_mobilenet = self.__pipeline.create(dai.node.XLinkOut)
		self.xout_mobilenet_network = self.__pipeline.create(dai.node.XLinkOut)
		self.xout_mobilenet.setStreamName(self.__useNN)
		self.xout_mobilenet_network.setStreamName(self.__useNN + " Network")
		self.mobilenet.setConfidenceThreshold(
			self.__params["processing"]["nn"]["confidenceThreshold"]
		)
		self.mobilenet.setNumInferenceThreads(
			self.__params["processing"]["nn"]["threads"]
		)
		self.mobilenet.setBlobPath(nnBlob)
		self.mobilenet.input.setBlocking(False)
		self.mobilenet.out.link(self.xout_mobilenet.input)
		self.mobilenet.outNetwork.link(self.xout_mobilenet_network.input)
		if self.__useRGB:
			if depth_switch:
				self.mobilenet.setDepthLowerThreshold(
					self.__params["processing"]["nn"]["depthLowerThreshold"]
				)
				self.mobilenet.setDepthUpperThreshold(
					self.__params["processing"]["nn"]["depthUpperThreshold"]
				)
				self.cam_stereo.depth.link(self.mobilenet.inputDepth)
				self.manip = self.__pipeline.create(dai.node.ImageManip)
				self.manip.initialConfig.setResize(input_dim[0], input_dim[1])
				self.manip.initialConfig.setKeepAspectRatio(False)
				self.cam_rgb.preview.link(self.manip.inputImage)
				self.manip.out.link(self.mobilenet.input)
			else:
				self.cam_rgb.setPreviewSize(input_dim[0], input_dim[1])
				# self.cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
				self.cam_rgb.preview.link(self.mobilenet.input)
				self.mobilenet.passthrough.link(self.xout_rgb.input)

	def initMobileNetSpatialNode(self):
		"""
		Initialize MobileNet Spatial Detection Network Node
		"""
		input_dim = self.__params["processing"]["nn"]["resolution"]
		nnBlob = blobconverter.from_zoo(
			name=self.__params["processing"]["nn"]["nnBlob"], shaves=6
		)

		# depth only
		if not self.__useDepth:
			self.initDepthNode()
		self.mobilenet = self.__pipeline.create(
			dai.node.MobileNetSpatialDetectionNetwork
		)
		# re-used from above
		self.xout_mobilenet = self.__pipeline.create(dai.node.XLinkOut)
		self.xout_mobilenet_network = self.__pipeline.create(dai.node.XLinkOut)
		self.xout_mobilenet.setStreamName(self.__useNN)
		self.xout_mobilenet_network.setStreamName(self.__useNN + " Network")
		self.mobilenet.setConfidenceThreshold(
			self.__params["processing"]["nn"]["confidenceThreshold"]
		)
		self.mobilenet.setNumInferenceThreads(
			self.__params["processing"]["nn"]["threads"]
		)
		self.mobilenet.setBlobPath(nnBlob)
		self.mobilenet.input.setBlocking(False)
		self.mobilenet.out.link(self.xout_mobilenet.input)
		self.mobilenet.outNetwork.link(self.xout_mobilenet_network.input)
		# depth only
		self.mobilenet.setDepthLowerThreshold(
			self.__params["processing"]["nn"]["depthLowerThreshold"]
		)
		self.mobilenet.setDepthUpperThreshold(
			self.__params["processing"]["nn"]["depthUpperThreshold"]
		)
		self.cam_stereo.depth.link(self.mobilenet.inputDepth)
		self.manip = self.__pipeline.create(dai.node.ImageManip)
		self.manip.initialConfig.setResize(input_dim[0], input_dim[1])
		self.manip.initialConfig.setKeepAspectRatio(False)
		self.cam_rgb.preview.link(self.manip.inputImage)
		self.manip.out.link(self.mobilenet.input)

	def initYOLONode(self):
		"""
		Initialize YOLO Detection Network Node
		"""
		input_dim = self.__params["processing"]["nn"]["resolution"]
		nnBlob = blobconverter.from_zoo(
			name=self.__params["processing"]["nn"]["nnBlob"], shaves=6
		)
		# dai yolo node
		self.yolo = self.__pipeline.create(dai.node.YoloDetectionNetwork)
		self.xout_yolo = self.__pipeline.create(dai.node.XLinkOut)
		self.xout_yolo_network = self.__pipeline.create(dai.node.XLinkOut)
		self.xout_yolo.setStreamName(self.__useNN)
		self.xout_yolo_network.setStreamName(self.__useNN + " Network")
		self.yolo.setConfidenceThreshold(
			self.__params["processing"]["nn"]["confidenceThreshold"]
		)
		self.yolo.setNumClasses(self.__params["processing"]["nn"]["numClasses"])
		self.yolo.setCoordinateSize(self.__params["processing"]["nn"]["coordinateSize"])
		self.yolo.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
		self.yolo.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
		self.yolo.setIouThreshold(self.__params["processing"]["nn"]["iouThreshold"])
		self.yolo.setBlobPath(nnBlob)
		self.yolo.input.setBlocking(False)
		self.yolo.out.link(self.xout_yolo.input)
		self.yolo.outNetwork.link(self.xout_yolo_network.input)

		if self.__useRGB:
			self.cam_rgb.setPreviewSize(input_dim[0], input_dim[1])
			# self.cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
			self.cam_rgb.preview.link(self.yolo.input)
			self.yolo.passthrough.link(self.xout_rgb.input)

	def startDevice(self):
		self.LOGGER.info("Starting OAK Camera")
		self.__device = dai.Device(self.__pipeline)
		if self.__useRGB:
			self.__rgbQueue = self.__device.getOutputQueue(
				name="rgb", maxSize=1, blocking=False
			)
		if self.__useDepth:
			self.__leftQueue = self.__device.getOutputQueue(
				name="left", maxSize=1, blocking=False
			)
			self.__rightQueue = self.__device.getOutputQueue(
				name="right", maxSize=1, blocking=False
			)
			self.__depthQueue = self.__device.getOutputQueue(
				name="depth", maxSize=1, blocking=False
			)
		if self.__useApril:
			self.__manipQueue = self.__device.getOutputQueue(
				name="aprilTagImage", maxSize=1, blocking=False
			)
			self.__aprilQueue = self.__device.getOutputQueue(
				name="aprilTagData", maxSize=1, blocking=False
			)
		if self.__useNN is not None and len(self.__useNN) > 0:
			self.__nnQueue = self.__device.getOutputQueue(
				name=self.__useNN, maxSize=1, blocking=False
			)
			self.__nnNetworkQueue = self.__device.getOutputQueue(
				name=self.__useNN + " Network", maxSize=1, blocking=False
			)
		self.__device.startPipeline()
		self.__streaming = True
		while self.__streaming:
			self.read()
		return

	def read(self):
		if self.__streaming:
			if self.__useRGB:
				rgb_frame = self.__rgbQueue.get()
				if rgb_frame is not None:
					rgb_im = rgb_frame.getCvFrame()
					self.frame_dict["rgb"] = rgb_im

			if self.__useDepth:
				depth_frame = self.__depthQueue.get()
				if depth_frame is not None:
					depth_im = depth_frame.getFrame()
					self.frame_dict["depth"] = depth_im

			if self.__useApril:
				aprilTagData = self.__aprilQueue.get().aprilTags
				aprilTag_frame = self.__manipQueue.get()
				april_im = aprilTag_frame.getCvFrame()
				self.frame_dict["april"] = {"tag_data": aprilTagData, "april_im": april_im}

			if self.__useNN is not None and len(self.__useNN) > 0:
				nnData = self.__nnQueue.get()
				nnNetworkData = self.__nnNetworkQueue.tryGet()
				detections = nnData.detections
				self.frame_dict["nn"] = detections

	def isOpened(self):
		return self.__streaming

	def close(self):
		if hasattr(self, "capture_thread"):
			self.capture_thread.join()
			self.__streaming = False
