#####################################################################
############################## IMPORTS ##############################
#####################################################################


import json
import os
import cv2
import boto3
from time import time

from processingPipelines.processingPipeline import ProcessingPipeline


######################################################################################
############################## DATA COLLECTION PIPELINE ##############################
######################################################################################


class DataCollectionPipeline(ProcessingPipeline):

	"""
	Dataset Collection Node.

	Experimenting with different fast ways to save images, such as h5 dataset format.

	"""

	def __init__(self, cfg_fname, LOGGER):
		super().__init__(cfg_fname, LOGGER)
		self.dataset_name = self.params["dataset"]["name"]
		self.dataset_root = os.path.join("./datasets", self.dataset_name)
		try:
			os.makedirs(self.dataset_root, exist_ok=False)
			if self.useRGB:
				os.mkdir(os.path.join(self.dataset_root, "rgb"))
			if self.useDepth:
				os.mkdir(os.path.join(self.dataset_root, "depth"))
			self.current_db_count = 0
			self.LOGGER.debug("Dataset Initialized")
		except Exception as e:
			self.LOGGER.debug(e)
			self.LOGGER.warning("Dataset with this name already exists")
			self.current_db_count = len(os.listdir(os.path.join(self.dataset_root, "rgb")))
		
		self.db_num_images = self.current_db_count + self.params["dataset"]["n_samples"]
		self.s3_client = boto3.client(
			"s3", 
			aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"), 
			aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
			region_name="us-east-2",
		)
		self.LOGGER.info("S3 Client Initialized")
		self.s3_client.upload_file(
			self.cfg_fname,
			os.getenv("S3_BUCKET_NAME"),
			os.path.join(self.dataset_root, "dataset_config.json")[2:],
		)

	def processPayload(self, frame_dict):
		"""
		Function to loop through the keys of the frame_dict argument and save their contents to the corresponding dataset locations.  Raw Dataset will be stored as folders for each data source from the device (rgb, depth, april, and oak_nn), under which are time_synchronized .h5 files holding <=10seconds of data.
		"""
		if self.current_db_count < self.db_num_images:
			# Loop through the frame_dict's keys
			try:
				for key, value in frame_dict.items():
					if key in ("rgb", "depth") and value is not None and value.shape[0] > 0:
						sample_datum_fname = os.path.join(
							self.dataset_root,
							key,
							f"{key}-{str(self.current_db_count).zfill(4)}.png",
						)
						cv2.imwrite(sample_datum_fname, value)
						if not os.path.exists(sample_datum_fname):
							self.LOGGER.warning(f"Sample not saved to {self.dataset_name}")
							return False
						else:
							self.s3_client.upload_file(
								sample_datum_fname,
								os.getenv("S3_BUCKET_NAME"),
								sample_datum_fname[2:],
							)
							self.LOGGER.debug(f"Sample successfully saved and uploaded to S3")
							os.remove(sample_datum_fname)
				# For each key, write to the corresponding current_db
				self.current_db_count += 1
				return True
			except Exception as e:
				self.LOGGER.warning(e)
				return True
		else:		
			return False
			# Update the current_db for each datasource
	
	def start(self):
		super().start()
		self.LOGGER.info("Starting Data Collection Pipeline")
		self.processing_thread = Thread(target=self.main)
		self.processing_thread.start()

	def main(self):
		counter = 1
		t0 = time()
		while self.oak_cam.isOpened():		
			current_frame_dict = self.oak_cam.frame_dict
			self.LOGGER.debug("Payload Processed")
			if not self.processPayload(current_frame_dict):
				self.running = False
				break
			if counter % 100 == 0:
				dt = time() - t0
				self.LOGGER.debug("Average FPS: ", counter / dt)
			counter += 1


	def stop(self):
		self.processing_thread.join()
		super().stop()
		shutil.rmtree(self.dataset_root)
