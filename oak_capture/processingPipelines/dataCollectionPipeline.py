#####################################################################
############################## IMPORTS ##############################
#####################################################################


import json
import os
import cv2
import boto3

from processingPipelines.processingPipeline import ProcessingPipeline


######################################################################################
############################## DATA COLLECTION PIPELINE ##############################
######################################################################################


class DataCollectionPipeline(ProcessingPipeline):

	"""
	Dataset Collection Node.

	Experimenting with different fast ways to save images, such as h5 dataset format.

	"""

	def __init__(self, cfg_fname, dataset_name):
		super().__init__(cfg_fname)
		self.dataset_name = dataset_name
		self.dataset_root = os.path.join("./datasets", dataset_name) #  + ".h5"
		try:
			os.makedirs(self.dataset_root, exist_ok=False)
			if self.useRGB:
				os.mkdir(os.path.join(self.dataset_root, "rgb"))
			if self.useDepth:
				os.mkdir(os.path.join(self.dataset_root, "depth"))
			self.current_db_count = 0
			self.db_num_images = self.params["fps"] * 100.0
		except Exception as e:
			print(e)
			print("Dataset with this name already exists")
			self.current_db_count = len(os.listdir(os.path.join(self.dataset_root, "rgb")))
			self.db_num_images = self.current_db_count + self.params["fps"] * 100.0
			
		#self.s3_client = boto3.client("s3", region_name="us-east-2")

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
							return False
						else:
							#s3_client.upload_file(
							#	sample_datum_fname,
							#	os.getenv("S3_BUCKET_NAME"),
							#	sample_datum_fname[1:]
							#)
							pass
				# For each key, write to the corresponding current_db
				self.current_db_count += 1
				return True
			except Exception as e:
				print(e)
		else:
			pass		
		return False
			# Update the current_db for each datasource


if __name__ == "__main__":
	pass
	'''
	oak_cam = OAKPipeline()
	oak_cam.startDevice()
	print("Device Started")
	oak_processor = DataCollectionPipeline()
	print("Processor Started")
	counter = 1
	t0 = time()
	while oak_cam.isOpened():
		oak_cam.read()
		if not oak_processor.processPayload(oak_cam.frame_dict):
			break
		if counter % 100 == 0:
			dt = time() - t0
			print("Time Elapsed: ", time() - t0)
			print("Average FPS: ", counter / dt)
		counter += 1
	'''
