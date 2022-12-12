#####################################################################
############################## IMPORTS ##############################
#####################################################################

from time import time, sleep
import csv
import cv2
from threading import Thread

from oak_d.OAKPipeline import OAKPipeline
from processingPipelines.processingPipeline import ProcessingPipeline
from processingPipelines.displayPipeline import DisplayPipeline
from processingPipelines.dataCollectionPipeline import DataCollectionPipeline


####################################################################
############################## RUNNER ##############################
####################################################################

if __name__ == "__main__":
	cfg_fname = "./oak_d/configs/oak_config.data_collection.json"
	# Define and start OAKPipeline Capture Thread
	oak_cam = OAKPipeline(cfg_fname)
	oak_capture_thread = Thread(target=oak_cam.startDevice)
	oak_capture_thread.start()
	sleep(3)

	# Enumerate Pipelines Here
	oak_processor = DataCollectionPipeline(cfg_fname, "test000")
	# oak_display = DisplayPipeline(cfg_fname)

	counter = 1
	t0 = time()
	while oak_cam.isOpened():		
		current_frame_dict = oak_cam.frame_dict
		oak_processor.processPayload(current_frame_dict)
		# oak_display.processPayload(current_frame_dict)
		if counter % 100 == 0:
			dt = time() - t0
			print("Time Elapsed: ", time() - t0)
			print("Average FPS: ", counter / dt)
		counter += 1
		#key = cv2.waitKey(1)
		#if key == ord("q"):
		if time() - t0 >= 30:
			break
	# cv2.destroyAllWindows()
	oak_capture_thread.join()

