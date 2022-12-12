#####################################################################
############################## IMPORTS ##############################
#####################################################################

import os
from time import time, sleep
import cv2
from threading import Thread

from processingPipelines import processingPipeline, displayPipeline, dataCollectionPipeline
# from processingPipelines.processingPipeline import ProcessingPipeline
# from processingPipelines.displayPipeline import DisplayPipeline
# from processingPipelines.dataCollectionPipeline import DataCollectionPipeline


####################################################################
############################## RUNNER ##############################
####################################################################

if __name__ == "__main__":

	pipeline_type = os.getenv("PIPELINE")
	cfg_fname = f'./oak_d/configs/oak_config.{pipeline_type}.json'
	if pipeline_type == "demo":
		pipeline = displayPipeline.DisplayPipeline(cfg_fname)
	elif pipeline_type == "data_collection":
		pipeline = dataCollectionPipeline.DataCollectionPipeline(cfg_fname)
	elif pipeline_type == "oaknn":
		pipeline = processingPipeline.ProcessingPipeline(cfg_fname)
	elif pipeline_type == "april":
		pipeline = processingPipeline.ProcessingPipeline(cfg_fname)
	else:
		pipeline = processingPipeline.ProcessingPipeline(cfg_fname)
	
	pipeline.start()

	while pipeline.running:
		continue
	
	pipeline.stop()
