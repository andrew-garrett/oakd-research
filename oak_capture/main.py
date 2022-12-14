#####################################################################
############################## IMPORTS ##############################
#####################################################################

import os
from time import time, sleep
import cv2
from threading import Thread
import logging

from setup_logging import CustomFormatter
from processingPipelines import processingPipeline, displayPipeline, dataCollectionPipeline


#####################################################################
############################## LOGGING ##############################
#####################################################################

# create logger with 'spam_application'
LOGGER = logging.getLogger("oak_d")
LOGGER.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
LOGGER.addHandler(ch)


####################################################################
############################## RUNNER ##############################
####################################################################

if __name__ == "__main__":

	pipeline_type = os.getenv("PIPELINE")
	cfg_fname = f'./oak_d/configs/oak_config.{pipeline_type}.json'
	LOGGER.info(f"Starting {pipeline_type} pipeline, sourcing params from {cfg_fname}")
	if pipeline_type == "demo":
		pipeline = displayPipeline.DisplayPipeline(cfg_fname, LOGGER=LOGGER)
		LOGGER.info(f"Created {pipeline_type} pipeline.")
		pipeline.start()
		LOGGER.info(f"Started {pipeline_type} pipeline.")
	elif pipeline_type == "data_collection":
		pipeline = dataCollectionPipeline.DataCollectionPipeline(cfg_fname, LOGGER=LOGGER)
		LOGGER.info(f"Created {pipeline_type} pipeline.")
		pipeline.start()
		LOGGER.info(f"Started {pipeline_type} pipeline.")
	elif pipeline_type == "oaknn":
		pipeline = processingPipeline.ProcessingPipeline(cfg_fname, LOGGER=LOGGER)
		LOGGER.info(f"Created {pipeline_type} pipeline.")
		pipeline.start()
		LOGGER.info(f"Started {pipeline_type} pipeline.")
		pipeline.main()
	elif pipeline_type == "april":
		pipeline = processingPipeline.ProcessingPipeline(cfg_fname, LOGGER=LOGGER)
		LOGGER.info(f"Created {pipeline_type} pipeline.")
		pipeline.start()
		LOGGER.info(f"Started {pipeline_type} pipeline.")
		pipeline.main()
	else:
		pipeline = processingPipeline.ProcessingPipeline(cfg_fname, LOGGER=LOGGER)
		LOGGER.info(f"Created {pipeline_type} pipeline.")
		pipeline.start()
		LOGGER.info(f"Started {pipeline_type} pipeline.")
		pipeline.main()

	while pipeline.running:
		continue
	
	pipeline.stop()
	LOGGER.info(f"Stopped {pipeline_type} pipeline.")
