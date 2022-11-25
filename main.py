#####################################################################
############################## IMPORTS ##############################
#####################################################################

from time import time
import csv
import cv2

from oak_d.OAKPipeline import OAKPipeline
from processingPipelines.processingPipeline import ProcessingPipeline
from displayPipeline import DisplayPipeline


####################################################################
############################## RUNNER ##############################
####################################################################

if __name__ == "__main__":
    oak_cam = OAKPipeline()
    oak_processor = ProcessingPipeline()
    oak_display = DisplayPipeline()
    oak_cam.startDevice()
    counter = 1
    t0 = time()
    while oak_cam.isOpened():
        oak_cam.read()
        oak_processor.processPayload(oak_cam.frame_dict)
        oak_display.show(oak_cam.frame_dict)
        if counter % 100 == 0:
            dt = time() - t0
            print("Time Elapsed: ", time() - t0)
            print("Average FPS: ", counter / dt)
        counter += 1
        key = cv2.waitKey(10)
        if key == ord("q"):
            break