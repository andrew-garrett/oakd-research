#####################################################################
############################## IMPORTS ##############################
#####################################################################

from time import time
import cv2

from oak_d.OAKPipeline import OAKPipeline


####################################################################
############################## RUNNER ##############################
####################################################################

if __name__ == "__main__":
    forward = False
    oak_cam = OAKPipeline(forward=forward)
    with oak_cam.startDevice() as virtual_cameras:
        counter = 1
        t0 = time()
        while oak_cam.isOpened():
            oak_cam.read()
            if counter % 100 == 0:
                dt = time() - t0
                print("Time Elapsed: ", time() - t0)
                print("Average FPS: ", counter / dt)
            counter += 1
            key = cv2.waitKey(10)
            if key == ord("q"):
                break