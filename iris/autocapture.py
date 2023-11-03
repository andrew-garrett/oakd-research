#################### IMPORTS ####################
#################################################


import argparse
import os
from typing import Sequence, Tuple

import cv2
import numpy as np

from iris.utils import box_iou, enlarge_box, save_sharpest_image

#################### GLOBAL VARIABLES ####################
##########################################################

CV2_FACE_CASCADE_MODEL = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore
)
CV2_EYE_CASCADE_MODEL = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"  # type: ignore
)

AUTOCAPTURE_MODE = "eye"  # autocapture mode ("face", "eye", "face and eye")
IOU_THRESH = 0.5  # threshold for autocapture alignment
PADDING_FACTOR = 1.5  # factor to enlarge/pad eye detections
CENTER_SQUARE_WIDTH_PCT = 0.65  # alignment box width as a percentage of frame width
NUM_CANDIDATES = 15  # number of candidate crops to collect for autofocus
ROOT = "./datasets/autocapture"  # root where dataset is located
DATASET_NAME = "autocapture_test_videos"  # name of the dataset

#################### AUTOCAPTURE PIPELINE ####################
##############################################################


def detect_faces(frame: np.ndarray) -> Tuple[Sequence, np.ndarray]:
    """
    Function to detect faces with OpenCV CascadeClassifier

    Arguments:
        - frame: the opencv frame

    Returns:
        - faces: a list of face detections in normalized [x_tl, y_tl, w, h] coordinates
        - processing_frame: the preprocessed opencv frame, which can be reused
    """
    # resizing (preserving aspect ratio) and grayscaling
    h, w = frame.shape[:2]
    ar = w / h
    new_shape = (int(min(h, 256) * ar), min(h, 256))  # width, height
    processing_frame = cv2.resize(frame, new_shape, interpolation=cv2.INTER_CUBIC)
    processing_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
    # detect faces
    faces = CV2_FACE_CASCADE_MODEL.detectMultiScale(
        processing_frame, scaleFactor=1.1, minNeighbors=6
    )
    if len(faces) > 0:
        # only process one face
        faces = [faces[0]]
        faces[0] = [  # type: ignore
            faces[0][0] / new_shape[0],
            faces[0][1] / new_shape[1],
            faces[0][2] / new_shape[0],
            faces[0][3] / new_shape[1],
        ]
    return faces, processing_frame


def detect_eyes(
    frame: np.ndarray,
    processed: bool = False,
    padding_factor: float = PADDING_FACTOR,
) -> Tuple[Sequence, np.ndarray]:
    """
    Function to detect faces with OpenCV CascadeClassifier

    Arguments:
        - frame: the opencv frame
        - processed: indicator of whether or not the frame has already been preprocessed

    Returns:
        - eyes: a list of eye detections in normalized [x_tl, y_tl, w, h] coordinates
        - processing_frame: the preprocessed opencv frame, which can be reused
    """
    # pre-allocate
    processing_frame = frame
    h, w = frame.shape[:2]
    new_shape = (w, h)  # width, height
    # if frame has not already been preprocessed
    if not processed:
        # resizing (preserving aspect ratio) and grayscaling
        ar = w / h
        new_shape = (int(min(h, 256) * ar), min(h, 256))  # width, height
        processing_frame = cv2.resize(
            processing_frame, new_shape, interpolation=cv2.INTER_CUBIC
        )
        processing_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
    # detect eyes
    eyes = CV2_EYE_CASCADE_MODEL.detectMultiScale(
        processing_frame, scaleFactor=1.05, minNeighbors=6
    )
    if len(eyes) > 0:
        # for at most 2 eye detections, enlarge and normalize the coordinates
        padded_eyes = []
        for i, eye in enumerate(eyes[:2]):
            padded_eye = enlarge_box(eye, padding_factor=padding_factor)
            # normalize coordinates, ensuring that the detection is contained within the image
            padded_eye[0] = min(1.0, max(0.0, padded_eye[0] / new_shape[0]))  # type: ignore
            padded_eye[1] = min(1.0, max(0.0, padded_eye[1] / new_shape[1]))  # type: ignore
            padded_eye[2] = min(  # type: ignore
                1.0 - padded_eye[0], max(0.0, padded_eye[2] / new_shape[0])
            )
            padded_eye[3] = min(  # type: ignore
                1.0 - padded_eye[1], max(0.0, padded_eye[3] / new_shape[1])
            )
            padded_eyes.append(padded_eye)
        eyes = padded_eyes
    return eyes, processing_frame


def detect_faces_and_eyes(frame: np.ndarray) -> Tuple[Sequence, np.ndarray]:
    """
    Function to detect faces and eyes with OpenCV CascadeClassifier

    Arguments:
        - frame: the opencv frame

    Returns:
        - detections: a list of face and eye detections in normalized [x_tl, y_tl, w, h] coordinates
        - processing_frame: the preprocessed opencv frame, which can be reused
    """
    # detect faces and re-use th processing frame
    faces, processing_frame = detect_faces(frame)
    h_p, w_p = processing_frame.shape[:2]
    detections = []
    for face in faces:
        detections.append(face)
        # detect eyes on a face crop, efficiently
        face_eyes, _ = detect_eyes(
            processing_frame[
                int(face[1] * h_p) : int((face[1] + face[3]) * h_p),
                int(face[0] * w_p) : int((face[0] + face[2]) * w_p),
            ],
            processed=True,
        )
        eyes = []
        for face_eye in face_eyes:
            # eye detections are relative to face crop, so we add the offset
            eyes.append(
                [
                    (face[0] + (face_eye[0] * face[2])),
                    (face[1] + (face_eye[1] * face[3])),
                    (face_eye[2] * face[2]),
                    (face_eye[3] * face[3]),
                ]
            )
        # in face and eye mode, we must detect the face and both eyes for a complete detection
        if len(eyes) == 2:
            detections.extend(eyes)

    return detections, processing_frame


def display_detections(
    frame: np.ndarray,
    detections: Sequence,
    mode: str = AUTOCAPTURE_MODE,
    iou_thresh: float = IOU_THRESH,
    center_square_width: float = CENTER_SQUARE_WIDTH_PCT,
) -> Tuple[np.ndarray, Sequence[np.ndarray], Sequence[float]]:
    """
    Function to annotate a frame with detections and UI components

    Arguments:
        - frame: the opencv frame
        - detections: the list of detections in normalized [x_tl, y_tl, w, h] coordinates
        - mode: the autocapture mode ("face", "eye", "face and eye")
        - iou_thresh: the minimum IOU threshold for alignment with target box
        - center_square_width: alignment box width as a percentage of frame width

    Returns:
        - display_frame: a downsized copy of the input frame, with detections and UI components
        - crops: a list of high resolution crops
        - ious: the IOUs of detections with alignment box
    """
    # parameters for display
    display_height = 720
    fontScale = 0.75
    thickness = 2
    color = (0, 0, 255)
    text = f"Align {mode} here"
    fontFace = cv2.FONT_HERSHEY_SIMPLEX

    # resize (preserving aspect ratio)
    h, w, _ = frame.shape
    ar = w / h
    new_shape = (int(display_height * ar), display_height)  # width, height
    display_frame = cv2.resize(frame, new_shape, interpolation=cv2.INTER_CUBIC)

    # alignment box coordinates
    center_box_w = center_square_width * new_shape[0]
    center_box = (
        int((new_shape[0] // 2) - (center_box_w // 2)),
        int((new_shape[1] // 2) - (center_box_w // 2)),
        int(center_box_w),
        int(center_box_w),
    )

    # compute iou of each detection with alignment box
    ious = []
    if len(detections) > 0:
        for det in detections:
            box = (
                int(det[0] * new_shape[0]),
                int(det[1] * new_shape[1]),
                int(det[2] * new_shape[0]),
                int(det[3] * new_shape[1]),
            )
            iou = box_iou(center_box, box)  # type: ignore
            ious.append(iou)
            # draw detections
            display_frame = cv2.rectangle(
                display_frame, box, color=(255, 0, 0), thickness=thickness
            )
        # if the first box is aligned, then change box color and text prompt
        if ious[0] >= iou_thresh:
            color = (0, 255, 0)
            text = f"{mode.capitalize()} aligned!"
    display_frame = cv2.rectangle(
        display_frame, center_box, color=color, thickness=thickness
    )
    display_frame = cv2.putText(
        display_frame,
        text,
        org=(center_box[0], center_box[1] - 15),
        fontFace=fontFace,
        fontScale=fontScale,
        color=color,
        thickness=thickness,
    )

    # for each detection, extract the crop of the original image
    crops = []
    if len(detections) > 0 and ious[0] >= iou_thresh:
        for i, det in enumerate(detections):
            crops.append(
                frame[
                    int(det[1] * h) : int((det[1] + det[3]) * h),
                    int(det[0] * w) : int((det[0] + det[2]) * w),
                ]
            )
    return (
        display_frame,
        crops,
        ious,
    )


def autocapture(
    cap,
    mode: str = AUTOCAPTURE_MODE,
    fname: str = "test_image.jpg",
    iou_thresh: float = IOU_THRESH,
    num_candidates: int = NUM_CANDIDATES,
    rotate: bool = True,
):
    """
    Uses OpenCV to detect eyes and capture high resolution images

    Arguments:
        - cap: a cv2.VideoCapture object
        - mode: the autocapture mode ("face", "eye", "face and eye")
        - fname: the desired filename for the best saved crop
        - iou_thresh: the minimum IOU threshold for alignment with target box
        - num_candidates: the number of crops to consider in sharpness optimization
    """
    # choose the detection function
    if mode == "face":
        detection_fn = lambda frame: detect_faces(frame)
    elif mode == "eye":
        detection_fn = lambda frame: detect_eyes(frame)
    else:
        detection_fn = lambda frame: detect_faces_and_eyes(frame)

    candidate_saves = []
    while cap.isOpened():
        retval, frame = cap.read()

        # TEMPORARY: For some reason cv2 4.8 rotates images when passed through a videocapture object
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        # END TEMPORARY
        if retval:
            detections, _ = detection_fn(frame)
            display_frame, crops, ious = display_detections(
                frame, detections, mode=mode, iou_thresh=iou_thresh
            )
            if len(crops) > 0:
                candidate_saves.append(crops)
            waitKey = cv2.waitKey(1)
            if waitKey == ord("q") or len(candidate_saves) >= num_candidates:
                break
        else:
            break

    # save the best focused image
    save_sharpest_image(
        candidate_saves,
        mode=mode,
        fname=fname,
    )


def test_autocapture(
    mode: str = AUTOCAPTURE_MODE,
    iou_thresh: float = IOU_THRESH,
    root: str = ROOT,
    dataset_name: str = DATASET_NAME,
    rotate: bool = True,
):
    """
    Function to test autoapture functionality on all images/videos in a folder

    Arguments:
        - mode: the autocapture mode ("face", "eye", "face and eye")
        - root: the root where the dataset folder is located
        - dataset_name: the name of the dataset folder
        - rotate: indicator as to whether images should be rotated 90 degrees or not
    """
    dataset_root = os.path.join(root, dataset_name, "images")
    crop_root = os.path.join(root, f"{dataset_name}_crops", "images")

    # TEST IMAGES AND VIDEOS
    for fname in os.listdir(dataset_root):
        filetype = fname.split(".")[-1]
        cap = cv2.VideoCapture(os.path.join(dataset_root, fname))
        autocapture(
            cap,
            mode=mode,
            fname=os.path.join(
                crop_root,
                fname.replace(filetype, "png"),
            ),
            iou_thresh=iou_thresh,
            rotate=rotate,
        )
        # cv2.waitKey()
        cap.release()

    # cv2.destroyAllWindows()


def main(cap_source: str, fname: str) -> None:
    """
    Function to run autocapture pipeline on a given source

    Arguments:
        - cap_source: the data source type ("webcam", "image", "dataset")
        - fname: the file/folder name for data, if relevant
    """

    # Dataset source
    root, dataset_name = os.path.dirname(fname), os.path.basename(fname)
    if cap_source == "dataset":
        test_autocapture(root=root, dataset_name=dataset_name)
    else:
        crop_root = root.replace(
            os.path.basename(root), f"{os.path.basename(root)}_crops"
        )
        crop_fname = os.path.join(
            crop_root,
            dataset_name.split(".")[0] + ".png",
        )

        cap = None
        if cap_source == "webcam":
            cap = cv2.VideoCapture(0)
        elif cap_source == "image":
            cap = cv2.VideoCapture(fname)

        if cap is not None:
            autocapture(
                cap,
                fname=crop_fname,
            )
            cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="autocapture.py",
        description="R&D autocapture feature for iris",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--src",
        default="dataset",
        type=str,
        choices=["image", "dataset", "webcam"],
        help="AI pipeline runs on this type of data",
    )
    parser.add_argument(
        "--path",
        type=str,
        help="If specified, the exact source of data (can be a folder or file)",
    )
    ARGS = parser.parse_args()

    CAP_SOURCE = ARGS.src
    if CAP_SOURCE == "webcam":
        fname = "./datasets/autocapture/autocapture_test_videos/best.png"
    else:
        if ARGS.path is not None:
            fname = ARGS.path
        else:
            if CAP_SOURCE == "image":
                fname = "./datasets/autocapture/autocapture_test_samples/test_image.jpg"
            else:
                # fname = "./datasets/autocapture/autocapture_test_videos"
                fname = "./datasets/corneacare/combined"

    main(cap_source=CAP_SOURCE, fname=fname)
