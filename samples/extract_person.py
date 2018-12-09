import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2
import glob
import moviepy.editor as mp

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


def extract_audio(input_video, output_video):
    clip_input = mp.VideoFileClip(input_video).subclip()
    clip_input.audio.write_audiofile('audio.mp3')


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


processed = sorted(glob.glob("output/*.jpg"))
len_processed = len(processed)

INPUT = 'PexelsVideos4809.mp4'
OUTPUT = 'video.mp4'
cap = cv2.VideoCapture(INPUT)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)
fps = cap.get(cv2.CAP_PROP_FPS)


fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
video = cv2.VideoWriter(OUTPUT, fourcc, fps, size)
i = 0
while(cap.isOpened()):
    flag, frame = cap.read()  # Capture frame-by-frame
    if flag == False:  # Is a frame left?
        break

    if not i < len_processed:
        # Resume from the last time
        # Convert color
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Run detection
        results = model.detect([frame], verbose=1)

        # Visualize results
        r = results[0]
        img = visualize.save_masked_image(frame, r['rois'], r['masks'], r['class_ids'], 
                                          class_names, r['scores'],
                                          sequence=i, extract_class_id=1)
        img = cv2.resize(img, size)
        video.write(img)
    i += 1
video.release()

# extract_audio(INPUT, OUTPUT)
