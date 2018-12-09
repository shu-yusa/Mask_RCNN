import os
import numpy as np
import cv2
import glob
import moviepy.editor as mp

def extract_audio(org_video, input_video, output_video):
    clip_input = mp.VideoFileClip(org_video).subclip()
    clip_input.audio.write_audiofile('audio.mp3')

    # Add audio to output video.
    video_clip = mp.VideoFileClip(input_video)
    audio_clip = mp.AudioFileClip('audio.mp3')
    video_clip.audio = audio_clip
    video_clip.write_videofile(output_video)

processed = sorted(glob.glob("output/*.jpg"))

INPUT = 'PexelsVideos4809.mp4'
OUTPUT = 'video.mp4'

cap = cv2.VideoCapture(INPUT)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
# video = cv2.VideoWriter(OUTPUT, fourcc, fps, (width, height))
# for path in processed:
#     img = cv2.imread(path)
#     img = cv2.resize(img, (width, height))
#     video.write(img)
# video.release()

extract_audio(INPUT, OUTPUT, 'video_audio.mp4')
