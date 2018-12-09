#!/bin/bash

FFMPEG=$HOME/.imageio/ffmpeg/ffmpeg-linux64-v3.3.1 

$FFMPEG -i video.mp4 -i audio.mp3 -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 video_audio.mp4

