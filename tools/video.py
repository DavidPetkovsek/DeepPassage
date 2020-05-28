from pytube import YouTube# misc
import os
import shutil
import math
import datetime# plots
import matplotlib.pyplot as plt
# %matplotlib inline# image operation
import cv2
from tqdm import tqdm

# https://towardsdatascience.com/the-easiest-way-to-download-youtube-videos-using-python-2640958318ab

file = open('videos.txt', 'r')
lines = file.readlines()

done = []

for i,line in enumerate(lines):
    line = line.strip()
    print(line)
    if not line in done:
        name = "YouTube"
        while name == "YouTube":
            try:
                video = YouTube(line)
                name = video.title
                if name == "YouTube":
                    continue
                print('Video: "'+name+'"')
                if len(video.streams.filter(file_extension = "mp4").filter(res='1080p', fps=30)) != 1:
                    for s in video.streams.filter(file_extension = "mp4").order_by('resolution'):
                        print(str(s))
                else:
                    name = video.streams.filter(file_extension = "mp4",res='1080p',fps=30)[0].download(output_path='videos')
                    done.append(line)
            except Exception as e:
                print("oops", e)
    else:
        print("Duplicate line \""+line+"\"")
