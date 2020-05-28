from pytube import YouTube
from tqdm import tqdm
import argparse
import os

parser = argparse.ArgumentParser(description='Download videos from youtube from urls in plain text file. (One url per line # for comments).')
parser.add_argument("-u", "--urls", type=str, default="videos.txt", help="The path to the plain text file for processing urls. default='videos.txt'")
parser.add_argument("-d", "--directory", type=str, default="videos", help="The path to the folder to save videos to (no closing slash). default='videos'")
parser.add_argument("-l", "--length", type=int, default=60*45, help="The max length of a video to download in seconds. default=2700")
parser.add_argument("-r", "--resolution", type=int, default=1080, help="The resolution of video to download. default=1080")
parser.add_argument("-f", "--fps", type=int, default=30, help="The fps of video to download. default=30")
args = parser.parse_args()
# Reference
# https://towardsdatascience.com/the-easiest-way-to-download-youtube-videos-using-python-2640958318ab

file = open(args.urls, 'r')
lines = file.readlines()
done = []
for i,line in enumerate(tqdm(lines, desc='Downloading', unit='video')):
    line = line.strip()
    sections = line.split("#")
    if len(sections) > 1:
        line = sections[0].strip()
    if len(line) <= 0:
        continue
    tqdm.write(line)
    if not line in done:
        name = "YouTube"
        while name == "YouTube":
            try:
                video = YouTube(line)
                name = video.title
                if name == "YouTube":
                    tqdm.write("Bad name")
                    continue
                tqdm.write('Video: "'+name+'"')
                if len(video.streams.filter(file_extension = "mp4").filter(res=str(args.resolution)+'p', fps=args.fps)) != 1:
                    for s in video.streams.filter(file_extension = "mp4").order_by('resolution'):
                        tqdm.write(str(s))
                else:
                    if(video.length <= args.length): # do not download if the video is more than 45 minutes
                        name = video.streams.filter(file_extension = "mp4",res=str(args.resolution)+'p', fps=args.fps)[0].download(output_path=args.directory)
                    else:
                        tqdm.write("Too long!")
                    done.append(line)
            except Exception as e:
                tqdm.write("oops "+ str(e))
    else:
        tqdm.write("Duplicate line "+str(i)+" \""+line+"\"")
