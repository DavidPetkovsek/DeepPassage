import numpy as np
import argparse
import cv2
import re, os, pathlib
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Resizes videos by cropping at the centre then resizing')
parser.add_argument("-W", "--width", type=int, default=256, help="The width of the new video. default=256")
parser.add_argument("-H", "--height", type=int, default=256, help="The height of the new video. default=256")
parser.add_argument("-f", "--fps", type=int, default=30, help="The fps of video to download. default=30")
parser.add_argument("-o", "--open", type=str, default="videos", help="The path to the folder to open videos to (no closing slash). default='videos'")
parser.add_argument("-s", "--save", type=str, default="dataset", help="The path to the folder to save videos to (no closing slash). default='dataset'")
args = parser.parse_args()

'''
    For the given path, get the List of all files in the directory tree
'''
def getListOfFiles(dirName, regex=None):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = []
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        # if os.path.isdir(fullPath):
        #     allFiles = allFiles + getListOfFiles(fullPath)
        # else:
        if not os.path.isdir(fullPath):
            if not regex is None:
                if re.search(regex, fullPath):
                    allFiles.append([fullPath,entry])
            else:
                allFiles.append([fullPath,entry])

    return allFiles

files = getListOfFiles(args.open, r"\.mp4$")
for file in tqdm(files):
    cap = cv2.VideoCapture(file[0])
    out = cv2.VideoWriter(str(pathlib.Path().absolute())+'\\'+args.save+'\\'+file[1], cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (args.width,args.height))

    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        # crop the image
        x = 0
        y = 0
        w = frame.shape[1]
        h = frame.shape[0]
        if args.width/args.height > frame.shape[1]/frame.shape[0]:
            nh = int(w*args.height/args.width)
            y = int(h/2)-int(nh/2)
            h = nh
        elif args.width/args.height < frame.shape[1]/frame.shape[0]:
            nw = int(h*args.width/args.height)
            x = int(w/2)-int(nw/2)
            w = nw
        croppedFrame = frame[y:y+h, x:x+w]

        # resize the image
        finalFrame = cv2.resize(croppedFrame, (args.width, args.height))

        cv2.imshow("Frame", finalFrame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # write the frame
        out.write(finalFrame)

    cap.release()
    out.release()
cv2.destroyAllWindows()
