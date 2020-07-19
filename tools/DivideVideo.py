import numpy as np
import argparse
import cv2
import re, os
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Opens all videos and provides a suite to produce meta files to breakup and categorize clips of videos.')
parser.add_argument("-m", "--meta", type=str, default="meta.txt", help="The name of the meta file. default='videos.txt'")
parser.add_argument("-o", "--open", type=str, default="videos", help="The path to the folder to open videos to (no closing slash). default='videos'")
parser.add_argument("-s", "--save", type=str, default="videos", help="The path to the folder to save videos and meta to (no closing slash). default='videos'")
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

box = [-1,-1,-1,-1]
drawing = False
def handleMouse(event, x, y, flags, param):
    global box, drawing
    x = max(min(x, 1920), 0)
    y = max(min(y, 1080), 0)
    if event == cv2.EVENT_LBUTTONDOWN:
        box[0] = x
        box[1] = y
        drawing = True
        print(box , end='      \r')
    elif event == cv2.EVENT_LBUTTONUP:
        box[2] = x
        box[3] = y
        drawing = False
        print(box , end='      \r')
    elif event == cv2.EVENT_RBUTTONUP or event == cv2.EVENT_RBUTTONDOWN:
        box = [-1,-1,-1,-1]
        drawing = False
        print(box , end='      \r')
    if drawing:
        box[2] = x
        box[3] = y

masterData = {}
data = []
files = getListOfFiles(args.open, r"\.mp4$")
speed = 2
img = None
for file in files:
    cap = cv2.VideoCapture(file[0])
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1
    pos = 0
    k = 0
    cats = ['still cam', 'moving cam', 'fast moving cam', 'WARNING', 'stars', 'storms', 'reflection', 'lightning', 'night', 'day', 'realtime', 'timelapse', 'water', 'forest', 'cloud', 'mountain', 'nothern light', 'city']
    categories = {x:False for x in cats}
    btnImg = np.full((20*((len(categories)+2)//3),400,3), 255, dtype=np.uint8)
    def display():
        global img, box
        # if not img is None:
        img2 = np.copy(img)
        if not -1 in box:
            cv2.rectangle(img2,(min(box[0], box[2]),min(box[1], box[3])),(max(box[0], box[2]),max(box[1], box[3])),(50,200,0),2)
        cv2.imshow(file[1], img2)

    def onChange(trackbarValue):
        global pos, k, img
        pos = min(max(trackbarValue, 0), length)
        cap.set(cv2.CAP_PROP_POS_FRAMES,min(max(pos, 0),length))
        err,img = cap.read()
        display()
        q = cv2.waitKey(1) & 0xff
        if q != 255 and k == ord('k'):
            k = 0 if q == ord('k') else q

    def clearCategories():
        global categories, btnImg, cats
        categories = {x:False for x in cats}
        for i,c in enumerate(cats):
            cv2.rectangle(btnImg, (btnImg.shape[1]//3*(i%3)+1,i//3*20+1), (btnImg.shape[1]//3*(i%3)+5, (i//3+1)*20-1), (0,0,255), -1)
        cv2.imshow('control', btnImg)
    cv2.namedWindow(file[1], cv2.WINDOW_NORMAL)
    cv2.namedWindow('control', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('Position','control', 0, length, onChange)
    cv2.line(btnImg, (0,0),(0,btnImg.shape[0]), (0,0,0), 1)
    cv2.line(btnImg, (btnImg.shape[1]//3,0),(btnImg.shape[1]//3,btnImg.shape[0]), (0,0,0), 1)
    cv2.line(btnImg, (btnImg.shape[1]//3*2,0),(btnImg.shape[1]//3*2,btnImg.shape[0]), (0,0,0), 1)
    for i,c in enumerate(cats):
        cv2.putText(btnImg, c, (btnImg.shape[1]//3*(i%3)+30, 20*(i//3)+13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1, cv2.LINE_AA)
        cv2.line(btnImg, (0,i*20), (btnImg.shape[1],i*20), (0,0,0), 1)
        cv2.rectangle(btnImg, (btnImg.shape[1]//3*(i%3)+1,i//3*20+1), (btnImg.shape[1]//3*(i%3)+5, (i//3+1)*20-1), (0,0,255), -1)
    def buttons(event, x, y, flags, param):
        global categories
        if event == cv2.EVENT_LBUTTONDOWN:
            x = x//(btnImg.shape[1]//3)
            y = y//20
            if x+y*3 < len(cats):
                categories[cats[x+y*3]] = not categories[cats[x+y*3]]
                col = (0,255,0) if categories[cats[x+y*3]] else (0,0,255)
                cv2.floodFill(btnImg, None, seedPoint=(5+x*(btnImg.shape[1]//3), 5+y*20), newVal=col, loDiff=(30,30,30,30),upDiff=(30,30,30,30))
                cv2.imshow('control', btnImg)
    cv2.imshow('control', btnImg)
    cv2.setMouseCallback(file[1], handleMouse)
    cv2.setMouseCallback('control', buttons)
    onChange(0)
    lastS = 0

    while cap.isOpened():
        if k == ord('k'):
            for i in range(speed):
                if cap.isOpened():
                    oImg = np.copy(img)
                    err,img = cap.read()
                    if not err:
                        img = np.copy(oImg)
                        print('reached end')
                        k = 0
                        break
                else:
                    pos = min(max(pos-1, 0),length)
            display()
            q = cv2.waitKey(max(int(1/30),1)) & 0xff
            if q != 255:
                k = 0 if q == ord('k') else q
                cv2.setTrackbarPos('Position', 'control', int(cap.get(cv2.CAP_PROP_POS_FRAMES))-speed)
        while not k in [27]+[ord(x) for x in 'qQjlJLk,.']:
            if k == ord('s'):
                lastS = pos
            elif k == ord('d'):
                betterBox = [min(box[0], box[2]),min(box[1], box[3]),max(box[0], box[2]),max(box[1], box[3])]
                data.append([lastS, pos, betterBox, categories])
                clearCategories()
                print('added',[lastS, pos])
                lastS = min(pos+1, length)
                k = ord('l')
                break
            elif k == ord('r'):
                cv2.setTrackbarPos('Position', 'control', data[-1][1])
                lastS = data[-1][0]
                print('removed',data.pop(-1)[:2])
            elif k == ord('p'):
                print('Start Setting:',lastS)
                print([d[:2] for d in data])
            display()
            k = cv2.waitKey(10) & 0xff

        if k==27 or k == ord('q') or k == ord('Q'):
            break
        elif k == ord('j') or k == ord(','):
            cv2.setTrackbarPos('Position', 'control', pos-1-speed)
        elif k == ord('J'):
            cv2.setTrackbarPos('Position', 'control', pos-speed-speed*2)
        elif k == ord('L'):
            cv2.setTrackbarPos('Position', 'control', pos+speed*2-speed)
        elif k == ord('l') or k == ord('.'):
            cv2.setTrackbarPos('Position', 'control', pos-speed+1)
        if k != ord('k'):
            k = 0
        pos = min(max(pos+speed, 0),length)
    clearCategories()
    cap.release()
    cv2.destroyAllWindows()
    print('--------------------------------------')
    print(file[1])
    print(data)
    masterData[file[1]] = data
    data = []
    box = [-1,-1,-1,-1]
    if k==27 or k == ord('Q'):
        break
lines = []
for d in masterData:
    lines.append(str(d))
    lines.append(str(masterData[d]))
f = open(args.save+'/'+args.meta, "w")
f.write('\n'.join(lines))
f.close()
print('\nwrote to file')
