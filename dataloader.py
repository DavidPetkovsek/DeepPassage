import os
import cv2
import ast
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
import numpy as np

MANUAL_ERROR = 1
FRAME_START = 0
FRAME_END = 1
FRAME_CATEGORIES = 3
IMG_SIZE = 256
UNWANTED_CATEGORIES = ['WARNING']

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, default="videos", help="The path to the folder containing videos")
parser.add_argument("-o", "--output", type=str, default="data/scenes.tfrecord", help="Where the data will be saved")
parser.add_argument("-m", "--meta", type=str, default="meta.txt", help="The path to the meta file")
args = parser.parse_args()

scene_count = 0

def extract_scenes(path_to_video, meta_data, tfrecord_writer):
    def crop_square_center(width, height):
        dim = min(width, height)
        if dim == width:
            return (0, dim, (height - width) // 2, (height - width) // 2 + dim)
        else:
            return ((width - height) // 2, (width - height) // 2 + dim, 0, dim)

    global scene_count

    video_capture = cv2.VideoCapture(path_to_video)
    if not video_capture.isOpened():
        print(f"Unable to open {path_to_video}", file=sys.stderr)
        return

    ret, frame = video_capture.read()
    if not ret:
        raise Exception(f"Unable to fetch frame 0 of {path_to_video}")
    
    crop_box = crop_square_center(frame.shape[1], frame.shape[0])

    total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    print(f"Spitting {path_to_video}")
    for scene_meta_data in tqdm(meta_data):
        # Skip if contains unwated category
        skip = False
        for unwanted_category in UNWANTED_CATEGORIES:
            if  unwanted_category in meta_data[FRAME_CATEGORIES] and meta_data[FRAME_CATEGORIES][unwated_category]:
                skip = True
                break
        if skip:
            continue
        
        start_frame = scene_meta_data[FRAME_START]
        end_frame = scene_meta_data[FRAME_END]

        # Skip if the frames are invalid
        if start_frame > end_frame or \
            start_frame < 0 or \
            start_frame >= total_frames or \
            end_frame < 0 or \
            end_frame >= total_frames:
            continue

        video_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        scene_array = None
        for i in range(end_frame - start_frame - 1):
            ret, frame = video_capture.read()
            if not ret:
                raise Exception(f"Unable to fetch frame {start_frame + i} of {path_to_video}")
            frame = frame[crop_box[2]:crop_box[3], crop_box[0]:crop_box[1]]
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            if scene_array is None:
                scene_array = np.array(frame)[..., np.newaxis]
            else:
                scene_array = np.append(scene_array, frame[..., np.newaxis], axis=3)

        scene_count += 1
        shape_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=scene_array.shape))
        pixel_feature = tf.train.Feature(float_list=tf.train.FloatList(value=scene_array.reshape(-1)))
        scene_example = tf.train.Example(features=tf.train.Features(feature={
            'Shape': shape_feature,
            'Pixel': pixel_feature
        }))

        tfrecord_writer.write(scene_example.SerializeToString())
    f.close()


def main():
    f = open(args.meta, "r")
    tfrecord_writer = tf.io.TFRecordWriter(args.output)
    while True:
        video_file_name = f.readline().strip()
        if not video_file_name:
            break
        meta_data = ast.literal_eval(f.readline().strip())
        if not meta_data:
            break
        extract_scenes(os.path.join(args.dir, video_file_name), meta_data, tfrecord_writer)
    tfrecord_writer.close()
    f.close()


if __name__ == "__main__":
    main()
