import os
import sys
import argparse
import subprocess
import datetime
from glob import glob

import cv2
from tqdm import tqdm
import numpy as np 
# initiate the driver for cuda
import pycuda.autoinit

# import digits recognizer tensorrt wrapper
from digits import DigitRecognizer


def getFirstFrame(videofile):
    vidcap = cv2.VideoCapture(videofile)
    success, image = vidcap.read()
    if success:
        vidcap.release()
        return image


def getLastFrame(videofile):
    vidcap = cv2.VideoCapture(videofile)
    #set cap position to last frame
    vidcap.set(1, vidcap.get(7)-1) 
    success, image = vidcap.read()
    if success:
        vidcap.release()
        return image


def getVideoLength(videofile):
    vidcap = cv2.VideoCapture(videofile)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    totalFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(totalFrames/fps)
    total = str(datetime.timedelta(seconds=duration))
    hour = int(total.split(":")[0])
    minute = int(total.split(":")[1])
    second = int(total.split(":")[2])

    print("Total duration of video : {} : {} : {}".format(hour, minute, second))

    return hour, minute, second


def main(cfg):
    """
    Run the annotation pipeline for raw video frames in given directory
    """

    # check if the directory exists or not
    if not os.path.exists(cfg.videos_dir):
        raise FileExistsError("Directory for videos doesn't exists")
    if not os.path.exists(cfg.frames_dir):
        raise FileExistsError("Directory for frames doesn't exist..")
    if not os.path.exists(cfg.trt):
        raise FileExistsError("TensorRT model : {} nor found!".format(cfg.trt))

    # get the videos that we have to extract one frame per second
    all_videos = glob(cfg.videos_dir + "/*.mp4")

    if len(all_videos) == 0:
        print("There is no videos under given directory.")
        print("exiting...")
        sys.exit()

    # sort the videos
    all_videos.sort()

    # load the Digit recognizer TensorRT model to extract timestamp information from frame
    digits_trt = DigitRecognizer(engine_path=cfg.trt, input_size=(28, 28), num_classes=10)

    # get the optimal second that we want to extract from arguments
    sec = cfg.second

    # extract one frame per second for all videos
    for video in all_videos:
        print("processing on video : {}".format(video))
        video_abs_path = os.path.abspath(video)

        # grab metadata of the video
        # grab the first and last frame of the video to get the timestamp
        first_frame = getFirstFrame(video_abs_path)
        last_frame  = getLastFrame(video_abs_path)
        hour, minute, second = getVideoLength(video_abs_path)

        first_hh, first_mm, first_ss = digits_trt.classify(first_frame)
        start_time = "00:" + "00:" + str(90 - int(first_ss))
        end_time   = str(hour) + ":" + str(minute) + ":" + "30"

        # extract frame per minute with ffmpeg
        command = "ffmpeg -ss {:s} -i {:s} -to {:s} -vf fps=1/60 {:s}/img%04d.jpg".format(start_time, video_abs_path, end_time, cfg.dir)
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        if error:
            print("Error happened while extracting frames from video using ffmpeg,")
            print("error : {}".format(error))
            sys.exit()

        # list all the image files under give directory
        video_frames = glob(cfg.frames_dir + "/*.jpg")
        # if there is no images under given directory,
        # print warning
        if len(video_frames) == 0:
            print("There is no images under given directory.")
            print("exiting.....")
            continue

        video_frames.sort()

        for image_pth in tqdm(video_frames):

            _name = image_pth.split("/")[-1]
            _initial = _name.split(".")[0][:3]
            if _initial == "img":
                try:
                    image_data = cv2.imread(image_pth)

                    # perform inference | extract timestamp
                    hh, mm, _ = digits_trt.classify(image_data)
                    
                    rename_str = cfg.frames_dir + "/" + str(hh) + ":" + str(mm) + ":" + "00" + ".jpg"
                    if not os.path.exists(rename_str):
                        os.rename(image_pth, rename_str)
                    else:
                        print("duplicate found! Data is removed.")
                        os.remove(image_pth)

                except:
                    print("Error happened while processing the video frames.")
                    continue
            else:
                continue


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trt", type=str, default="checkpoints/digits_6.trt",
                        help="Generated TensorRT runtime model")
    parser.add_argument("-vd", "--videos_dir", type=str, default="/home/htut/Desktop/SYN_CV/data/SYN-CI-F1-16/videos")
    parser.add_argument("-fd", "--frames_dir", type=str, default="/home/htut/Desktop/SYN_CV/data/SYN-CI-F1-16/frames")
    parser.add_argument("-s", "--second", type=str, default="00")

    return parser.parse_args()

if __name__ == "__main__":
    """
    Parse arguments and pass it to main function
    """

    cfg = parse_args()
    print(cfg)
    main(cfg)