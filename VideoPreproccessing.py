import json
import os
import cv2
from pytube import YouTube
from moviepy.editor import VideoFileClip
import shutil
import re

CURRENT_PATH = os.getcwd()
VIDEO_PATH = os.path.join(os.getcwd(), 'videos')
TRAIN_CLIP_PATH = os.path.join(os.getcwd(), 'videos_train')
VAL_CLIP_PATH = os.path.join(os.getcwd(), 'videos_val')

if not os.path.exists(VIDEO_PATH):
    os.makedirs(VIDEO_PATH)

with open("MSASL_classes.json", "r") as file:
    classes = json.load(file)
    classes = classes[:-600]

with open("MSASL_train.json", "r") as file:
    train_file = json.load(file)

with open("MSASL_val.json", "r") as file:
    val_file = json.load(file)

train_data = []
val_data = []
for i in range(len(train_file)):
    if train_file[i]["label"] < 400:
        train_data.append(train_file[i])

for i in range(len(val_file)):
    if val_file[i]["label"] < 400:
        val_data.append(val_file[i])


def Downloader():
    total_data = train_data + val_data
    videos = []
    for i in range(len(total_data)):
        vid = (total_data[i]["file"], total_data[i]["url"])
        videos.append(vid)
    
    videos = list(dict.fromkeys(videos))

    downloaded = os.listdir(VIDEO_PATH)

    Download(videos, downloaded)

def Download(videos, downloaded):
    for i in range(len(videos)):
        youtubeObject = YouTube(videos[i][1])
        try:
            if videos[i][0] + '.mp4' not in downloaded:
                print("Downloading:" + videos[i][0])
                youtubeObject = youtubeObject.streams.get_highest_resolution()
                youtubeObject.download(output_path=VIDEO_PATH, filename=videos[i][0] + '.mp4')
                print("Download is completed successfully")
        except Exception as e:
            print("An error has occurred: " + str(e))

        print(str(len(videos) - i) + " videos left to download")

def VideoTrimmer(CLIP_PATH, data):
    for i in range(len(classes)):
        try:
            os.makedirs(os.path.join(CLIP_PATH, classes[i]))
        except:
            continue
        
    videos = os.listdir(VIDEO_PATH)
    errors = 0
    for video in videos:        
        i = 0
        k = len(data)
        while i < k:
            if video[:-4] == data[i]["file"]:
                try:
                    clip = VideoFileClip(os.path.join(VIDEO_PATH, video))
                    clip = clip.subclip(data[i]['start_time'], data[i]['end_time'])
                    os.chdir(os.path.join(CLIP_PATH, classes[data[i]["label"]]))
                    clip.write_videofile(f"{len(os.listdir(os.getcwd()))}.mp4", audio=False)
                    clip.close()
                    data.pop(i)
                    k -= 1
                    i -= 1
                except Exception as e:
                    print("An error has occurred: " + str(e))
                    print("Error in video: " + video)
                    errors += 1

                os.chdir(CURRENT_PATH)
            i += 1

    print("Errors: " + str(errors))

def FrameRateGeneralization(CLIP_PATH):
    folders = os.listdir(CLIP_PATH)
    min_frames = 30
    max_frames = 120

    for folder in folders:
        files = os.listdir(os.path.join(CLIP_PATH, folder))
        if len(files) == 0:
            print(folder + " is empty")

        for file in files:
            cap = cv2.VideoCapture(os.path.join(CLIP_PATH, folder, file))
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            cv2.destroyAllWindows()

            if frames < min_frames:
                os.remove(os.path.join(CLIP_PATH, folder, file)) 
            if frames > max_frames:
                os.remove(os.path.join(CLIP_PATH, folder, file))

def VideoRenamer(CLIP_PATH):
    folders = os.listdir(CLIP_PATH)
    for folder in folders:
        files = os.listdir(os.path.join(CLIP_PATH, folder))
        files = sorted(files, key=custom_sort)
        for i in range(len(files)):
            os.rename(os.path.join(CLIP_PATH, folder, files[i]), os.path.join(CLIP_PATH, folder, f"{i}.mp4"))
    
def VideoMultiplier(video_count, CLIP_PATH):
    folders = os.listdir(CLIP_PATH)
    for folder in folders:
        files = os.listdir(os.path.join(CLIP_PATH, folder))
        files = sorted(files, key=custom_sort)
        i = 0
        while len(files) < video_count and len(files) != 0:
            shutil.copy(os.path.join(CLIP_PATH, folder, files[i]), os.path.join(CLIP_PATH, folder, f"{len(files)}.mp4"))
            i += 1
            files = os.listdir(os.path.join(CLIP_PATH, folder))
            files = sorted(files, key=custom_sort)

def FolderRemover(CLIP_PATH):
    folders = os.listdir(CLIP_PATH)
    for folder in folders:
        if len(os.listdir(os.path.join(CLIP_PATH, folder))) == 0:
            os.rmdir(os.path.join(CLIP_PATH, folder))

def custom_sort(s):
    match = re.search(r'(\d+)', s)
    if match:
        return int(match.group())
    else:
        return float('inf')


if __name__ == "__main__":
    while True:
        user_input = input("Have you downloaded the necessary videos? (y/n): ")
        if user_input == 'n' or user_input == 'y':
            break
        else:
            print("Invalid input. Please enter 'y' or 'n'")

    if user_input == 'n':
        Downloader()
        print("Videos Downloaded Successfully")
        print("Please run the script again to continue to video preproccessing")

    if user_input == 'y':
        print("Continuing to video preproccessing...")
        print("Cutting videos into clips...")
        VideoTrimmer(TRAIN_CLIP_PATH, train_data)
        VideoTrimmer(VAL_CLIP_PATH, val_data)
        
        print("Generalizing frame rates...")
        FrameRateGeneralization(TRAIN_CLIP_PATH)
        FrameRateGeneralization(VAL_CLIP_PATH)

        print("Fixing videos name order...")
        VideoRenamer(TRAIN_CLIP_PATH)
        VideoRenamer(VAL_CLIP_PATH)

        print("Multiplying videos for better training...")
        VideoMultiplier(30, TRAIN_CLIP_PATH)
        VideoMultiplier(30, VAL_CLIP_PATH)

        print("Removing empty folders...")
        FolderRemover(TRAIN_CLIP_PATH)
        FolderRemover(VAL_CLIP_PATH)

        print("Video preproccessing is completed successfully")

