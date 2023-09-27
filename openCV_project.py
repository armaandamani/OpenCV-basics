import librosa
import cv2
import argparse
import numpy as np
import os
import random
import time
from pygame import mixer as pgm
import own

ap = argparse.ArgumentParser()

ap.add_argument("-a", "--audio", required = True, help = "Path to audio file input")
ap.add_argument("-d", "--directory", required = True, help = "Path to directory with images for music video")
ap.add_argument("-s", "--save", required = True, help = "Path where you want to save the video... Folder will be created if it doesn't exist")

args = vars(ap.parse_args())

#Loads Audio file and returns values for amplitude and sample rate
audioFile = args["audio"]
(amp, sr) = librosa.load(audioFile, sr = 24000)

#Caluclates where the beats occur in the track, but in a weird format...
(tempo, beatFrame) = librosa.beat.beat_track(y = amp, sr = sr)

#Creates an array of timestaps at which each of the beats occur
beatTime = librosa.frames_to_time(beatFrame, sr = sr)

#initializes the frame rate array -- differences between values in beatTime
frameRateArray = np.diff(beatTime)

def loadImages(directory):
    images = []
    contours = []
    for filename in os.listdir(directory): #Loads and processes all of the images
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            r = 480 / image.shape[1] #image.shape[0] = height, image.shape[1] = width
            dim = (480, int(image.shape[0] * r)) #[width, height]
            resized = cv2.resize(image, dim)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edged = cv2.Canny(blurred, 45, 90)
            cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            images.append(resized)
            contours.append(cnts)
    return images, contours #Returns a list of the images and their derived contours from the specified directory

images, contours_list = loadImages(args["directory"])

newImages = []

for img, cnts in zip(images, contours_list): #Creates a black canvas with white outlining to then mask the original image over and create a cool-looking colored copy
    mask = np.zeros_like(img)
    for cnt in cnts:
        cv2.drawContours(mask, [cnt], -1, (255, 255, 255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    new = cv2.bitwise_and(img, img, mask = mask)
    newImages.append(new)



def Synchronize_Video(images, beatTime, filename, frameRateArray):
    frameRate = np.mean(np.diff(beatTime))
    frameRateArray = np.append(frameRateArray, frameRate) #Appends one more value so that frameratearray doesn't return an index error
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') #Writes the video
    
    max_height = max(img.shape[0] for img in images) #Sets values for video height and width to the biggest images from the Images list
    max_width = max(img.shape[1] for img in images)

    video = cv2.VideoWriter(filename, fourcc, frameRate, (max_width, max_height))
    
    selected_images = random.sample(images, 58) #Allows us to select 50 different images
    
    for img in selected_images:
        y = (max_height - img.shape[0]) // 2  # Center the height
        x = (max_width - img.shape[1]) // 2   # Center the width
        
        canvas = np.ones((max_height, max_width, 3), dtype='uint8') #Clear canvas to paste image atop
        canvas[y: y + img.shape[0], x: x + img.shape[1]] = img

        video.write(canvas)

    #cv2.destroyAllWindows
    video.release()
    print(f"Video is saved at {filename}")
    
    if not os.path.exists(filename):
        print(f"Failed to create the video at {filename}")

    return frameRateArray #For the play function


def Play_Video(filename, audiofile, frameRateArray):
    cap = cv2.VideoCapture(filename) #Identifies video
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {filename}")
        return
    
    pgm.init()
    pgm.music.load(audiofile) #Loads and plays audio
    pgm.music.play()
    
    while cap.isOpened():
        for i in range(len(frameRateArray)):
            ret, frame = cap.read() #Reads video and returns a value of True or False for ret, as well as the frames of the video
            if not ret:
                break
            cv2.imshow('Video', frame)

            wait_time = int(1000 * frameRateArray[i])
        
            cv2.waitKey(wait_time)

        cap.release() #Stops everything once video is done
        pgm.music.stop()
        cv2.destroyAllWindows()
        

#Establishing the proper file/folder paths to ensure easy use of the program
if not os.path.exists(args["save"]):
    os.makedirs(args["save"]) 

absolute = os.path.abspath(args["save"])
videoFilePath = os.path.join(absolute, "openCVProject.mp4")
if os.path.exists(videoFilePath):
    os.remove(videoFilePath)

#Calls the two functions
frameRateArray = Synchronize_Video(newImages, beatTime, videoFilePath, frameRateArray)
Play_Video(videoFilePath, audioFile, frameRateArray)