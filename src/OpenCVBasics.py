import argparse
import os
import random
import time
import numpy as np
import cv2
import librosa
from pygame import mixer as pgm
import own  # Assuming 'own' is a custom module for contour detection

def parse_arguments():
    """Parse command-line arguments for audio, image directory, and save path."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--audio", required=True, help="Path to audio file input")
    parser.add_argument("-d", "--directory", required=True, help="Path to directory with images for music video")
    parser.add_argument("-s", "--save", required=True, help="Path where you want to save the video; folder will be created if it doesn't exist")
    return parser.parse_args()

def load_audio(audio_path):
    """Load an audio file and return amplitude and sample rate."""
    return librosa.load(audio_path, sr=24000)

def detect_beats(amplitude, sample_rate):
    """Detect beats in audio and calculate beat times and frame rates."""
    tempo, beat_frames = librosa.beat.beat_track(y=amplitude, sr=sample_rate)
    beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate)
    frame_rate_array = np.diff(beat_times)
    frame_rate = np.mean(frame_rate_array)
    frame_rate_array = np.append(frame_rate_array, frame_rate)  # Match array length
    return beat_times, frame_rate_array, frame_rate

def load_and_process_images(directory):
    """Load images from a directory and process them to extract contours."""
    images = []
    contours_list = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                continue
            r = 480 / image.shape[1]
            dim = (480, int(image.shape[0] * r))
            resized = cv2.resize(image, dim)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edged = cv2.Canny(blurred, 45, 90)
            contours = own.contours(cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))
            images.append(resized)
            contours_list.append(contours)
    return images, contours_list

def create_masked_images(images, contours_list):
    """Create masked images with white outlines from original images and contours."""
    processed_images = []
    for image, contours in zip(images, contours_list):
        mask = np.zeros_like(image)
        for contour in contours:
            cv2.drawContours(mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        processed_images.append(masked_image)
    return processed_images

def create_video(images, output_path, frame_rate):
    """Generate a video from a list of images and save it to the specified path."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    video = cv2.VideoWriter(output_path, fourcc, frame_rate, (max_width, max_height))
    
    selected_images = random.sample(images, min(50, len(images)))  # Select up to 50 images
    
    for image in selected_images:
        y_offset = (max_height - image.shape[0]) // 2
        x_offset = (max_width - image.shape[1]) // 2
        canvas = np.ones((max_height, max_width, 3), dtype='uint8') * 255  # White background
        canvas[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image
        video.write(canvas)
    
    video.release()
    print(f"Video saved at {output_path}")
    if not os.path.exists(output_path):
        print(f"Failed to create video at {output_path}")

def play_synced_video(video_path, audio_path, beat_times, frame_rate_array):
    """Play a video synchronized with audio beats."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    pgm.init()
    pgm.music.load(audio_path)
    pgm.music.play()
    
    for i in range(len(frame_rate_array)):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video', frame)
        wait_time = int(1000 * frame_rate_array[i])
        cv2.waitKey(wait_time)
    
    cap.release()
    pgm.music.stop()
    cv2.destroyAllWindows()

def main():
    """Main function to orchestrate audio-video synchronization and video creation."""
    args = parse_arguments()
    
    # Load and process audio
    amplitude, sample_rate = load_audio(args.audio)
    beat_times, frame_rate_array, frame_rate = detect_beats(amplitude, sample_rate)
    
    # Load and process images
    images, contours_list = load_and_process_images(args.directory)
    processed_images = create_masked_images(images, contours_list)
    
    # Set up save directory and video path
    save_dir = args.save
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    video_file_path = os.path.join(os.path.abspath(save_dir), "open_cv_project.mp4")
    if os.path.exists(video_file_path):
        os.remove(video_file_path)
    
    # Create and play the video
    create_video(processed_images, video_file_path, frame_rate)
    play_synced_video(video_file_path, args.audio, beat_times, frame_rate_array)

if __name__ == "__main__":
    main()
