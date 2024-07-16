'''
Credits: https://www.kaggle.com/discussions/general/491148
'''

import cv2
import os

def convert_video_to_images(input_video: os.PathLike, output_folder: os.PathLike):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video_capture = cv2.VideoCapture(input_video)
    success, frame = video_capture.read()
    count = 0

    # Read each frame and save it as an image
    while success:
        image_path = os.path.join(output_folder, f"frame_{count:04d}.jpg")  # Adjust the format as per your requirement
        cv2.imwrite(image_path, frame)  # Save the frame as an image
        success, frame = video_capture.read()  # Read next frame
        count += 1

    # Release the video capture object
    video_capture.release()

def convert_fisheye():
    # TODO: Implement fisheye to equi-rectangular conversion.
    pass