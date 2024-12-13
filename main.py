import cv2
import os
from ultralytics import YOLO
import pandas as pd
import numpy as np
from tqdm import tqdm  # For progress bar
from datetime import datetime

# Load the YOLO model (using yolov8m.pt)
model = YOLO('yolov8m.pt')  # Use 'yolov8n.pt' for the nano version

# Getting names from classes
dict_classes = model.model.names

# Function to resize frames
def resize_frame(frame, scale_percent):
    """Function to resize an image by a percentage scale"""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    return resized

# Configurations
scale_percent = 100  # Change this if you want to resize frames

# Objects to detect (YOLO class IDs for vehicles)
class_IDS = [1, 2, 3, 4, 5, 7]  # Update based on your YOLO model classes

# Function to calculate estimated time to clear traffic (based on vehicle count)
def calculate_clear_time(total_vehicles):
    # Assume an arbitrary time per vehicle (e.g., 5 seconds to clear each vehicle)
    time_per_vehicle = 5  # seconds
    total_time = total_vehicles * time_per_vehicle
    minutes, seconds = divmod(total_time, 60)
    return f"{int(minutes)} min {int(seconds)} sec"

# Function to process frames with YOLOv8
def process_frame_with_yolo(frame, frame_count, output_folder):
    # Resize the frame if necessary
    frame = resize_frame(frame, scale_percent)

    # Get predictions (lower confidence threshold for better detection of smaller objects)
    y_hat = model.predict(frame, conf=0.5, classes=class_IDS, device='cpu', verbose=False)

    # Bounding boxes, confidence, and classes of detected objects
    boxes = y_hat[0].boxes.xyxy.cpu().numpy()
    classes = y_hat[0].boxes.cls.cpu().numpy()

    # Store information in a dataframe
    positions_frame = pd.DataFrame(np.hstack([boxes, classes[:, None]]),
                                   columns=['xmin', 'ymin', 'xmax', 'ymax', 'class'])

    # Count all detected vehicles
    total_vehicles = len(positions_frame)

    # Draw bounding boxes and count vehicles
    for ix, row in enumerate(positions_frame.iterrows()):
        xmin, ymin, xmax, ymax, category = row[1].astype('int')

        # Calculate center of the bounding-box
        center_x, center_y = int((xmax + xmin) / 2), int((ymax + ymin) / 2)

        # Draw bounding-box and center
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
        # Display only the class name (without the confidence score)
        cv2.putText(frame, f'{dict_classes[category]}',
                    (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display total vehicle count on the frame
    cv2.putText(frame, f'Total Vehicles: {total_vehicles}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Calculate and display the time slot to clear vehicles
    clear_time = calculate_clear_time(total_vehicles)
    cv2.putText(frame, f'Time to Clear: {clear_time}', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save the processed frame to the output folder
    output_frame_path = os.path.join(output_folder, f'processed_frame_{frame_count:04d}.jpg')
    cv2.imwrite(output_frame_path, frame)

    return frame  # Return the processed frame for video creation

# Function to convert processed frames into a video
def frames_to_video(output_folder, output_video_path, fps=25):
    """Convert processed frames in a folder to a video."""
    frame_files = [f for f in os.listdir(output_folder) if f.endswith('.jpg')]
    frame_files.sort()  # Ensure frames are in the correct order

    if len(frame_files) == 0:
        return

    # Read the first frame to get video properties
    first_frame_path = os.path.join(output_folder, frame_files[0])
    frame = cv2.imread(first_frame_path)
    height, width, _ = frame.shape

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(output_folder, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    video_writer.release()

# Function to extract frames from a video and process them with YOLO
def video_to_frames_and_process(video_path, output_folder):
    """Function to extract frames from a video, process them with YOLO, and save processed frames."""
    os.makedirs(output_folder, exist_ok=True)  # Create output folder if it doesn't exist
    cap = cv2.VideoCapture(video_path)  # Capture the video
    if not cap.isOpened():
        return  # Exit early if video can't be opened

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames
    processed_frames = 0

    # Create a progress bar for frame processing
    with tqdm(total=frame_count, desc=f'Processing {os.path.basename(video_path)}', ncols=100) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Break if no more frames

            # Process the frame with YOLO and save the processed frame
            process_frame_with_yolo(frame, processed_frames, output_folder)
            processed_frames += 1

            # Update the progress bar
            pbar.update(1)

    cap.release()  # Release video capture

    # Convert processed frames to video
    output_video_path = os.path.join(output_folder, f'{os.path.basename(video_path).split(".")[0]}_processed.mp4')
    frames_to_video(output_folder, output_video_path)

if _name_ == "_main_":
    video_folder = 'data/input_videos/'  # Folder containing input videos
    output_base_folder = 'data/output_videos/'  # Base folder to save processed frames

    # Ensure output folder exists
    os.makedirs(output_base_folder, exist_ok=True)

    # Start timer
    start_time = datetime.now()

    # Loop through each video in the video folder
    for video_file in os.listdir(video_folder):
        if video_file.endswith('.mp4'):  # Process only .mp4 files (you can adjust for other formats)
            video_path = os.path.join(video_folder, video_file)
            video_name = os.path.splitext(video_file)[0]  # Get the video file name without extension
            output_folder = os.path.join(output_base_folder, video_name)  # Create a folder for each video

            video_to_frames_and_process(video_path, output_folder)  # Process each video

    # End timer and print the elapsed time
    end_time = datetime.now()
    elapsed_time = end_time - start_time