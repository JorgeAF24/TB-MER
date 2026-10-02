import os
# Import OpenCV (Open Source Computer Vision Library)
# Used for video processing and image handling
import cv2
import numpy as np
import argparse
import pandas as pd

def extract_frames_sequence(video_path, save_path, sampling_mode="fps", target_fps=4, num_frames=8, start_time=None, end_time=None):
    """
    Extract T uniformly sampled frames from a video.

    Args:
        video_path (str): Path to input video
        save_dir (str): Directory to save frames
        T (int): Number of frames to extract
        start_time (float): Start time for frame extraction
        end_time (float): End time for frame extraction
    """
    # Use VideoCapture to read the video file
    # cap being the video capture object, which allows video manipulation
    cap = cv2.VideoCapture(video_path)
    # Check if the video was opened successfully
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        return
    # Get the total number of frames in the video
    # The frames in the video depend on FPS (Frames Per Second) and duration
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    # If start_time or end_time is not provided, default to the entire video duration
    # MELD: entire video, IEMOCAP: utterance duration
    if start_time is None or end_time is None:
        start_frame = 0
        end_frame = frame_count
    else:
        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps)
    
    if sampling_mode == "fps":
        # Calculate the rate of each frame in the video (frames per second) and determine the step size to sample frames at the target rate
        step = video_fps/target_fps

        indices = []
        current = float(start_frame)
        while int(current) < end_frame:
            indices.append(int(current))
            current += step
    elif sampling_mode == "uniform":
        # For uniform sampling
        available_frames = np.arange(start_frame, end_frame)

        if len(available_frames) == 0:
            cap.release()
            return

        indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    else:
        raise ValueError("Invalid sampling mode. Choose 'fps' or 'uniform'.")


    # DEBUG
    #print(f"[DEBUG] {video_path}")
    #print("FPS:", video_fps)
    #print("Step:", step)
    #print("Extracted frames:", len(indices))
    #print("First indices:", indices[:5])
    
    # Ensure the directory exists before saving
    os.makedirs(save_path, exist_ok=True)

    # Initialize a counter for the number of frames saved and a variable to keep track of the last valid frame index
    saved = 0
    
    # Use enumarate function that returns an iterable of pairs (index, value) from the indices list
    for i,idx in enumerate(indices):
        # Set the video position to the specified frame index
        # cv2.CAP_PROP_POS_FRAMES represents the current position of the video file in terms of frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        # Read the frame at the current position
        # ret is a boolean indicating if the frame was read successfully, and frame is the actual image data
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: could not read frame {idx} in {video_path}")
            continue

        # Save the extracted frame as a JPEG image in the specified save path
        frame_path = os.path.join(save_path, f"frame_{i}.jpg")
        cv2.imwrite(frame_path, frame)
        saved += 1
    
    cap.release()

    if saved == 0:
        print(f"[ERROR] No valid frames in {video_path}")
        return

    final_count = len(os.listdir(save_path))
    print(f"[INFO] {video_path} → extracted {final_count} frames")

def process_meld_split(split, raw_dir, frames_dir, sampling_mode, target_fps, num_frames):
    """
    Process all videos in a split (train/dev/test)
    Args:
        split (str): dataset split ("train", "dev", "test")
        raw_dir (str): path to MELD.Raw directory
        frames_dir (str): path to MELD.Frames directory
    """
    # Create input and output directories where frames will be saved
    input_dir = os.path.join(raw_dir, split)
    output_dir = os.path.join(frames_dir, split)

    # Use os.listdir to list all files in the input directory
    # video_file will temporarily hold each file name (As a String)in the directory
    # Loop through each file in the input directory
    for video_file in os.listdir(input_dir):
        if video_file.endswith(".mp4"):
            # Create a string path to the video file
            video_path = os.path.join(input_dir, video_file)
            # Replace the .mp4 extension with .jpg for the output frame file name
            video_name = video_file.replace(".mp4", "")
            save_path = os.path.join(output_dir, video_name)
            
            # Only extract if the frame does not already exist
            if not os.path.exists(save_path):  # skip if already extracted
                extract_frames_sequence(video_path, save_path, sampling_mode=sampling_mode, target_tps=target_fps, num_frames=num_frames)

def process_iemocap_split(split_csv, frames_dir, sampling_mode, target_fps, num_frames):
    """
    Extract utterance-level frames for IEMOCAP.
    """

    df = pd.read_csv(split_csv)

    for _, row in df.iterrows():


        utterance_id = row["utterance_id"]

        save_path = os.path.join(
            frames_dir,
            utterance_id
        )

        if os.path.exists(save_path):
            continue

        extract_frames_sequence(
            video_path=row["video_path"],
            save_path=save_path,
            sampling_mode=sampling_mode,
            target_fps=target_fps,
            num_frames=num_frames,
            start_time=row["start_time"],
            end_time=row["end_time"]
        )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", choices=["meld", "iemocap"], required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sampling_mode", choices=["fps","uniform"], default="uniform")
    parser.add_argument("--target_fps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=8)

    args = parser.parse_args()

    if args.dataset == "meld":

        raw_dir = "data/MELD.Raw"
        if args.sampling_mode == "fps":
            root_dir = f"data/MELD.Frames_{args.target_fps}Hz"
        else:
            root_dir = f"data/MELD.Frames_uniform{args.num_frames}"
        frames_dir = os.path.join(root_dir, args.split)
        process_meld_split(args.split, raw_dir, frames_dir, sampling_mode=args.sampling_mode, target_fps=args.target_fps, num_frames=args.num_frames)
    elif args.dataset == "iemocap":

        split_csv = (f"data/IEMOCAP/splits/iemocap_{args.split}.csv")
        if args.sampling_mode == "fps":
            frames_dir = f"data/IEMOCAP.Frames_{args.target_fps}Hz"
        else:
            frames_dir = f"data/IEMOCAP.Frames_uniform{args.num_frames}"
        frames_dir = os.path.join(frames_dir, args.split)

        process_iemocap_split(split_csv, frames_dir, sampling_mode=args.sampling_mode, target_fps=args.target_fps, num_frames=args.num_frames)

# To download and extract the MELD dataset, you can use the following commands:
#curl.exe -C - -L "https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz" -o MELD.Raw.tar.gz
# tar -xvzf test.tar.gz -C ../data/MELD.Raw/
#Move-Item -Path "data/MELD.Raw/test/output_repeated_splits_test/*.mp4" -Destination "data/MELD.Raw/test/"
#Get-ChildItem -Path "data/MELD.Raw/test" -Filter "._*" | Remove-Item
# Running args: python src/extract_frames.py --dataset iemocap --split train --sampling_mode fps --target_fps 4 
# python src/extract_frames.py --dataset iemocap --split train --sampling_mode uniform --num_frames 8



