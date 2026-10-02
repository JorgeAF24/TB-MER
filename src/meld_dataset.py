#Import os library to handle file paths
import os
#Import pandas to read, manipulate and write data in tabular form
import pandas as pd
import cv2

class MELDDataset:
    def __init__(self, root_dir: str, frames_dir: str, split: str = "train"):
        """
        MELD Dataset Loader (text + labels)
        Args:
            root_dir (str): Path to MELD.Raw folder
            split (str): "train", "dev", or "test"
        """
        self.root_dir = root_dir
        self.frames_dir = frames_dir
        self.split = split
        #Use a f-string to pass the split variable into the file name
        # self.csv_path builds and stores the path as a string to the CSV file 
        self.csv_path = os.path.join(root_dir, f"{split}_sent_emo.csv")
        
        # Load CSV file located at self.csv_path into a pandas DataFrame
        # self.data stores the DataFrame
        self.data = pd.read_csv(self.csv_path)

        # self.emotions stores the unique emotions labels from column "Emotion" 
        # And sorted in alphabetical order
        self.emotions = sorted(self.data["Emotion"].unique())

    def __len__(self):
        # Return the number of samples or utterances (rows) in the dataset
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Return utterance text + emotion label
        """
        # Use iloc to access the row at index idx
        # Example: if idx=0, it returns the first row of the DataFrame
        row = self.data.iloc[idx]
        # Extract the text from the "Utterance" column and the label from the "Emotion" column
        text = row["Utterance"]
        label = row["Emotion"]

        #Match video file to extracted frame image
        video_id = row["Dialogue_ID"]
        utterance_id = row["Utterance_ID"]
        video_name = f"dia{video_id}_utt{utterance_id}"
        #Construct the path to the directory containing the frames for this video
        frame_dir = os.path.join(self.frames_dir, self.split, video_name)

        #Define a buffer to store the frames as they are loaded. 
        # Buffer: temporary storage area
        frames = []

        # Load the Image (as numpy array)
        # Check if the frame directory exists before trying to load frames
        if os.path.exists(frame_dir):
            #Load all frames in order
            # os.listdir(frame_dir) returns a list of all file names in the directory
            # sorted() sorts the list of file names in alphabetical order, which is important to maintain the correct temporal order of frames
            frame_files = sorted(os.listdir(frame_dir))

            for f in frame_files:
                # Construct the full path to the frame image by joining the frame directory and the file name
                frame_path = os.path.join(frame_dir, f)
                # Read the image using OpenCV
                # OpenCV decodes JPEG into a NumPy array in BGR format
                image = cv2.imread(frame_path)
                if image is None:
                    print(f"[Warning] Failed to load image: {frame_path}")
                    frames.append(None)
                    continue

                # Convert BGR to RGB compatible with CLIP
                # Copy the image to ensure it is not modified by OpenCV
                image = image[:,:, ::-1].copy()
                # Resize images to 224x224 for CLIP
                # So the Dataloader can apply batches of images with same size
                image = cv2.resize(image, (224, 224))
                # Append the loaded and processed image to the frames buffer
                frames.append(image)
        else:
            print(f"[Warning] Missing file: {frame_dir}")
            frames = []  # No image available
        # Return Python objects: text (str), label (str), frames (list of numpy arrays or None)
        return text, label, frames
        

if __name__ == "__main__":
    # Example usage
    dataset = MELDDataset(
        root_dir="data/MELD.Raw",
        frames_dir="data/MELD.Frames",
        split="train"
    )
    print("Number of samples:", len(dataset))
    print("Emotions:", dataset.emotions)
    sample = dataset[0]
    print("Sample text:", sample[0])
    print("Sample label:", sample[1])
    print(f"[DEBUG] Frames container type: {type(sample[2])}")
    print(f"[DEBUG] Number of frames: {len(sample[2])}")
    if sample[2][0] is not None:
        print(f"[DEBUG] sample[2][0].shape = {sample[2][0].shape}")
    else:
        print("[DEBUG] sample[2][0] is None")
    for i, frame in enumerate(sample[2]):
        if frame is None:
            print(f"[DEBUG] Frame {i}: None")
        else:
            print(f"[DEBUG] Frame {i}: shape={frame.shape}")
