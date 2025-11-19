import cv2
import numpy as np
from pathlib import Path

def sample_and_save_frames(video_path, output_dir, num_frames=20):
    """
    Saves 20 evenly spaced frames from the video.
    Filenames: video_id_frameIndex.jpg
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 20 evenly spaced indices
    indices = np.linspace(0, total - 1, num_frames).astype(int)

    video_id = video_path.stem  # filename without extension

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue

        frame_name = f"{video_id}_{idx}.jpg"
        cv2.imwrite(str(output_dir / frame_name), frame)

    cap.release()
    print("Saved", len(indices), "frames for", video_id)


# Example usage
# from glob import glob
# sample_and_save_frames(glob(f"{"path_to_video_folder"}/*.*") path_to_output_directory)
