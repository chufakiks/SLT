import os
import cv2
import pandas as pd
import datetime

def timestamp_to_seconds(ts: str) -> float:
    """Convert a timestamp 'HH:MM:SS.mmm' to total seconds as float."""
    if not ts or ts.strip() == "":
        return 0.0
    dt = datetime.datetime.strptime(ts, "%H:%M:%S.%f")
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6


download_folder = "full_videos_openASL"
clip_folder = "openaslminisource"

src_tsv = pd.read_csv("openasl-v1.0.tsv", sep='\t')  # adjust separator if needed

# Group the TSV by video ID for efficiency
grouped = src_tsv.groupby('yid')

# Iterate over all video files in the folder
for filename in os.listdir(download_folder):
    if not filename.endswith(".mp4"):
        continue

    vid_id = os.path.splitext(filename)[0]  # remove extension to get yid
    input_path = os.path.join(download_folder, filename)

    if vid_id not in grouped.groups:
        continue  # skip if this video has no entries in TSV

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Failed to open {input_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Get all clips for this video
    clips = grouped.get_group(vid_id)
    for idx, row in clips.iterrows():
        start_sec = row['start']
        end_sec = row['end']

        output_path = os.path.join(clip_folder, f"{start_sec}_{end_sec}_{vid_id}.mp4")

        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        start_sec = timestamp_to_seconds(row['start'])
        end_sec = timestamp_to_seconds(row['end'])

        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()

    cap.release()
