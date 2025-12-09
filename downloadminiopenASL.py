import yt_dlp
import pandas as pd
import os
import cv2

openasl = pd.read_csv('openasl-v1.0.tsv', sep='\t')

download_folder = "full_videos_openASL"
clip_folder = "openaslminisource"
os.makedirs(download_folder, exist_ok=True)
os.makedirs(clip_folder, exist_ok=True)


videoidset = set()

for yid in openasl['yid']:
    if len(videoidset) >= 10:
        break
    else:
        if yid not in videoidset and not(os.path.exists(os.path.join(download_folder, yid + ".mp4" ))):
            videoidset.add(yid)
        else:
            continue

url = []

for vid in videoidset:
    url.append("https://www.youtube.com/watch?v=" + vid)

ydl_opts = {
    'format': 'best', 
    'outtmpl': os.path.join(download_folder, '%(id)s.%(ext)s'),
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(url)