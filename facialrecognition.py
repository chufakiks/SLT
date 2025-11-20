from pathlib import Path
import cv2
from inference_onnx import LVFaceONNXInferencer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
import numpy as np

def sample_and_save_frames(video, num_frames=10):
    frames = []
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Evenly spaced indices
    indices = np.linspace(0, total - 1, num_frames).astype(int)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(frame)
    
    cap.release()
    return frames

inferencer = LVFaceONNXInferencer(
    model_path="LVFace-B_Glint360K.onnx",
    use_gpu=False
)

embs = []

# Single temp file to reuse
temp_frame_path = "temp_frame.jpg"

try:
    for video in Path("videos").iterdir():      
        emb = []
        frames = sample_and_save_frames(video, num_frames=10)

        for frame in frames:
            cv2.imwrite(temp_frame_path, frame)
            framefeat = inferencer.infer_from_image(temp_frame_path)
            emb.append(framefeat)
        
        emb = np.mean(emb, axis=0)
        emb = normalize(emb.reshape(1, -1), norm='l2')
        embs.append(emb.flatten())

finally:
    # Clean up temp file
    if Path(temp_frame_path).exists():
        Path(temp_frame_path).unlink()

print(f"Extracted {len(embs)} video embeddings")

clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5, metric='cosine', linkage='average')
labels = clustering.fit_predict(embs)
print(labels)




