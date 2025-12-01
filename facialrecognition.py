from pathlib import Path
import cv2
from inference_onnx import LVFaceONNXInferencer
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
    model_path="LVFace-L_Glint360K.onnx",
    use_gpu=True
)

embs = []

# Single temp file to reuse
temp_frame_path = "temp_frame.jpg"

try:
    for video in Path("/work3/s235253/raw_vid").iterdir():      
        emb = []
        frames = sample_and_save_frames(video, num_frames=10)

        for frame in frames:
            cv2.imwrite(temp_frame_path, frame)
            framefeat = inferencer.infer_from_image(temp_frame_path)
            # Skip if inference failed or returned None/empty
            if framefeat is None:
                continue
            if isinstance(framefeat, np.ndarray) and framefeat.size == 0:
                continue
            if isinstance(framefeat, np.ndarray) and not np.isfinite(framefeat).all():
                continue
            print("emb extracted")
            emb.append(framefeat)
        
        #skip empty embeddings
        if len(emb) == 0:
            continue
        emb = np.mean(emb, axis=0)
        emb = normalize(emb.reshape(1, -1), norm='l2')
        embs.append(emb.flatten())

finally:
    # Clean up temp file
    if Path(temp_frame_path).exists():
        Path(temp_frame_path).unlink()

np.save("video_embeddings.npy", np.array(embs))



