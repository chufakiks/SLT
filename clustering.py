from sklearn.cluster import AgglomerativeClustering
import numpy as np
import csv
import pandas as pd

embeddings = np.load('video_embeddings.npy')

#trying different distance thresholds out- we know there are at least 200 people.

for threshold in [0.24,0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        linkage='average',
        metric='cosine'
    )
    labels = clustering.fit_predict(embeddings)
    n_clusters = len(set(labels))
    
    print(f"Threshold {threshold}: {n_clusters} clusters")
    
#We decide on 0.24 as it gives a little over 200 clusters. The paper said there were more than 200 people in the dataset.