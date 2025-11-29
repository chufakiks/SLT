from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd

embeddings = np.load('video_embeddings.npy')

#trying different distance thresholds out- we know there are at least 200 people.

""" for threshold in [0.24,0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        linkage='average',
        metric='cosine'
    )
    labels = clustering.fit_predict(embeddings)
    n_clusters = len(set(labels))
    
    print(f"Threshold {threshold}: {n_clusters} clusters") """
    
#We decide on 0.24 as it gives a little over 200 clusters. The paper said there were more than 200 people in the dataset.

clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.07,
        linkage='average',
        metric='cosine'
    )
labels = clustering.fit_predict(embeddings)

df = pd.read_csv('video_person_mapping.csv') 
video_ids = df['video_name'].tolist()

results_df = pd.DataFrame({
    'video_id': video_ids,
    'person_id': labels
})

# Save to CSV
output_filename = 'video_clusters.csv'
results_df.to_csv(output_filename, index=False)