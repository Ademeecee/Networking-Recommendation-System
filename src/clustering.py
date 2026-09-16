import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load engineered feature matrix
df = pd.read_csv('data/artifacts/engineered_member_profiles.csv')
metadata_cols = ['email', 'full_name', 'job_title', 'organization',
                 'clean_sector', 'seniority_level_name', 'event_attendance_count']
X = df.drop(columns=metadata_cols)

# Optimal K-Means Model
optimal_k = 5
km = KMeans(n_clusters=optimal_k, random_state=24, n_init=10)
df['cluster'] = km.fit_predict(X)

# 2D PCA Dimensionality Reduction
pca = PCA(n_components=2, random_state=24)
pca_coords = pca.fit_transform(X)
df['pca_x'], df['pca_y'] = pca_coords[:, 0], pca_coords[:, 1]

# Save Clustered Data for Recommendation Engine
df.to_csv('data/artifacts/clustered_member_profiles.csv', index=False)

# Save the trained clustering model
joblib.dump(km, "data/artifacts/kmeans_cluster_model.joblib")

# Save Clustered Data as compressed joblib file in the current folder
joblib.dump(df, "data/artifacts/cynam_members_prod.joblib", compress=3)
print(
    f"Saved model and profiles to 'data/artifacts/' directory with {len(df)} records and {optimal_k} clusters.")
