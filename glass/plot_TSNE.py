# import cool modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# can avoid importing these two to reduce dependencies
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import jaccard_score

path = "../data/glass.csv"
glass = pd.read_csv(path)

# define features and target columns
X = glass.iloc[:, :-1]  # All rows, all columns except the last one (features)
y = glass.iloc[:, -1] 


# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Define the range of k values
k_values = [2, 3, 4, 5, 6]  # Change these values as per your requirement

# Run KMeans for each k and plot the t-SNE
best_Jaccard, best_k = -1, None
for k in k_values:
    # Run KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    try:
        jaccard = jaccard_score(y_encoded, labels, average='macro')
        if jaccard > best_Jaccard:
            best_Jaccard = jaccard
            best_k = k
        print(f'Jaccard Score for k={k}: {jaccard:.4f}')
    except ValueError as e:
        print(f"Jaccard score calculation failed for k={k} due to: {e}")
    
    # Reduce dimensions using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    
    # # Plot t-SNE with cluster labels
    # plt.figure(figsize=(8, 6))
    # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
    # plt.title(f't-SNE plot for k={k}')
    # plt.xlabel('t-SNE 1')
    # plt.ylabel('t-SNE 2')
    # plt.colorbar(label='Cluster Label')
    # plt.show()