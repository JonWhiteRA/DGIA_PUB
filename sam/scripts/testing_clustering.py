import os
from io import BytesIO
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering, MeanShift
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, calinski_harabasz_score
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist
import json

true_labels = None

# Function to calculate Dunn Index
def dunn_index(X, labels):
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0
    inter_cluster_distances = []
    intra_cluster_distances = []
    
    for i in unique_labels:
        cluster_i = X[labels == i]
        intra_distance = np.mean(cdist(cluster_i, cluster_i, 'euclidean')) if len(cluster_i) > 1 else 0
        intra_cluster_distances.append(intra_distance)

        for j in unique_labels:
            if i != j:
                cluster_j = X[labels == j]
                inter_distance = np.min(cdist(cluster_i, cluster_j, 'euclidean'))
                inter_cluster_distances.append(inter_distance)

    return np.min(inter_cluster_distances) / np.max(intra_cluster_distances)

# Load JSON data
def load_data(filename):
    with open(filename) as file:
        data = json.load(file)
    # Extract frequencies into a DataFrame
    freq_data = []
    for doc, keywords in data.items():
        for keyword, frequency in keywords:
            freq_data.append([keyword, frequency])
    return pd.DataFrame(freq_data, columns=["Keyword", "Frequency"])

def run_clustering(X, min_clusters, max_clusters):
    results = []
    
    for n_clusters in range(min_clusters, max_clusters + 1):
        with st.spinner(f'Running clustering for {n_clusters} clusters...'):
            # Gaussian Mixture Model
            gmm = GaussianMixture(n_components=n_clusters)
            gmm_labels = gmm.fit_predict(X)
            gmm_results = {
                'Algorithm': 'GMM',
                'Clusters': n_clusters,
                'Silhouette Score': silhouette_score(X, gmm_labels),
                'Davies-Bouldin': davies_bouldin_score(X, gmm_labels),
                'Calinski-Harabasz': calinski_harabasz_score(X, gmm_labels)
            }
            results.append(gmm_results)

            # K-Means
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans_labels = kmeans.fit_predict(X)
            kmeans_results = {
                'Algorithm': 'K-Means',
                'Clusters': n_clusters,
                'Silhouette Score': silhouette_score(X, kmeans_labels),
                'Davies-Bouldin': davies_bouldin_score(X, kmeans_labels),
                'Calinski-Harabasz': calinski_harabasz_score(X, kmeans_labels)
            }
            results.append(kmeans_results)

            # Agglomerative Clustering
            agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
            agglomerative_labels = agglomerative.fit_predict(X)
            agglomerative_results = {
                'Algorithm': 'Agglomerative',
                'Clusters': n_clusters,
                'Silhouette Score': silhouette_score(X, agglomerative_labels),
                'Davies-Bouldin': davies_bouldin_score(X, agglomerative_labels),
                'Calinski-Harabasz': calinski_harabasz_score(X, agglomerative_labels)
            }
            results.append(agglomerative_results)

    return pd.DataFrame(results)

# Streamlit Interface
st.title("Clustering Algorithm Testing Suite")

# Hardcoded filename
filename = os.getenv('OUTPUT_PATH', '/app/output') + '/keywords.json'

# Selected filename
output_path = os.getenv('OUTPUT_PATH', '/app/output')
file_path = st.selectbox("Select Dataset", options=[output_path + '/general/keywords.json', output_path + '/govInfo/keywords.json', output_path + '/hcpss/keywords.json'])

if file_path:
    filename = file_path

# Load data
data = load_data(filename)

# Convert frequencies to a NumPy array for clustering
frequency_data = data['Frequency'].values.reshape(-1, 1)

# Parameters for clustering
min_clusters = st.number_input("Minimum Number of Clusters", min_value=1, max_value=10, value=2)
max_clusters = st.number_input("Maximum Number of Clusters", min_value=1, max_value=10, value=5)

# Perform clustering
if st.button("Run Clustering"):
    results_df = run_clustering(frequency_data, min_clusters, max_clusters)
    st.write("Clustering Results:")
    st.dataframe(results_df)

    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name='results.csv',
        mime='text/csv'
    )
