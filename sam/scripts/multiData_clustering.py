# To run test environment from browser

import streamlit as st
import subprocess
import os
from io import BytesIO
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

def test_clusters(out):
    filename = out + '/keywords.json'

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

# Define the available data folders
data_folders = {
    "Howard County Public School System": "/data/hcpss/",
    "GovInfo": "/data/gov/",
    "General": "/data/corpus/"
}

output_folders = {
    "Howard County Public School System": "output/hcpss/",
    "GovInfo": "output/govInfo/",
    "General": "output/general/"
}

# pre-processing
script_path = "corpus_processor_1.py"

# Streamlit app interface
st.title("Cluster and Dataset Testing")

selected_dataset = st.selectbox("Choose a dataset", list(data_folders.keys()))
folder_path = data_folders[selected_dataset]
output_dir = output_folders[selected_dataset]

if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'output' not in st.session_state:
    st.session_state.output = None
if 'error' not in st.session_state:
    st.session_state.error = None

# Button to run the script
if st.button("Run"):
    st.session_state.is_running = True  # Set running state to True
    command = f"python {script_path} --output_dir {output_dir} {folder_path}"
    command2 = f"python corpus_processor_2.py --output_dir {output_dir} {folder_path}"
    
    # Execute the command
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            st.session_state.output = result.stdout  # Store output
            st.session_state.error = None
        else:
            st.session_state.error = result.stderr  # Store error
            st.session_state.output = None
    except Exception as e:
        st.session_state.error = str(e)  # Store exception message
        st.session_state.output = None
    finally:
        try:
            result = subprocess.run(command2, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                st.session_state.output = result.stdout  # Store output
                st.session_state.error = None
            else:
                st.session_state.error = result.stderr  # Store error
                st.session_state.output = None
        except Exception as e:
            st.session_state.error = str(e)  # Store exception message
            st.session_state.output = None
        st.session_state.is_running = False  # Reset running state

# Conditional display based on running state
if st.session_state.is_running:
    st.spinner("Running script...")  # Show spinner while running
else:
    # Display output or error if available
    if st.session_state.output is not None:
        st.write("Results:")
        st.code(st.session_state.output)
        test_clusters(output_dir)

    if st.session_state.error is not None:
        st.error("Error occurred:\n" + st.session_state.error)
