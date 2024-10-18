import os
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Function to compute Dunn Index
def dunn_index(clusters, feature_matrix):
    unique_clusters = np.unique(clusters)
    
    # Return NaN if there is only one cluster
    if len(unique_clusters) < 2:
        return float('nan')
    
    inter_cluster_distances = []
    intra_cluster_distances = []

    for i in unique_clusters:
        cluster_points = feature_matrix[clusters == i]
        if len(cluster_points) < 2:
            continue
        
        # Calculate intra-cluster distance
        intra_distance = np.mean([np.linalg.norm(p1 - p2) for p1 in cluster_points for p2 in cluster_points if not np.array_equal(p1, p2)])
        intra_cluster_distances.append(intra_distance)

        for j in unique_clusters:
            if i >= j:
                continue
            
            # Calculate inter-cluster distance
            other_cluster_points = feature_matrix[clusters == j]
            inter_distance = np.min([np.linalg.norm(p1 - p2) for p1 in cluster_points for p2 in other_cluster_points])
            inter_cluster_distances.append(inter_distance)

    if not inter_cluster_distances or not intra_cluster_distances:
        return float('nan')
    
    return np.min(inter_cluster_distances) / np.max(intra_cluster_distances)

# Load the JSON data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Prepare the feature matrix
def prepare_feature_matrix(data):
    titles = []
    keyword_set = set()
    
    for title, keywords in data.items():
        titles.append(title)
        for keyword, frequency in keywords:
            keyword_set.add(keyword)
    
    keyword_list = sorted(keyword_set)
    feature_matrix = np.zeros((len(data), len(keyword_list)))
    
    for i, (title, keywords) in enumerate(data.items()):
        for keyword, frequency in keywords:
            index = keyword_list.index(keyword)
            feature_matrix[i, index] = frequency
    
    return titles, feature_matrix, keyword_list

# Perform Mean Shift Clustering
def mean_shift_clustering(feature_matrix, bandwidth):
    ms = MeanShift(bandwidth=bandwidth)
    clusters = ms.fit_predict(feature_matrix)
    return clusters

# Main function for clustering
def run_clustering(file_path):
    st.title("Mean Shift Clustering of Titles")
    
    data = load_data(file_path)
    
    if isinstance(data, dict):
        titles, feature_matrix, keyword_list = prepare_feature_matrix(data)
        
        # Standardize features
        feature_matrix = StandardScaler().fit_transform(feature_matrix)

        # Slider for bandwidth
        bandwidth = st.slider("Select Bandwidth", min_value=0.01, max_value=5.0, value=1.0, step=0.01)

        # Perform clustering
        clusters = mean_shift_clustering(feature_matrix, bandwidth)

        # Dimensionality reduction for visualization
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(feature_matrix)

        # Create a scatter plot using plotly.graph_objects
        fig = go.Figure()

        distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        for cluster_id in set(clusters):
            cluster_points = reduced_features[clusters == cluster_id]
            fig.add_trace(go.Scatter(
                x=cluster_points[:, 0],
                y=cluster_points[:, 1],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(color=distinct_colors[cluster_id % len(distinct_colors)], size=10),
                text=[titles[i] for i in range(len(titles)) if clusters[i] == cluster_id],
                hoverinfo='text'
            ))

        fig.update_layout(
            title="Mean Shift Clustering of Titles",
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2'
        )

        # Show the plot in Streamlit
        st.plotly_chart(fig)

        # Calculate clustering metrics
        silhouette = silhouette_score(feature_matrix, clusters)
        davies_bouldin = davies_bouldin_score(feature_matrix, clusters)
        dunn = dunn_index(clusters, feature_matrix)
        
        # Inertia is not applicable to Mean Shift Clustering
        inertia = None  
        
        # Display metrics
        st.subheader("Clustering Metrics")
        st.write(f"Silhouette Score: {silhouette:.2f}")
        st.write(f"Davies-Bouldin Index: {davies_bouldin:.2f}")
        st.write(f"Dunn Index: {dunn:.2f}")
        st.write(f"Inertia: Not applicable for Mean Shift Clustering")

        # Checkbox to display clustered titles
        show_clusters = st.checkbox("Show Clustered Titles")
        
        if show_clusters:
            # Display results
            clustered_titles = {i: [] for i in set(clusters)}
            for title, cluster in zip(titles, clusters):
                clustered_titles[cluster].append(title)

            st.subheader("Clustered Titles")
            for cluster_id, titles in clustered_titles.items():
                st.write(f"**Cluster {cluster_id}:**")
                st.write(", ".join(titles))
    else:
        st.error("Loaded data is not in the expected format. Please check the JSON file.")

# Streamlit app entry point
def main():
    st.title("Mean Shift Clustering App")

    # Dropdown for selecting the dataset
    file_path = st.selectbox("Select Dataset", options=[os.getenv('OUTPUT_PATH', '/app/output') + '/keywords.json'])

    if file_path:
        run_clustering(file_path)

if __name__ == '__main__':
    main()
