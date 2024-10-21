import os
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, MeanShift, DBSCAN, OPTICS, Birch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Load the JSON data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Prepare feature matrix for keywords
def prepare_feature_matrix_keywords(data):
    titles = list(data.keys())
    keyword_set = set()
    
    for keywords in data.values():
        for keyword, frequency in keywords:
            keyword_set.add(keyword)
    
    keyword_list = sorted(keyword_set)
    feature_matrix = np.zeros((len(data), len(keyword_list)))
    
    for i, (title, keywords) in enumerate(data.items()):
        for keyword, frequency in keywords:
            index = keyword_list.index(keyword)
            feature_matrix[i, index] = frequency
    
    return titles, feature_matrix

# Prepare feature matrix for embeddings
def prepare_feature_matrix_embeddings(data):
    titles = list(data.keys())
    feature_matrix = np.array(list(data.values()))
    return titles, feature_matrix

# Function to perform clustering
def perform_clustering(algorithm, feature_matrix, n_clusters):
    if algorithm == 'K-Means':
        model = KMeans(n_clusters=n_clusters)
    elif algorithm == 'Agglomerative':
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif algorithm == 'Spectral':
        model = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize")
    elif algorithm == 'Mean Shift':
        model = MeanShift()
    elif algorithm == 'DBSCAN':
        model = DBSCAN()
    elif algorithm == 'OPTICS':
        model = OPTICS()
    elif algorithm == 'Birch':
        model = Birch()
    
    return model.fit_predict(feature_matrix)

# Main function for clustering
def run_clustering(selected_file, file_path, pca_dims, algorithm, n_clusters):
    data = load_data(file_path)

    if isinstance(data, dict):
        # Determine if the data is embeddings or keywords
        if selected_file == 'output_embeddings.json':
            titles, feature_matrix = prepare_feature_matrix_embeddings(data)
        else:
            titles, feature_matrix = prepare_feature_matrix_keywords(data)

        # Standardize features
        feature_matrix = StandardScaler().fit_transform(feature_matrix)

        # Run clustering immediately after selection
        clusters = perform_clustering(algorithm, feature_matrix, n_clusters)

        # Dimensionality reduction for visualization
        pca = PCA(n_components=pca_dims)
        reduced_features = pca.fit_transform(feature_matrix)

        # Create a scatter plot using Plotly
        fig = go.Figure()
        distinct_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        if pca_dims == 3:
            # 3D scatter plot
            fig.add_trace(go.Scatter3d(
                x=reduced_features[:, 0],
                y=reduced_features[:, 1],
                z=reduced_features[:, 2],
                mode='markers',
                marker=dict(color=[distinct_colors[cluster_id % len(distinct_colors)] for cluster_id in clusters],
                            size=5),
                text=titles,
                hoverinfo='text'
            ))
            fig.update_layout(title="3D Clustering Visualization", scene=dict(
                xaxis_title='Principal Component 1',
                yaxis_title='Principal Component 2',
                zaxis_title='Principal Component 3'))
        else:
            # 2D scatter plot
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
            fig.update_layout(title="2D Clustering Visualization",
                              xaxis_title='Principal Component 1',
                              yaxis_title='Principal Component 2')

        # Show the plot in Streamlit
        st.plotly_chart(fig)

        # Calculate clustering metrics
        silhouette = silhouette_score(feature_matrix, clusters)
        davies_bouldin = davies_bouldin_score(feature_matrix, clusters)

        # Display metrics
        st.subheader("Clustering Metrics")
        st.write(f"Silhouette Score: {silhouette:.2f}")
        st.write(f"Davies-Bouldin Index: {davies_bouldin:.2f}")

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

# Main entry point
def main():
    st.title("Graph Analysis")

    # Dropdown for selecting the dataset
    output_path = os.getenv('OUTPUT_PATH', '/app/output')
    datasets = {
        'General': os.path.join(output_path, 'general'),
        'GovInfo': os.path.join(output_path, 'govInfo'),
        'Howard County Public School System': os.path.join(output_path, 'hcpss')
    }

    selected_dataset = st.selectbox("Select Dataset", options=list(datasets.keys()))
    selected_file = st.selectbox("Select File", options=['keywords.json', 'output_embeddings.json'])
    file_path = os.path.join(datasets[selected_dataset], selected_file)

    # Dropdown for selecting PCA dimensions
    pca_dims = st.selectbox("Select PCA Dimensions", options=[2, 3])

    # Dropdown for selecting clustering algorithm
    algorithm = st.selectbox("Select Clustering Algorithm", 
                              options=['K-Means', 'Agglomerative', 'Spectral', 
                                       'Mean Shift', 'DBSCAN', 'OPTICS', 'Birch'])

    # Slider for number of clusters (only for algorithms that require it)
    n_clusters = None
    if algorithm in ['K-Means', 'Agglomerative', 'Spectral']:
        n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)

    # Button to run clustering
    if st.button("Run Clustering"):
        run_clustering(selected_file, file_path, pca_dims, algorithm, n_clusters)

if __name__ == '__main__':
    main()
