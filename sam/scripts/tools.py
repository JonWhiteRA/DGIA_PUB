import os
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, MeanShift, DBSCAN, OPTICS, Birch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

ALGORITHMS = ['K-Means', 'Agglomerative', 'Spectral', 
                'Mean Shift', 'DBSCAN', 'OPTICS', 'Birch']
OUTPUT_PATH = os.getenv('OUTPUT_PATH', '/app/output')
DATASETS_OUTPUT = {
            'General': os.path.join(OUTPUT_PATH, 'general'),
            'GovInfo': os.path.join(OUTPUT_PATH, 'govInfo'),
            'Howard County Public School System': os.path.join(OUTPUT_PATH, 'hcpss')
        }
DATASETS_DATA = {
        "Howard County Public School System": "/data/hcpss/",
        "GovInfo": "/data/gov/",
        "General": "/data/corpus/"
    }

element_count = 0

# Load the JSON data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def get_element_count():
    global element_count
    element_count += 1
    return element_count

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

# Prepare feature matrix for entities
def prepare_feature_matrix_entities(data):
    print("in tools.py")
    titles = list(data.keys())
    entity_set = set()
    
    print("entering for loop")
    for entities in data.values():
        for entity in entities:
            entity_set.add(entity[0])
    print("done for loop")
    entity_list = sorted(entity_set)
    feature_matrix = np.zeros((len(data), len(entity_list)))
    print("entering another for loop")
    for i, (title, entities) in enumerate(data.items()):
        for entity in entities:
            index = entity_list.index(entity[0])  # Get the entity name
            feature_matrix[i, index] = entity[2]  # Use the frequency
    print("returning")
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
    
    labels = model.fit_predict(feature_matrix)
    return labels

# Main function for clustering
def run_clustering(selected_file, file_path, algorithm, n_clusters):
    print("running algorithm: " + str(algorithm) + " on " + file_path)
    data = load_data(file_path)

    if isinstance(data, dict):
        if selected_file == 'output_embeddings.json':
            titles, feature_matrix = prepare_feature_matrix_embeddings(data)
        elif selected_file == 'entities.json':
            titles, feature_matrix = prepare_feature_matrix_entities(data)
        else:
            titles, feature_matrix = prepare_feature_matrix_keywords(data)

        feature_matrix = standardize_feature_matrix(feature_matrix)
        clusters = perform_clustering(algorithm, feature_matrix, n_clusters)

        return (feature_matrix, clusters, titles)

def standardize_feature_matrix(feature_matrix):
        return StandardScaler().fit_transform(feature_matrix)

def plot_clusters(feature_matrix, clusters, pca_dims, titles):
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

def calculate_metrics(data, labels):
    silhouette = silhouette_score(data, labels) if len(set(labels)) > 1 else None,
    davies_bouldin = davies_bouldin_score(data, labels) if len(set(labels)) > 1 else None,
    calinski =  calinski_harabasz_score(data, labels) if len(set(labels)) > 1 else None
    return (silhouette, davies_bouldin, calinski)

def display_metrics(feature_matrix, clusters):
        # Calculate clustering metrics
        silhouette = silhouette_score(feature_matrix, clusters)
        davies_bouldin = davies_bouldin_score(feature_matrix, clusters)

        # Display metrics
        st.subheader("Clustering Metrics")
        st.write(f"Silhouette Score: {silhouette:.2f}")
        st.write(f"Davies-Bouldin Index: {davies_bouldin:.2f}")

def show_clustered_titles(clusters, titles):
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

def dataset_selection():
    selected_dataset = st.selectbox("Select Dataset", options=list(DATASETS_OUTPUT.keys()), key=get_element_count())
    return selected_dataset

def file_selection(selected_dataset):
    selected_file = st.selectbox("Select File", options=['keywords.json', 'output_embeddings.json', 'entities.json'], key=get_element_count())
    file_path = os.path.join(DATASETS_OUTPUT[selected_dataset], selected_file)
    return (selected_file, file_path)

def multifile_selection(selected_dataset):
    selected_files = st.multiselect("Select JSON files for analysis", ['keywords.json', 'output_embeddings.json', 'entities.json'])
    return selected_files

def pca_dims_selection():
    pca_dims = st.selectbox("Select PCA Dimensions", options=[2, 3], key=get_element_count())
    return pca_dims

def algorithm_selection():
    return st.selectbox("Select Clustering Algorithm", options=ALGORITHMS, key=get_element_count())

def multialgorithm_selection():
    return st.multiselect("Select Clustering Algorithm", options=ALGORITHMS, default=ALGORITHMS)

def dataset_data(dataset):
    return DATASETS_DATA[dataset]

def dataset_output(dataset):
    return DATASETS_OUTPUT[dataset]