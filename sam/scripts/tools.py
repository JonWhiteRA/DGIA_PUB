import os
import json
import subprocess
import bisect
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, MeanShift, DBSCAN, OPTICS, Birch, HDBSCAN, estimate_bandwidth

from metrics import *

CLUSTER_NUM_ALGORITHMS = ['K-Means', 'Agglomerative']
OUTPUT_PATH = os.getenv('OUTPUT_PATH', '/app/output')
AVAILABLE_FILES = ['keywords.json', 'output_embeddings.json', 'entities.json']
DATASETS_OUTPUT = {
            'General': os.path.join(OUTPUT_PATH, 'corpus'),
            'GovInfo': os.path.join(OUTPUT_PATH, 'govInfo'),
            'Howard County Public School System': os.path.join(OUTPUT_PATH, 'hcpss'),
            'GDPR By Paragraph' : os.path.join(OUTPUT_PATH, 'gdpr_by_paragraph'),
            'GDPR By Article' : os.path.join(OUTPUT_PATH, 'gdpr_by_article')
        }
DATASETS_DATA = {
        "Howard County Public School System": "/data/hcpss/",
        "GovInfo"                           : "/data/gov/",
        "General"                           : "/data/corpus/",
        "GDPR By Paragraph"                 : "/data/gdpr_by_paragraph",
        "GDPR By Article"                   : "/data/gdpr_by_article"
    }
ALGORITHM_LOOKUP = {
    'K-Means'       :   KMeans,
    'Agglomerative' :   AgglomerativeClustering,
    'Spectral'      :   SpectralClustering,
    'Mean Shift'    :   MeanShift,
    'DBSCAN'        :   DBSCAN,
    'HDBSCAN'       :   HDBSCAN,
    'OPTICS'        :   OPTICS,
    'Birch'         :   Birch
}

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

# Prepare feature matrix for entities
def prepare_feature_matrix_entities(data):
    titles = list(data.keys())
    entity_set = set()
    for entities in data.values():
        for entity in entities:
            entity_set.add(entity[0])
    entity_list = sorted(entity_set)
    feature_matrix = np.zeros((len(data), len(entity_list)))
    for i, (title, entities) in enumerate(data.items()):
        for entity in entities:
            index = entity_list.index(entity[0])
            feature_matrix[i, index] = entity[2]
    return titles, feature_matrix

def run_alg(a, features, n):
    model_class = ALGORITHM_LOOKUP[a]
    if a in CLUSTER_NUM_ALGORITHMS:
        if a == "Spectral":
            model = model_class(n_clusters=n, assign_labels="discretize")
        else:
            model = model_class(n_clusters=n)
    else:
        if a == "Mean Shift":
            bandwidth = estimate_bandwidth(features, quantile=0.2)
            model = model_class(bandwidth=bandwidth)
        else:
            model = model_class()
    clusters = model.fit_predict(features)
    return clusters

# Function to run clustering algorithms
def generate_features(selected_file, file_path):
    data = load_data(file_path)

    if isinstance(data, dict):
        if selected_file == 'output_embeddings.json':
            titles, feature_matrix = prepare_feature_matrix_embeddings(data)
        elif selected_file == 'entities.json':
            titles, feature_matrix = prepare_feature_matrix_entities(data)
        else:
            titles, feature_matrix = prepare_feature_matrix_keywords(data)

        feature_matrix = StandardScaler().fit_transform(feature_matrix)

        return (titles, feature_matrix)

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

def spreadsheet_metrics(dataset_name, file_name, algorithm_name, data, labels):
    (s, db, ch, d, coh, sep, x) = calculate_metrics(data, labels)
    (score, grades) = grade(s, db, ch, d, coh, sep, x)
    results = {
        'Dataset'          : dataset_name,
        'File'             : file_name,
        'Algorithm'        : algorithm_name,
        'Clusters'         : len(set(labels)) - (1 if -1 in labels else 0),
        'Silhouette Score' : s,
        'Davies-Bouldin'   : db,
        'Calinski-Harabasz': ch,
        'Dunn Index'       : d,
        'Cohesion'         : cohesion(data, labels),
        'Separation'       : separation(data, labels),
        'Xie-Beni Index'   : xie_beni_index(data, labels),
        'Score'            : score,
        'Grade'            : grades
    }
    return results

def display_metrics(feature_matrix, clusters):
        # Calculate clustering metrics
        (s, db, ch, d, coh, sep, x) = calculate_metrics(feature_matrix, clusters)
        (score, letter) = grade(s, db, ch, d, coh, sep, x)

        # Display metrics
        st.subheader("Clustering Metrics")
        st.write(f"Silhouette Score: {str(s)}")
        st.write(f"Davies-Bouldin Index: {str(db)}")
        st.write(f"Calinski-Harabasz: {str(ch)}")
        st.write(f"Dunn Index: {str(d)}")
        st.write(f"Overall Score: {str(score)}")
        st.write(f"Overall Grade: {str(letter)}")

def show_clustered_titles(clusters, titles):
        # Display results
        clustered_titles = {i: [] for i in set(clusters)}
        for title, cluster in zip(titles, clusters):
            clustered_titles[cluster].append(title)

        st.subheader("Clustered Titles")
        for cluster_id, titles in clustered_titles.items():
            st.write(f"**Cluster {cluster_id}:**")
            st.write(", ".join(titles))

def single_selection(prompt, options, key):
    # Initialize the session state for the specific key if it doesn't exist
    if key not in st.session_state:
        st.session_state[key] = options[0]  # Set default to the first option

    # Create the selectbox and update the session state
    selected_option = st.selectbox(
        prompt,
        options=options,
        index=options.index(st.session_state[key]),
        key=key
    )

    # Update session state only if the selected option changes
    if selected_option != st.session_state[key]:
        st.session_state[key] = selected_option

    return st.session_state[key]

def multi_selection(prompt, options, key):
    # Initialize the session state for the specific key if it doesn't exist
    if key not in st.session_state:
        st.session_state[key] = []  # Set default to an empty list

    # Create the multiselect without a default parameter
    selected_options = st.multiselect(
        prompt,
        options=options,
        key=key
    )

    # Update session state only if the selected options change
    if selected_options != st.session_state[key]:
        st.session_state[key] = selected_options

    return st.session_state[key]

def preprocess_data(output_dir, folder_path):
    if st.button("Load Data"):
        st.session_state.is_running = True
        st.spinner("Running script...")
        command1 = f"python corpus_processor_1.py --output_dir {output_dir} --input_dir {folder_path}"
        command2 = f"python corpus_processor_2.py --output_dir {output_dir} --input_dir {folder_path}"

        try:
            output_dir_contents = os.listdir(output_dir)
        except FileNotFoundError:
            output_dir_contents = []

        if "entities.json" in output_dir_contents:
            if "top_related_files_entities.json" in output_dir_contents:
                st.session_state.output = "Data already processed."
            else:
                result = subprocess.run(command2, shell=True, capture_output=True, text=True)
                st.session_state.output = result.stdout if result.returncode == 0 else result.stderr
        else:
            result = subprocess.run(command1, shell=True, capture_output=True, text=True)
            st.session_state.output = result.stdout if result.returncode == 0 else result.stderr
            if result.returncode == 0:
                result2 = subprocess.run(command2, shell=True, capture_output=True, text=True)
                st.session_state.output += "\n" + (result2.stdout if result2.returncode == 0 else result2.stderr)

        st.session_state.is_running = False

    # Conditional display
    if st.session_state.is_running:
        st.spinner("Running script...")
    else:
        if st.session_state.output:
            st.write("Results:")
            st.code(st.session_state.output)
