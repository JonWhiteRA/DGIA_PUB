import streamlit as st
import subprocess
import os
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, MeanShift, DBSCAN, Birch, OPTICS, AffinityPropagation
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import json

from tools import *

total_steps = 0
step = 0
progress_bar = 0

def update_progress():
    global step
    global total_steps
    step += 1
    progress_bar.progress(step / total_steps)
    return step


def process_files_and_run_clustering(output_dir, min_clusters, max_clusters, selected_files, algs):
    all_results = []
    for file_name in selected_files:
        filename = os.path.join(output_dir, file_name)
        data = load_data(filename)

        if file_name == 'output_embeddings.json':
            titles, feature_matrix = prepare_feature_matrix_embeddings(data)
        elif file_name == 'entities.json':
            titles, feature_matrix = prepare_feature_matrix_entities(data)
        else:
            titles, feature_matrix = prepare_feature_matrix_keywords(data)

        for a in algs:
            for i in range(min_clusters, max_clusters +1):
                (feature_matrix, labels, titles) = run_clustering(file_name, filename, a, i)
                all_results.append(generate_metrics(a, feature_matrix, labels))
                update_progress()
    
    return pd.DataFrame(all_results)

def generate_metrics(algorithm_name, data, labels):
    results = {
        'Algorithm': algorithm_name,
        'Clusters': len(set(labels)) - (1 if -1 in labels else 0),
        'Silhouette Score': silhouette_score(data, labels) if len(set(labels)) > 1 else None,
        'Davies-Bouldin': davies_bouldin_score(data, labels) if len(set(labels)) > 1 else None,
        'Calinski-Harabasz': calinski_harabasz_score(data, labels) if len(set(labels)) > 1 else None
    }
    return results

def preprocess_data(output_dir, folder_path):
    if st.button("Load Data"):
        st.session_state.is_running = True
        command1 = f"python corpus_processor_1.py --output_dir {output_dir} {folder_path}"
        command2 = f"python corpus_processor_2.py --output_dir {output_dir} {folder_path}"

        output_dir_contents = os.listdir(output_dir)

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

def show_page():
    st.title("Cluster Testing")

    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'output' not in st.session_state:
        st.session_state.output = None
    if 'error' not in st.session_state:
        st.session_state.error = None
    if 'results_df' not in st.session_state:
        st.session_state.results_df = pd.DataFrame()

    dataset = dataset_selection()
    folder_path = dataset_data(dataset)
    output_dir = dataset_output(dataset)

    preprocess_data(output_dir, folder_path)

    selected_files = multifile_selection(dataset)

    algs = multialgorithm_selection()

    if selected_files:
        min_clusters = st.number_input("Minimum Number of Clusters", min_value=2, max_value=99, value=2)
        max_clusters = st.number_input("Maximum Number of Clusters", min_value=3, max_value=100, value=5)

        if st.button("Run Analysis on Selected Files"):
            global total_steps
            global progress_bar
            total_steps = ((max_clusters - min_clusters + 1) * len(algs)) * len(selected_files)
            progress_bar = st.progress(0)
            st.session_state.results_df = process_files_and_run_clustering(output_dir, min_clusters, max_clusters, selected_files, algs)
            if not st.session_state.results_df.empty:
                        progress_bar.progress(1.0)
                        st.success("Analysis complete!")

                        # Display results in a table
                        st.write("Clustering Results:")
                        st.dataframe(st.session_state.results_df)

                        # Create CSV download
                        csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name='clustering_results.csv',
                            mime='text/csv',
                        )      
