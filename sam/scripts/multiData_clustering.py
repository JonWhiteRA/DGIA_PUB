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

total_steps = 0
step =0

def update_step():
    global step
    step += 1
    return step

def get_total_steps():
    global total_steps
    return total_steps

# Load JSON data based on file format
def load_data(filename):
    with open(filename) as file:
        data = json.load(file)

    if isinstance(data, list):
        if filename.endswith("output_entities.json") or filename.endswith("scored_keywords.json"):
            # Process lists of dictionaries
            processed_data = []
            for item in data:
                processed_data.append([
                    item.get("file1", ""),
                    item.get("file2", ""),
                    item.get("overlap_count", 0),
                    item.get("total_unique", 0),
                    item.get("crude_score", 0.0)
                ])
            return pd.DataFrame(processed_data, columns=["File1", "File2", "Overlap Count", "Total Unique", "Crude Score"])
        
    elif filename.endswith("entities.json"):
        freq_data = []
        for doc, keywords in data.items():
            for keyword, category, frequency in keywords:
                freq_data.append([keyword, frequency])
        return pd.DataFrame(freq_data, columns=["Keyword", "Frequency"])
    
    elif filename.endswith("keywords.json"):
        freq_data = []
        for doc, keywords in data.items():
            for keyword_info in keywords:
                keyword = keyword_info[0]
                frequency = keyword_info[1]
                freq_data.append([keyword, frequency])
        return pd.DataFrame(freq_data, columns=["Keyword", "Frequency"])
    
    elif filename.endswith("output_embeddings.json"):
        embeddings_data = []
        for doc, embeddings in data.items():
            if isinstance(embeddings, list):
                embeddings_data.append(embeddings)
        return pd.DataFrame(embeddings_data)

    # Add more conditions for other JSON formats as needed
    else:
        st.error(f"Unknown file format for {filename}.")
        return pd.DataFrame(columns=["Keyword", "Frequency"])

def run_single_clustering_algorithm(algorithm_name, algorithm, data, n_clusters=None):
    labels = algorithm.fit_predict(data)
    results = {
        'Algorithm': algorithm_name,
        'Clusters': len(set(labels)) - (1 if -1 in labels else 0),
        'Silhouette Score': silhouette_score(data, labels) if len(set(labels)) > 1 else None,
        'Davies-Bouldin': davies_bouldin_score(data, labels) if len(set(labels)) > 1 else None,
        'Calinski-Harabasz': calinski_harabasz_score(data, labels) if len(set(labels)) > 1 else None
    }
    if n_clusters is not None:
        results['Clusters'] = n_clusters
    progress_bar.progress(update_step() / get_total_steps())
    return results


def run_clustering(X, min_clusters, max_clusters, min_dim, max_dim, file_name):
    results = []

    # Iterate through selected dimensions
    for dim in range(min_dim, max_dim + 1):
        data_to_dim = X[:, :dim]  # Use the first 'dim' features

        # Run clustering algorithms that require number of clusters
        for n_clusters in range(min_clusters, max_clusters + 1):
            results.append(run_single_clustering_algorithm('GMM', GaussianMixture(n_components=n_clusters), data_to_dim, n_clusters))
            results.append(run_single_clustering_algorithm('K-Means', KMeans(n_clusters=n_clusters), data_to_dim, n_clusters))
            results.append(run_single_clustering_algorithm('Agglomerative', AgglomerativeClustering(n_clusters=n_clusters), data_to_dim, n_clusters))
            results.append(run_single_clustering_algorithm('Spectral', SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=42), data_to_dim, n_clusters))


        # Run clustering algorithms that do not require number of clusters (only once)
        results.append(run_single_clustering_algorithm('Mean Shift', MeanShift(), data_to_dim))
        results.append(run_single_clustering_algorithm('DBSCAN', DBSCAN(eps=0.5, min_samples=5), data_to_dim))
        results.append(run_single_clustering_algorithm('OPTICS', OPTICS(min_samples=5), data_to_dim))
        results.append(run_single_clustering_algorithm('Birch', Birch(), data_to_dim))

    return pd.DataFrame(results)


def process_files_and_run_clustering(output_dir, min_clusters, max_clusters, min_dim, max_dim, selected_files, lda_dim):
    all_results = []
    for file_name in selected_files:
        filename = os.path.join(output_dir, file_name)
        data = load_data(filename)

        if data.empty:
            continue  # Skip empty DataFrame if unknown format

        # Clustering logic for other files
        if file_name.endswith("output_embeddings.json"):
            frequency_data = data.values  # No reshaping needed
        else:
            frequency_data = data['Frequency'].values.reshape(-1, 1)

        results_df = run_clustering(frequency_data, min_clusters, max_clusters, min_dim, max_dim, file_name)
        all_results.append(results_df)
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()  # Return empty DataFrame if no results


def show_page():
    # Streamlit app interface
    st.title("Cluster and Dataset Testing")
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
    script_path = "corpus_processor_1.py"

    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'output' not in st.session_state:
        st.session_state.output = None
    if 'error' not in st.session_state:
        st.session_state.error = None
    if 'results_df' not in st.session_state:
        st.session_state.results_df = pd.DataFrame()

    selected_dataset = st.selectbox("Choose a dataset", list(data_folders.keys()))
    folder_path = data_folders[selected_dataset]
    output_dir = output_folders[selected_dataset]

    # Button to load data
    if st.button("Load Data"):
        st.session_state.is_running = True
        command1 = f"python {script_path} --output_dir {output_dir} {folder_path}"
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

            # Dropdown for selecting files after processing
            all_files = os.listdir(output_dir)
            filtered_files = [f for f in all_files if f.endswith('.json') and not (f.startswith('scored') or f.startswith('top'))]
            selected_files = st.multiselect("Select JSON files for analysis", filtered_files)

            if selected_files:
                min_clusters = st.number_input("Minimum Number of Clusters", min_value=2, max_value=99, value=2)
                max_clusters = st.number_input("Maximum Number of Clusters", min_value=3, max_value=100, value=5)
                lda_dim = st.number_input("LDA Dimensions", min_value=1, max_value=10, value=2)  # Allow user to specify LDA dimensions
                num_features = 1  # Since we only have frequency data

                # one-dimensional data right now
                min_dim = 1
                max_dim = 1

                if st.button("Run Analysis on Selected Files"):
                    total_steps = ((((max_dim - min_dim + 1) * (max_clusters - min_clusters + 1) + (max_dim - min_dim + 1) ) * 4) + 4) * len(selected_files)
                    progress_bar = st.progress(0)
                    st.session_state.results_df = process_files_and_run_clustering(output_dir, min_clusters, max_clusters, min_dim, max_dim, selected_files, lda_dim)
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

        if st.session_state.error:
            st.error("Error occurred:\n" + st.session_state.error)
