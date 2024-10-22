from tools import * 

def main():
    st.title("Graph Analysis")

    # Select dataset and file
    dataset = single_selection("Select dataset:", list(DATASETS_OUTPUT.keys()), 'graphing_dataset')
    selected_file = single_selection("Select file:", AVAILABLE_FILES, 'graphing_file')
    file_path = os.path.join(DATASETS_OUTPUT[dataset], selected_file)
    
    # Select dimensions for PCA
    pca_dims = single_selection("Select PCA dimensions:", [2, 3], 'graphing_pca_dimensions')
    
    # Select algorithm
    alg = single_selection("Select clustering algorithm:", list(ALGORITHM_LOOKUP.keys()), 'graphing_algorithm')
    
    # Select umber of clusters wanted
    n_clusters = None
    if alg in ['K-Means', 'Agglomerative', 'Spectral']:
        n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)

    # Display clustered titles after clustering
    show_clusters = st.checkbox("Show Clustered Titles")

    # Run clustering
    if st.button("Run Clustering"):
        (titles, feature_matrix) = generate_features(selected_file, file_path)
        clusters = run_alg(alg, feature_matrix, n_clusters)

        # Display output
        plot_clusters(feature_matrix, clusters, pca_dims, titles)
        display_metrics(feature_matrix, clusters)

        # Show titles if configured
        if show_clusters:
            show_clustered_titles(clusters, titles)
