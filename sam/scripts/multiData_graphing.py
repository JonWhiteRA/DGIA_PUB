from tools import * 

def main():
    st.title("Graph Analysis")

    # Get user input
    dataset = dataset_selection()
    (selected_file, file_path) = file_selection(dataset)
    pca_dims = pca_dims_selection()
    alg = algorithm_selection()
    
    # Number of clusters wanted
    n_clusters = None
    if alg in ['K-Means', 'Agglomerative', 'Spectral']:
        n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)

    # Run clustering
    if st.button("Run Clustering"):
        (feature_matrix, clusters, titles) = run_clustering(selected_file, file_path, alg, n_clusters)
        plot_clusters(feature_matrix, clusters, pca_dims, titles)
        display_metrics(feature_matrix, clusters)
        show_clustered_titles(clusters, titles)
