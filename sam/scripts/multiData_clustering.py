from tools import *

# For progress bar
total_steps = 0
step = 0
progress_bar = 0

# Initialize progress bar
def reset_progress(x):
    global total_steps
    global step
    global progress_bar

    total_steps = x
    step = 0
    progress_bar = st.progress(0)

# Update progress bar as algorithms run
def update_progress():
    global step
    global total_steps
    step += 1
    if step < total_steps:
        progress_bar.progress(step / total_steps)

# Gather data from selected files and run clustering algorithms
def process_files_and_run_clustering(dataset_name, output_dir, min_clusters, max_clusters, selected_files, algs):
    all_results = []
    for file_name in selected_files:
        (t, f) = generate_features(file_name, os.path.join(output_dir, file_name))
        for a in algs:
            if a in CLUSTER_NUM_ALGORITHMS:
                for i in range(min_clusters, max_clusters +1):
                    clusters = run_alg(a, f, i)
                    all_results.append(spreadsheet_metrics(dataset_name, file_name, a, f, clusters))
                    update_progress()
            else:
                clusters = run_alg(a, f, None)
                all_results.append(spreadsheet_metrics(dataset_name, file_name, a, f, clusters))
                update_progress()
    
    # Return results for display
    return pd.DataFrame(all_results)

# Main function for Streamlit app
def show_page():
    st.title("Cluster Testing")

    # Add session states to Streamlit
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'output' not in st.session_state:
        st.session_state.output = None
    if 'error' not in st.session_state:
        st.session_state.error = None
    if 'results_df' not in st.session_state:
        st.session_state.results_df = pd.DataFrame()

    # Select dataset and process data
    dataset = single_selection("Select dataset:", list(DATASETS_OUTPUT.keys()), 'dataset')
    folder_path = DATASETS_DATA[dataset]
    output_dir = DATASETS_OUTPUT[dataset]
    preprocess_data(output_dir, folder_path)
    
    # Wait until data has been processed
    if st.session_state.output:
        # Files to gather data from
        selected_files = multi_selection("Select files:", AVAILABLE_FILES, 'files')

        # Clustering algorithms to run
        algs = multi_selection("Select algorithms:", list(ALGORITHM_LOOKUP.keys()), 'algorithms')

        # Gather number of clusters only if necessary
        if not set(algs).isdisjoint(set(CLUSTER_NUM_ALGORITHMS)):
            min_clusters = st.number_input("Minimum Number of Clusters", min_value=2, max_value=99, value=2)
            max_clusters = st.number_input("Maximum Number of Clusters", min_value=3, max_value=100, value=5)
        else:
            min_clusters = 1
            max_clusters = 1

        # Wait to run until user finishes selections
        if st.button("Run Analysis on Selected Files"):
            reset_progress(((max_clusters - min_clusters + 1) * len(algs)) * len(selected_files) + 4)
            st.session_state.results_df = process_files_and_run_clustering(dataset, output_dir, min_clusters, max_clusters, selected_files, algs)
            def highlight_grades(s):
                return [
                    'background-color: red' if v in ['F', 'E'] else
                    'background-color: yellow' if v in ['D', 'C'] else
                    'background-color: green' if v in ['A', 'B'] else ''
                    for v in s
                ]

            styled = st.session_state.results_df.style.apply(highlight_grades, subset=['Grade'])
            # st.write(styled)
            # Display results when clustering algorithms complete
            if not st.session_state.results_df.empty:
                        progress_bar.progress(1.0)
                        st.success("Analysis complete!")

                        # Display results in a table
                        st.write("Clustering Results:")
                        st.write(styled)
                        # st.dataframe(st.session_state.results_df)

                        # Create CSV to download
                        csv = st.session_state.results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name='clustering_results.csv',
                            mime='text/csv',
                        )

                        # Reset page
                        if st.button("Reset"):
                            st.experimental_rerun()     
