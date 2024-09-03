import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import distance

# Function to print the top words for each topic
def print_top_words(model, feature_names, n_top_words):
    topic_keywords = []
    for topic_idx, topic in enumerate(model.components_):
        keywords = " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        topic_keywords.append(keywords)
    return topic_keywords

# Function to load or calculate t-SNE results
def load_or_calculate_tsne(documents, num_topics, n_top_words):
    tsne_file = f"{output_directory_path}/tsne_results_topics_{num_topics}_words_{n_top_words}.pkl"
    
    if os.path.exists(tsne_file):
        with open(tsne_file, 'rb') as f:
            df_tsne = pickle.load(f)
        st.success(f"Loaded t-SNE results from {tsne_file}")
    else:
        st.error("Generate the t-SNE file")
    
    return df_tsne

# Function to find the 5 closest files
def find_closest_files(df_tsne, selected_files):
    closest_files = []
    for selected_file in selected_files:
        selected_row = df_tsne[df_tsne['filename'] == selected_file]
        other_rows = df_tsne[df_tsne['filename'] != selected_file]
        other_rows['distance'] = other_rows.apply(lambda row: distance.euclidean(
            [row['x'], row['y']],
            [selected_row['x'].values[0], selected_row['y'].values[0]]
        ), axis=1)
        closest_files_df = other_rows.nsmallest(5, 'distance')[['filename', 'distance']]
        closest_files_df['selected_file'] = selected_file
        closest_files.append(closest_files_df)
    return pd.concat(closest_files)

num_topics = st.sidebar.number_input('Number of topics', min_value=1, value=25)
n_top_words = st.sidebar.number_input('Number of top words per topic', min_value=1, value=10)

# Load and preprocess the documents
documents = []
filenames = []
directory_path = st.sidebar.text_input("Enter directory path", "../data/privacy_law_corpus-original_english_text_files")
output_directory_path = st.sidebar.text_input("Enter output directory path", "./output")

if directory_path:
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    text = file.read()
            except UnicodeDecodeError as e:
                st.warning(f"Skipping file {filename} due to decoding error: {e}")
                continue  # Skip to the next file
            
            text = text.replace('\n', ' ').replace('\r', ' ')
            documents.append(text)
            filenames.append(filename)

    # Sort filenames alphabetically
    filenames.sort()

    # App title and sidebar inputs
    st.title(f"LDA Topic Clustering and Visualization ({len(filenames)} Files)")

    # Load or calculate the t-SNE results
    df_tsne = load_or_calculate_tsne(documents, num_topics, n_top_words)

    # Create a second plot with clusters highlighted by topic
    fig_cluster = px.scatter(
        df_tsne, x='x', y='y', color='dominant_topic',
        labels={'color': 'Topic: Keywords'},
        hover_data={'filename': True, 'topic_keywords': True, 'dominant_topic': True},
        title='Cluster Visualization by Topic'
    )
    
    fig_cluster.update_traces(
        marker=dict(size=10),
        hovertemplate="<b>%{customdata[0]}</b><br>Topic: %{customdata[2]}<br>Keywords: %{customdata[1]}<extra></extra>"
    )

    st.plotly_chart(fig_cluster)

    # Streamlit multiselect widget for file selection
    selected_files = st.multiselect("Select files to highlight", filenames)

    # Define colors based on selection
    df_tsne['color'] = 'blue'  # Default color
    df_tsne['color_label'] = 'File'  # Default legend label
    if selected_files:
        selected_topics = df_tsne[df_tsne['filename'].isin(selected_files)]['dominant_topic'].unique()
        df_tsne.loc[df_tsne['dominant_topic'].isin(selected_topics), 'color'] = 'lightblue'
        df_tsne.loc[df_tsne['dominant_topic'].isin(selected_topics), 'color_label'] = 'Cluster'
        df_tsne.loc[df_tsne['filename'].isin(selected_files), 'color'] = 'limegreen'
        df_tsne.loc[df_tsne['filename'].isin(selected_files), 'color_label'] = 'Selected File'

        # Find and mark the 5 closest files
        closest_files_df = find_closest_files(df_tsne, selected_files)
        df_tsne.loc[df_tsne['filename'].isin(closest_files_df['filename']), 'color'] = 'yellow'
        df_tsne.loc[df_tsne['filename'].isin(closest_files_df['filename']), 'color_label'] = 'Close File'

    # Plot the t-SNE results using plotly
    fig = go.Figure()

    # Adding points by categories to the legend
    for color, label in [('limegreen', 'Selected File'), ('yellow', 'Close File'), ('lightblue', 'Cluster'), ('blue', 'File')]:
        filtered_df = df_tsne[df_tsne['color'] == color]
        fig.add_trace(go.Scatter(
            x=filtered_df['x'],
            y=filtered_df['y'],
            mode='markers',
            marker=dict(color=color, size=10),
            text=filtered_df['filename'] + '<br>Topic: ' + filtered_df['dominant_topic'] + '<br>Keywords: ' + filtered_df['topic_keywords'],
            hovertemplate="<b>%{text}</b><extra></extra>",
            name=label
        ))

    # Update layout with legend
    fig.update_layout(
        title='t-SNE Visualization of LDA Topic Clusters',
        xaxis_title='t-SNE 1',
        yaxis_title='t-SNE 2',
        legend=dict(
            itemsizing='constant',
            title="Legend",
            traceorder="normal"
        )
    )

    st.plotly_chart(fig)

    # If files are selected, display the closest files
    if selected_files:
        st.subheader("5 Closest Files to Each Selected File")
        st.dataframe(closest_files_df)
        
        # Display a link, content, and download button for each selected file
        st.subheader("Selected Files Content")
        for selected_file in selected_files:
            file_path = os.path.join(directory_path, selected_file)
            with open(file_path, "r", encoding="utf-8") as file:
                file_content = file.read()
            
            with st.expander(f"Content of {selected_file}"):
                st.write(file_content)
            
        
