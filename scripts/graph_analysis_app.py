import json
import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import os

# Function to create a NetworkX graph based on a threshold
def create_graph(data, threshold):
    G = nx.Graph()
    for file, related_files in data.items():
        G.add_node(file)
        for related_file, score in related_files:
            if score > threshold:
                G.add_node(related_file)
                G.add_edge(file, related_file, weight=score)
    return G

# Function to create a DataFrame for the table output with percentage
def create_table(G, selected_files):
    table_data = []
    for file in selected_files:
        if file in G.nodes():  # Check if the node exists in the graph
            connections = [(neighbor, G.edges[file, neighbor]['weight']) for neighbor in G.neighbors(file)]
            connections = sorted(connections, key=lambda x: x[1], reverse=True)[:10]  # Sort by score and limit to 10
            row = [f"{file}"] + [f"{conn[0]} ({conn[1]*100:.2f}%)" for conn in connections]
            table_data.append(row)
        else:
            # If the file is not in the graph, add a row with no connections
            table_data.append([f"{file}"] + ["No connections found"])

    # Create a DataFrame with each row's first element as the index
    df = pd.DataFrame(table_data)
    return df

def plot_graph(G, selected_files, title, color='blue', selected_color='limegreen'):
    # Create a subgraph for the selected files and their relationships
    subgraph = nx.Graph()
    for file in selected_files:
        if file in G.nodes():
            subgraph.add_node(file)
            for neighbor in G.neighbors(file):
                if G.edges[file, neighbor]['weight'] > 0:
                    subgraph.add_node(neighbor)
                    subgraph.add_edge(file, neighbor, weight=G.edges[file, neighbor]['weight'])

    # Create the graph visualization
    pos = nx.spring_layout(subgraph, seed=42)  # for better layout

    # Create edges
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color=color),
        hoverinfo='none',
        mode='lines')

    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    # Create nodes
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text' if show_labels else 'markers',
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            color=[],
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        )
    )

    for node in subgraph.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([node])
        if node in selected_files:
            node_trace['marker']['color'] += tuple([selected_color])  # Selected files in a different color
        else:
            node_trace['marker']['color'] += tuple([color])

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=title,
                        titlefont=dict(size=16),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper"
                        )],
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )
    return fig

# Get environment variables
corpus_path = os.getenv('CORPUS_PATH', '/data/corpus')  # Optional default value
output_path = os.getenv('OUTPUT_PATH', '/app/output')  # Optional default value CHANGED

corpus_path = st.sidebar.text_input("Enter Input Directory Path", corpus_path)
output_path = st.sidebar.text_input("Enter Output Directory Path", output_path)

# Step 1: Load the JSON data
with open(output_path + '/top_related_files_keywords.json', 'r') as file:
    keyword_data = json.load(file)

with open(output_path + '/top_related_files_similarity.json', 'r') as file:
    document_data = json.load(file)

# Step 2: Streamlit interface
st.title('File Relationships Graph')

# Dropdown for selecting files, ordered alphabetically
common_files = sorted(set(keyword_data.keys()).union(set(document_data.keys())))

# Print out the total number of files
st.write(f"Total number of files: {len(common_files)}")

# Checkbox for selecting all files
select_all = st.checkbox("Select All Files")

# Adjust the selected files based on the select all checkbox
if select_all:
    selected_files = common_files
else:
    selected_files = st.multiselect('Select file(s) to visualize', common_files)

# Toggle for showing/hiding labels
show_labels = st.checkbox('Show labels', value=True)

# Step 3: Create and display the graphs and tables if files are selected
if selected_files:
    # Slider for adjusting keyword similarity threshold
    keyword_threshold = st.slider('Threshold for Keyword Similarity (Graph 1)', 0.0, 1.0, 0.30)
    keyword_graph = create_graph(keyword_data, keyword_threshold)
    
    st.write("Graph 1: Keyword Similarity")
    fig_keyword = plot_graph(keyword_graph, selected_files, f'Keyword Similarity (Threshold: {keyword_threshold})', color='blue', selected_color='purple')
    st.plotly_chart(fig_keyword)
    
    # Create and display the table for keyword similarity
    st.write("Table 1: Keyword Similarity Connections")
    df_keyword = create_table(keyword_graph, selected_files)
    st.dataframe(df_keyword)

    # Slider for adjusting document similarity threshold
    document_threshold = st.slider('Threshold for Document Similarity (Graph 2)', 0.0, 1.0, 0.70)
    document_graph = create_graph(document_data, document_threshold)
    
    st.write("Graph 2: Document Similarity")
    fig_document = plot_graph(document_graph, selected_files, f'Document Similarity (Threshold: {document_threshold})', color='red', selected_color='limegreen')
    st.plotly_chart(fig_document)

    # Create and display the table for document similarity
    st.write("Table 2: Document Similarity Connections")
    df_document = create_table(document_graph, selected_files)
    st.dataframe(df_document)

    # Expander for displaying content of selected files if not selecting all files
    if not select_all:
        st.subheader("Selected Files Content")
    
        for selected_file in selected_files:
            file_path = os.path.join(corpus_path, selected_file)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as file:
                    file_content = file.read()

                with st.expander(f"Content of {selected_file}"):
                    st.write(file_content)
            else:
                st.warning(f"File {selected_file} not found in the directory {corpus_path}.")

else:
    st.write("Please select a file to visualize its relationships.")
