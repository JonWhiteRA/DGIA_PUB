import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import re
import streamlit as st

def create_graph_from_csv(csv_filename):
    df = pd.read_csv(csv_filename)
    G = nx.DiGraph()

    # Add nodes for each article
    for _, row in df.iterrows():
        article_num = row['Article Number']
        article_label = f"Article {article_num}"

        # Add the article node
        G.add_node(article_label, type='article')

        # Check for links in the 'Links' column
        if pd.notna(row['Links']):
            links = row['Links'].split(', ')
            for link in links:
                match = re.search(r'https://gdpr-info.eu/art-(\d+)-gdpr/', link)
                if match:
                    linked_article_num = match.group(1)
                    linked_article_label = f"Article {linked_article_num}"

                    # Add the linked article node if it doesn't exist
                    G.add_node(linked_article_label, type='article')
                    # Connect the current article to the linked article
                    G.add_edge(article_label, linked_article_label)

    return G

def visualize_graph(G, search_node=None):
    pos = nx.spring_layout(G, dim=3, k=5)

    # Create edge traces
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        line=dict(width=0.5, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    # Create node traces
    node_x = []
    node_y = []
    node_z = []
    node_labels = []
    node_colors = []
    node_hoverinfo = []

    for node in G.nodes():
        x, y, z = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        node_labels.append(node)

        # Get connected nodes for hover information
        connected_nodes = [n for n in G.neighbors(node)]
        connected_info = f"Connected to: {', '.join(connected_nodes)}" if connected_nodes else "No connections"

        # Highlight the searched node
        if search_node and search_node == node:
            node_colors.append('red')  # Highlight color
        else:
            node_colors.append('blue')  # Default color

        node_hoverinfo.append(f"{node}<br>{connected_info}")

    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        hoverinfo='text',
        text=node_labels,
        marker=dict(
            showscale=False,
            size=10,
            color=node_colors,
            line=dict(width=2)
        )
    )

    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='GDPR Articles Graph in 3D',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        scene=dict(
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        )
                    ))

    return fig

def main():
    st.title("GDPR Articles Graph Visualization")

    # Load the CSV and create the graph
    csv_filename = '/output/gdpr_articles.csv'
    G = create_graph_from_csv(csv_filename)

    # Search box for node highlighting
    search_term = st.text_input("Enter Article Number to Search:")
    search_node = f"Article {search_term}" if search_term else None

    # Visualize the graph
    fig = visualize_graph(G, search_node)
    st.plotly_chart(fig)

if __name__ == '__main__':
    main()
