import streamlit as st
import subprocess
from multiData_clustering import show_page as show_processing
from multiData_graphing import main as show_graphs

# Sidebar menu for navigation
page = st.sidebar.selectbox("Select a page", ["Processing", "Graphs"])

# Show the selected page
if page == "Processing":
    show_processing()
elif page == "Graphs":
    show_graphs()