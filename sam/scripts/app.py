import streamlit as st
from multiData_clustering import show_page as show_processing
from multiData_graphing import main as show_graphs

tab = st.tabs(["Processing", "Graphing"])

with tab[0]:
    show_processing()
with tab[1]:
    show_graphs()