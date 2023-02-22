import time
from .navbar import vspace
import streamlit as st


def run():
    st.markdown('<h3>Create New Project</h3>', unsafe_allow_html=True)
    project_name = st.text_input('Project Name')
    type = st.radio('File type', options=['Dataset', 'Smiles'], horizontal=True)
    vspace(3)
    if st.button('Create', type='primary'):
        if len(project_name) == 0:
            st.warning('Project name can\'t be empty.')
        else:
            done = False
            try:
                st.session_state.project_name = project_name
                st.session_state.project_type = type
                done = True
            except:
                st.error('Please try again')
                time.sleep(2)
                st._rerun()
            if done:
                st.session_state.current_page = 'project_files'
                st._rerun()
    st.markdown('''
        <style>
        .css-91z34k
        {
        padding-right:500px;
        }
        </style>
        ''', unsafe_allow_html=True)
