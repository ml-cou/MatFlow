import random
import streamlit as st
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'login'
if st.session_state.current_page=='project_files':
    st.set_page_config(layout='wide', initial_sidebar_state='expanded')
else:
    st.set_page_config(layout='wide', initial_sidebar_state='collapsed')

from subpage.navbar import navbar,vspace,body_padding
from subpage import create_project, project_files

navbar()

if 'one' not in st.session_state:
    st.session_state.one = True
    st._rerun()



if st.session_state.current_page == 'login':
    vspace(10)
    c0,c1, c2 = st.columns([2,2,2])
    with c1:
        st.text_input('Enter email')
        st.text_input('Enter password')
        vspace(3)
        btn = st.button('Submit', type='primary')
        if btn:
            st.session_state.current_page = 'project'
            st._rerun()

if st.session_state.current_page == 'project':
    vspace(10)
    c0,col1, col2,c1 = st.columns([4,2, 2,3])
    with st.container():
        with col1:
            if st.button('Create New'):
                st.session_state.current_page = 'create_project'
                st._rerun()
        with col2:
            if st.button('Upload Files'):
                st.session_state.clicked = 'open_project'
                st._rerun()
    # button height width
    # st.markdown('''
    # <style>
    # .edgvbvh10
    # {
    # height:200px;
    # width:190px;
    # box-shadow: rgba(0, 0, 0, 0.15) 1.95px 1.95px 2.6px;
    # }
    #
    # </style>
    # ''', unsafe_allow_html=True)

    c0, col1, col2 = st.columns([1, 3,1])

if st.session_state.current_page == 'create_project':
    vspace(10)
    create_project.run()

if st.session_state.current_page == 'project_files':
    project_files.run()
    st.markdown('''
    <style>
    .main{
    background:white;
    }
    </style>''', unsafe_allow_html=True
                )


