import random
import streamlit as st
from PIL import Image
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'login'
if st.session_state.current_page=='project_files':
    st.set_page_config(layout='wide',page_title='Start', initial_sidebar_state='expanded')
else:
    st.set_page_config(layout='wide',page_title='Start', initial_sidebar_state='collapsed')
image = Image.open('./Tiny scientists developing AI using machine learning.jpg')

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
            st._rerun()

if st.session_state.current_page == 'project':
    vspace(10)
    x1,x2,x3=st.columns([1,6,1])
    with x2:
        c0, col1, col2,c1 = st.columns([0.1,6, 5,0.5])
        with st.container():
            with col1:
                vspace(13)
                st.image(image)
            with col2:
                vspace(4)
                project,file=st.tabs(['New Project','Upload File'])
                with project:
                    vspace(10)
                    create_project.run()
                with file:
                    if st.button('Upload Files',type="primary"):
                        st.session_state.clicked = 'open_project'
                        st._rerun()


            st.markdown('''
            <style>
#root > div:nth-child(1) > div.withScreencast > div > div > div > section.main.css-k1vhr4.egzxvld5 > div.block-container.css-k1ih3n.egzxvld4 > div:nth-child(1) > div > div.css-ocqkz7.e1tzin5v4 > div.css-ml2xh6.e1tzin5v2 > div:nth-child(1) > div
            {
            background:white;
            }

            </style>
            ''', unsafe_allow_html=True)

            c0, col1, col2 = st.columns([1, 3, 1])

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
