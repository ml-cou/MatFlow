import time
from streamlit_option_menu import option_menu
import streamlit as st
from .navbar import vspace
from streamlit_extras.switch_page_button import switch_page
import pandas as pd
from modules.dataset import display
from pathlib import Path
from .Dataset_Analysis import ds_analysis
from .Dataset_Visualization import ds_visualization
from .Feature_Engineering import ds_feature_engineering
from modules.classes import data as dataaa

from .Final_Data_Analysis import ds_final_data_analysis
from feature import change_dtype


def run():
    if 'dataset' not in st.session_state:
        st.session_state["dataset"]=dataaa.Dataset()

    path = Path().absolute()
    if 'sample_taken' not in st.session_state:
        st.session_state.sample_taken = False

    class funFile:
        def __init__(self, file_name, file_data):
            self.file_name = file_name
            self.file_data = file_data

    def validiate_data(data, name):
        try:
            for i in st.session_state.project_files:
                if name == i.file_name:
                    if data.equals(i.file_data):
                        st.error('File already exists')
                        time.sleep(2)
                        st._rerun()
                    return False
            return True
        except:
            return False

    sample_data = [
        f"{path}/rawData/Deep4Chem/DB for chromophore_Sci_Data_rev02.csv",
        f"{path}/rawData/Deep4Chem/DoubleCheck-High Extinction.csv",
        f"{path}/rawData/Dyomics/Dyomics_2017.pdf",
        f"{path}/rawData/Dyomics/SmilesData.csv",
        f"{path}/rawData/PhotoChemCAD3/SmilesData.csv",
        f"{path}/rawData/Experimental_SMILES_Predictions.csv"
    ]

    if 'project_name' not in st.session_state:
        st.error('Create a project first')
        time.sleep(2)
        # switch_page('create_project')
    opn = []

    try:
        for i in st.session_state.project_files:
            opn.append(i.file_name)
    except:
        opn.append('No Files')

    try:
        with st.sidebar:
            selected = option_menu(st.session_state.project_name, opn, menu_icon="folder2-open",
                                   styles={"container": {"padding-top": "10px !important",
                                                     "background-color": "transparent"}})
            st.markdown('''
            <style>
            .css-110bie7
            {
            min-width: 200px !important;
            max-width: 400px !important;
            }
            </style>
            ''', unsafe_allow_html=True)

    except:
        pass
    if not selected == 'No Files':

        main_funtionality=['Dataset','EDA','Feature Engineering','Final Dataset','Pipeline','Model Building','Model Deployment','Reverse ML']
        selected_function=option_menu(None, main_funtionality, menu_icon="folder2-open",orientation="horizontal",
                    styles=
                    {"container": {"padding": "5px 2% 5px 2% !important",
                                   "font-size": "15px",
                                   "background-color": "#ECF2FF", "width": "100vw",
                                  "border-radius": "0rem", "display": "block !important",
                                  "margin": "0px", "z-index": "999991"}})


        for i in st.session_state.project_files:
            if i.file_name == selected:
                c0, col2, c1 = st.columns([0.5, 7, 0.3])
                with col2:
                    if selected_function==main_funtionality[0]:
                        ds_analysis(i)
                    elif selected_function==main_funtionality[1]:
                        ds_visualization(i)
                    elif selected_function==main_funtionality[2]:
                        ds_feature_engineering(i)
                    elif selected_function==main_funtionality[3]:
                        st.write(i.file_data)
                        # ds_final_data_analysis(i)
                    elif selected_function==main_funtionality[5]:
                        # st.write(i.file_data)
                        model_builder(i)
                    elif selected_function==main_funtionality[4]:
                        # st.write(i.file_data)
                        pipeline(i)







    with st.sidebar:
        new_file = st.file_uploader('Upload a file')
        vspace(2)
        if st.button('Upload'):
            if new_file:
                try:
                    if 'project_files' not in st.session_state:
                        st.session_state.project_files = []
                        dt = pd.read_csv(new_file)
                        st.session_state.project_files.append(funFile(new_file.name, dt))
                        st.session_state["dataset"].add(st.session_state.project_files[-1].file_name,
                                                        st.session_state.project_files[-1].file_data)

                    else:
                        data = pd.read_csv(new_file)
                        if validiate_data(data, new_file.name):
                            st.session_state.project_files.append(funFile(new_file.name, data))
                            st.session_state["dataset"].add(st.session_state.project_files[-1].file_name,
                                                            st.session_state.project_files[-1].file_data)

                except:
                    pass
                print(st.session_state.dataset)
                st._rerun()
        rad = st.radio('Use default data set', options=['Yes', 'No'], index=1)
        if rad == 'Yes' and st.session_state.sample_taken == False:
            st.session_state.sample_taken = True
            for i in sample_data:
                x = i.rfind('\\')
                y = i.rfind('/')
                if x < y:
                    x = y
                x += 1
                if i.endswith('.csv'):
                    new_file_name = i[x:]
                    dt = pd.read_csv(i)
                    if 'project_files' not in st.session_state:
                        st.session_state.project_files = []
                        st.session_state.project_files.append(funFile(new_file_name, dt))
                        st.session_state["dataset"].add(st.session_state.project_files[-1].file_name,st.session_state.project_files[-1].file_data)
                    elif validiate_data(dt, new_file_name):
                        st.session_state.project_files.append(funFile(new_file_name, dt))
                        st.session_state["dataset"].add(st.session_state.project_files[-1].file_name,st.session_state.project_files[-1].file_data)

            st._rerun()
