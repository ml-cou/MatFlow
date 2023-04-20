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
from .model_builder import model_builder
from .pipeline import pipeline
from .model_deployment import model_deployment
from .Feature_Selection import feature_selection,feature_graph

from .Final_Data_Analysis import ds_final_data_analysis
from feature import change_dtype


def run():
    if 'dataset' not in st.session_state:
        st.session_state["dataset"] = dataaa.Dataset()
        selected_table_name="No Files"
        st.session_state.selected_table_name='No Files'
        st.session_state.del_btn=True
    else:
        selected_table_name=st.session_state.selected_table_name

    opn=[i for i in st.session_state.dataset.data.keys()]

    if len(opn)==0:
        opn.append('No Files')

    path = Path().absolute()
    if 'sample_taken' not in st.session_state:
        st.session_state.sample_taken = False

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


    try:
        with st.sidebar:
            selected_table_name = option_menu(st.session_state.project_name, opn, menu_icon="folder2-open",
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
    if not selected_table_name == 'No Files':

        main_funtionality = ['Dataset', 'EDA', 'Feature Engineering', 'Final Dataset', 'Pipeline', 'Model Building',
                             'Model Deployment', 'Reverse ML']
        selected_function = option_menu(None, main_funtionality, menu_icon="folder2-open", orientation="horizontal",
                                        styles=
                                        {"container": {"padding": "5px 2% 5px 2% !important",
                                                       "font-size": "15px",
                                                       "background-color": "#ECF2FF", "width": "100vw",
                                                       "border-radius": "0rem", "display": "block !important",
                                                       "margin": "0px", "z-index": "999991"}})

        c0, col2, c1 = st.columns([0.5, 7, 0.3])
        with col2:
            if selected_function == main_funtionality[0]:
                ds_analysis(st.session_state.dataset.data, selected_table_name)
            elif selected_function==main_funtionality[1]:
                ds_visualization(st.session_state.dataset.data, selected_table_name)
            elif selected_function==main_funtionality[2]:
                ds_feature_engineering(st.session_state.dataset.data, selected_table_name)
            elif selected_function==main_funtionality[3]:
                for i,j in st.session_state.dataset.data.items():
                    with st.expander(i,expanded= True if i==selected_table_name else False):
                        st.dataframe(j)
                # ds_final_data_analysis(st.session_state.dataset.data, selected_table_name)
            elif selected_function==main_funtionality[5]:
                # st.write(i.file_data)
                model_builder(st.session_state.dataset.data, selected_table_name)
            elif selected_function==main_funtionality[4]:
                # st.write(i.file_data)
                pipeline(st.session_state.dataset, selected_table_name)
            elif selected_function==main_funtionality[6]:
                model_deployment()
            elif selected_function==main_funtionality[7]:
                feature_select_data=feature_selection(st.session_state.dataset.data, selected_table_name)
                if not feature_select_data.empty:
                    feature_graph(feature_select_data)





    with st.sidebar:
        new_file = st.file_uploader('Upload a file')
        vspace(2)
        if st.button('Upload'):
            if new_file:
                filename = new_file.name
                extension = filename.split('.')[-1]

                if extension == 'csv':
                    new_file_data = pd.read_csv(new_file)
                elif extension == 'xls' or extension == 'xlsx':
                    new_file_data = pd.read_excel(new_file)
                elif extension == 'tsv':
                    new_file_data = pd.read_csv(new_file, sep='\t')
                st.session_state["dataset"].add(new_file.name, new_file_data)
            st._rerun()
        st.write('#')

        if not st.session_state.del_btn:
            del_list=st.multiselect('Select Files to Delete',options=list(st.session_state.dataset.data.keys()))
            st.write('#')
        del_file=False
        if st.session_state.del_btn:
            if st.button('Delete',type='primary'):
                st.session_state.del_btn = False
                st._rerun()
        else:
            c0,c1=st.columns(2)
            with c1:
                if st.button('Cancel',type='primary'):
                    st.session_state.del_btn=True
                    st._rerun()
            with c0:
                if st.button('Submit'):
                    for i in del_list:
                        st.session_state.dataset.remove(i)
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
                new_file_data = pd.read_csv(st.session_state.dataset.data, selected_table_name)
                st.session_state["dataset"].add(new_file_name, new_file_data)
        st._rerun()

