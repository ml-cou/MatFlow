import streamlit as st
import time
import pandas as pd
from modules import utils
import numpy as np
from modules.feature import encoding, imputation, scaling, creation, dropping, change_dtype
from modules.feature import change_fieldname
from .navbar import vspace

def ds_feature_engineering(ds):
    if 'save_as' not in st.session_state:
        st.session_state.save_as = True
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

    try:

        data = ds.file_data
        data_opt = ds.file_name
    except KeyError:
        st.header("No Dataset Found")
        st.stop()

    except Exception as e:
        st.warning(e)
        st.stop()

    menus = ["Add/Modify", "Change Dtype","Alter Field Name", "Imputation", "Encoding", "Scaling", "Drop Column","Drop Rows","Merge Dataset"]
    tabs = [tab for tab in st.tabs(menus)]

    with tabs[0]:
        creation.creation(data, data_opt)

    with tabs[1]:
        change_dtype.change_dtype(data, data_opt)

    with tabs[2]:
        change_fieldname.change_field_name(data, data_opt)

    with tabs[3]:
        imputation.imputation(data, data_opt)

    with tabs[4]:
        encoding.encoding(data, data_opt)

    with tabs[5]:
        scaling.scaling(data, data_opt)

    with tabs[6]:
        dropping.dropping(data, data_opt)
    with tabs[7]:
        dropping.drop_raw(data,data_opt)
    if len(st.session_state.project_files)<=1:
        with tabs[8]:
            st.header('Please add at least two datasets !')
    else:
        with tabs[8]:
            li=[]
            for i in st.session_state.project_files:
                if i.file_name!=ds.file_name:
                    li.append(i.file_name)
            merge_name=st.selectbox('Select Dataset You Wanna Merge With',li)

            file_name = st.text_input('New Dataset Name', autocomplete='off')
            how = st.selectbox('How', ['left', 'right', 'outer', 'inner', 'cross'], index=3)
            left_on = st.selectbox("Select column name for left dataframe:", ds.file_data.columns)
            right_on = st.selectbox("Select column name for right dataframe:", st.session_state.dataset.data[merge_name].columns)
            vspace(5)
            if st.button('Merge', type='primary'):
                if len(file_name) == 0:
                    st.error('Name can\'t be empty')
                else:
                    tmp = pd.DataFrame(ds.file_data)
                    for i in st.session_state.project_files:
                        if i.file_name == merge_name:
                            try:
                                temp2=tmp.merge(i.file_data, left_on=left_on,right_on=right_on, how=how)
                            except Exception as e:
                                st.warning(e)
                    if validiate_data(temp2, file_name):
                        st.session_state.project_files.append(funFile(file_name, temp2))
                        st.session_state.dataset.add(file_name,temp2)
                        st._rerun()
                        
