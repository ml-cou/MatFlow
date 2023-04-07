import streamlit as st
import time
import pandas as pd
from modules import utils
import numpy as np
from modules.feature import encoding, imputation, scaling, creation, dropping, change_dtype,feature_selection
from modules.feature import change_fieldname
from .navbar import vspace


def ds_feature_engineering(dataset, table_name):
    if 'save_as' not in st.session_state:
        st.session_state.save_as = True
    try:

        data = dataset[table_name]
        data_opt = table_name
    except KeyError:
        st.header("No Dataset Found")
        st.stop()

    except Exception as e:
        st.warning(e)
        st.stop()

    menus = ["Add/Modify", "Change Dtype", "Alter Field Name", "Imputation", "Encoding", "Scaling", "Drop Column",
             "Drop Rows", "Merge Dataset", "Append Dataset","Feature Selection"]
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
        dropping.drop_raw(data, data_opt)
    if len(dataset) <= 1:
        with tabs[8]:
            st.header('Please add at least two datasets !')
    else:
        with tabs[8]:
            li = []
            for i in st.session_state.dataset.data.keys():
                if i != table_name:
                    li.append(i)
            merge_name = st.selectbox('Select Dataset You Wanna Merge With', li)

            file_name = st.text_input('New Dataset Name', autocomplete='off')
            how = st.selectbox('How', ['left', 'right', 'outer', 'inner', 'cross'], index=3)
            left_on = st.selectbox("Select column name for left dataframe:", data.columns)
            right_on = st.selectbox("Select column name for right dataframe:",
                                    st.session_state.dataset.data[merge_name].columns)
            vspace(5)
            if st.button('Merge', type='primary'):
                if len(file_name) == 0:
                    st.error('Name can\'t be empty')
                else:
                    tmp = pd.DataFrame(data)
                    try:
                        temp2 = tmp.merge(dataset[file_name], left_on=left_on, right_on=right_on, how=how)
                        st.session_state.dataset.add(file_name, temp2)
                        st._rerun()
                    except Exception as e:
                        st.warning(e)
    if len(dataset) <= 1:
        with tabs[9]:
            st.header('Please add at least two datasets !')
    else:
        with tabs[9]:
            li = []
            for i in st.session_state.dataset.data.keys():
                if i != table_name:
                    li.append(i)

            append_name = st.selectbox('Select Dataset You Wanna Append', li)
            file_name = st.text_input('New Dataset Name', autocomplete='off', key='append')
            st.write('#')
            if st.button('Append', type='primary'):
                if len(file_name) == 0:
                    st.error('Name can\'t be empty')
                else:
                    tmp = pd.DataFrame(data)
                    try:
                        temp2 = tmp.append(dataset.data[append_name])
                        temp2 = temp2.reset_index()
                        st.session_state.dataset.add(file_name, temp2)
                    except Exception as e:
                        st.warning(e)
                st._rerun()
    with tabs[10]:
        feature_selection.feature_selection(data,data_opt)

