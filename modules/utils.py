import streamlit as st
import time

from modules.classes import pipeline


class funFile:
    def __init__(self, file_name, file_data):
        self.file_name = file_name
        self.file_data = file_data


def validiate_data(name, data):
    try:
        if len(name)==0:
            st.warning('Field Name Can\'t be Empty')
            return False
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

def dataset_opt(list_data, default_idx):
    col1, _ = st.columns([4, 6])
    data_opt = col1.selectbox(
        "Choose Data",
        list_data,
        default_idx,
        key="display_data_opt"
    )

    return data_opt


def get_variables(data, add_hypen=False):
    variables = data.columns.to_list()

    if add_hypen:
        variables.insert(0, "-")

    return variables


def get_categorical(data, add_hypen=False):
    cat_var = data.loc[:, data.dtypes == 'object'].columns.to_list()

    if add_hypen:
        cat_var.insert(0, "-")

    return cat_var


def get_numerical(data, add_hypen=False):
    num_var = data.loc[:, data.dtypes != 'object'].columns.to_list()

    if add_hypen:
        num_var.insert(0, "-")

    return num_var


def get_low_cardinality(data, max_unique=10, add_hypen=False):
    variables = data.loc[:, (data.nunique() <= max_unique)].columns.to_list()

    if add_hypen:
        variables.insert(0, "-")

    return variables


def get_null(data):
    null_var = data.loc[:, data.isna().sum() > 0].columns.to_list()

    return null_var


def get_dtypes(data):
    dtypes = data.dtypes.values.astype(str)

    return dtypes


def get_nunique(data, column=None):
    n_unique = data.nunique().to_list()
    if column:
        idx = data.columns.get_loc(column)
        n_unique = n_unique[idx]

    return n_unique


def update_value(data_opt, new_value,temp_name,save_as=False):
    try:
        if save_as:
            if validiate_data(temp_name, new_value):
                st.session_state.dataset.add(temp_name, new_value)
                st.session_state.project_files.append(funFile(temp_name, new_value))
            else:
                return
        else:
            st.session_state["dataset"].data[data_opt] = new_value
            for i in st.session_state.project_files:
                if i.file_name == data_opt:
                    i.file_data = new_value
                    break
    except Exception as e:
        st.error(e)


def add_pipeline(name, class_obj):
    if "pipeline" in st.session_state:
        st.session_state["pipeline"].add(name, class_obj)
    else:
        st.session_state["pipeline"] = pipeline.Pipeline()
        st.session_state["pipeline"].add(name, class_obj)


def split_xy(data, target_var):
    X = data.drop(target_var, axis=1)
    y = data[target_var]

    return X, y


def rerun(delay=1.5):
    time.sleep(delay)
    st.experimental_rerun()
