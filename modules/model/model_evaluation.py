import numpy as np
import pandas as pd
import streamlit as st
from lazypredict.Supervised import LazyRegressor, LazyClassifier
from matplotlib import pyplot as plt
from modules.utils import split_xy
import seaborn as sns

def model_evaluation():
    try:
        train_data= st.session_state.dataset.get_data(st.session_state.splitted_data['train_name'])
        test_data=st.session_state.dataset.get_data(st.session_state.splitted_data['test_name'])
        target_var=st.session_state.splitted_data['target_var']
        X_train,y_train=split_xy(train_data,target_var)
        X_test,y_test=split_xy(test_data,target_var)
        target_var_type=train_data[target_var].dtype
    except:
        st.header("Properly Split Dataset First")
        return

    st.write("ML type: ")
    col1, col2 = st.columns([4, 4])
    options = ["Regresion", "Classification"]

    if target_var_type == "float64" or target_var_type == "int64":
        st.write(f"{target_var} is numerical")
        default_option = "Regresion"
        with col1:
            selected_option = st.radio("", options, index=options.index(default_option))

        reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
        models, predictions = reg.fit(X_train, X_test,y_train, y_test)
        st.dataframe(models)

        try:

            st.header("LazyRegressor Results - R-Squared Comparison")
            plt.figure(figsize=(4, 8))
            sns.set_theme(style="whitegrid")
            ax = sns.barplot(y=models.index, x="R-Squared", data=models)
            ax.set(xlim=(0, 1))
            st.pyplot(plt)
        except:
            st.error('Choose target variable properly.')

        # st.write(predictions)

    else:
        default_option = "Classification"
        with col1:
            selected_option = st.radio("", options, index=options.index(default_option))
            st.write(f"{target_var} is not numerical")

        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_train, X_test,y_train, y_test)
        st.dataframe(models)

        try:
            st.header("Accuracy Comparison of Models")
            plt.figure(figsize=(4, 8))
            sns.set_theme(style="whitegrid")
            ax = sns.barplot(y=models.index, x="Accuracy", data=models)
            ax.set(xlim=(0, 1))
            plt.xticks(rotation=90)
            st.pyplot(plt)
        except:
            st.error('Choose target variable properly.')
