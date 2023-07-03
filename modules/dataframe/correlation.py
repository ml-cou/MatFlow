import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from modules import utils


def correlation(data):
    st.title("Feature Correlation")

    num_var = utils.get_numerical(data)


    st.markdown("#")
    correlation_type = st.selectbox('Choose a correlation type', ["Pair to pair", "With respect to target"],
                                        key="correlation_type")

    col1, col2 = st.columns([6, 4])



    if correlation_type == "Pair to pair":
        correlation_var = col1.multiselect(
            "Columns",
            num_var,
            num_var,
            key="correlation_var"
        )
    else:
        correlation_var = col1.multiselect(
            "Columns",
            num_var,
            key="correlation_var"
        )
        correlation_var_target = col2.multiselect(
            "Target",
            num_var,
            default=[num_var[0]],
            key="correlation_var_target"
        )

    col1, col2, col3 = st.columns([4, 4, 2.02])
    correlation_method = col1.selectbox(
        "Method",
        ["pearson", "kendall", "spearman"],
        key="correlation_method"
    )

    display_type = col2.selectbox(
        "Display Type",
        ["Table", "Heatmap", "Feature Pair"],
        key="correlation_display_type"
    )

    if correlation_var:
        if display_type == "Table":
            col3.markdown("#")
            bg_gradient = col3.checkbox("Gradient", key="correlation_bg_gradient")
        elif display_type == "Heatmap":
            col3.markdown("#")
            annot = col3.checkbox("Annotate", key="correlation_annot")
        else:
            col3.markdown("#")
            bg_gradient = col3.checkbox("Gradient", key="correlation_bg_gradient")
    else:
        bg_gradient = col3.checkbox("Gradient", key="correlation_bg_gradient")

    if correlation_type == "Pair to pair":
        correlation_data = data[correlation_var].corr(method=correlation_method)
    else:
        correlation_data = data[correlation_var + correlation_var_target].corr(method=correlation_method)[correlation_var_target]
        correlation_data=correlation_data.T

    if display_type == "Table":
        display_table(correlation_data, bg_gradient)
    elif display_type == "Heatmap":
        display_heatmap(correlation_data, annot)
    else:
        display_pair(correlation_data, bg_gradient)


def display_table(correlation_data, bg_gradient):

    if bg_gradient:
        st.dataframe(correlation_data.style.background_gradient())
    else:
        st.dataframe(correlation_data)

    csv_data = correlation_data.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name=f"correlation.csv",
        mime="text/csv"
    )


def display_heatmap(correlation_data, annot):
    fig, ax = plt.subplots()

    decimal = 0
    if annot:
        col1, _ = st.columns([4, 6])
        decimal = col1.number_input(
            "Decimal",
            1, 3, 3,
            key="decimal_value"
        )

    ax = sns.heatmap(correlation_data.round(2), annot=annot, fmt=f".{int(decimal)}f")
    ax.set_title("Feature Correlation Heatmap", pad=20)
    st.pyplot(fig)


def display_pair(correlation_data, bg_gradient):
    features = correlation_data.columns.to_list()
    features.insert(0, "-")

    col1, col2, col3 = st.columns([3.8, 3.8, 2.4])
    feature1 = col1.selectbox(
        "Feature 1 Filter",
        features,
        key="feature_pair1"
    )

    feature2 = col2.selectbox(
        "Feature 2 Filter",
        features,
        key="feature_pair2"
    )

    higher_than = col3.number_input(
        "Correlation higher than",
        0.0, 1.0, 0.0,
        key="correlation_higher_than"
    )

    col1, col2, _ = st.columns([2.5, 2.5, 5])
    drop_perfect = col1.checkbox("Drop Perfect", key="correlation_drop_perfect")
    convert_abs = col2.checkbox("Absolute Value", key="convert_absolute")

    if convert_abs:
        # convert to absolute value to take negative correlation into consideration and then sort by the highest correlation
        sorted_corr = correlation_data.abs().unstack().sort_values(ascending=False).reset_index()
    else:
        sorted_corr = correlation_data.unstack().sort_values(ascending=False).reset_index()
    sorted_corr.rename(
        columns={
            "level_0": "Feature 1",
            "level_1": "Feature 2",
            0: 'Correlation Coefficient'
        }, inplace=True
    )

    if drop_perfect:
        sorted_corr = sorted_corr.drop(sorted_corr[sorted_corr['Correlation Coefficient'] == 1.0].index)

    if higher_than:
        sorted_corr = sorted_corr[sorted_corr['Correlation Coefficient'] > higher_than].reset_index(drop=True)

    if feature1 != "-" and feature2 == "-":
        sorted_corr = sorted_corr.loc[sorted_corr["Feature 1"] == feature1].reset_index(drop=True)
    elif feature1 == "-" and feature2 != "-":
        sorted_corr = sorted_corr.loc[sorted_corr["Feature 2"] == feature2].reset_index(drop=True)
    elif feature1 != "-" and feature2 != "-":
        if feature1 == feature2:
            # drop observation with same features but different column
            sorted_corr.drop(sorted_corr.iloc[1::2].index, inplace=True)
            sorted_corr = sorted_corr.loc[
                (sorted_corr["Feature 1"] == feature1) | (sorted_corr["Feature 2"] == feature2)].reset_index(drop=True)
        else:
            sorted_corr = sorted_corr.loc[
                (sorted_corr["Feature 1"] == feature1) | (sorted_corr["Feature 2"] == feature2)].reset_index(drop=True)

    else:
        sorted_corr = sorted_corr.drop(sorted_corr.iloc[1::2].index).reset_index(drop=True)

    if bg_gradient:
        st.dataframe(sorted_corr.style.background_gradient())
    else:
        st.dataframe(sorted_corr)

    csv_data = sorted_corr.to_csv(index=False)
    
    st.download_button(
        label="Download CSV",
        data=csv_data,
        file_name=f"correlation.csv",
        mime="text/csv"
    )
