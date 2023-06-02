import base64

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import cross_validate
import streamlit as st
import numpy as np


def feature_selection(dataset, table_name, target_var, problem_type):
    # try:
    #     if "prev_table" not in st.session_state:
    #         st.session_state.prev_table = table_name
    #         st.session_state.prev_target=target_var
    #     elif st.session_state.prev_table == table_name and st.session_state.prev_target==target_var:
    #         feature_graph(st.session_state.df_result, st.session_state.df_all_result, problem_type,
    #                       st.session_state.dropped_columns)
    #
    #         return
    #     else:
    #         st.session_state.prev_table = table_name
    #         del st.session_state.df_result
    #         del st.session_state.df_all_result
    #         del st.session_state.dropped_columns
    #         del st.session_state.prev_target
    # except:
    #     pass

    try:
        tab = dataset[table_name]
    except KeyError:
        st.error(f"The dataset '{table_name}' does not exist.")
        return pd.DataFrame

    if tab.isnull().values.any():
        st.header("There are Null values in the DataFrame.")
        return pd.DataFrame

    try:
        X_n = tab.drop(columns=target_var)
        Y_n = tab[target_var]
    except Exception as e:
        st.error(f"Error while getting input and output data: {str(e)}")
        return pd.DataFrame

    kfold = st.number_input("Enter the value for k-fold", value=2)

    if problem_type == 'regression':
        scoring = ['neg_mean_absolute_error', 'neg_mean_absolute_percentage_error',
                   'neg_mean_squared_error', 'neg_root_mean_squared_error']
        df_columns = ['MAE', 'MAPE', 'MSE', 'RMSE']
        estimator = ExtraTreesRegressor()
    else:
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        df_columns = ['Accuracy', 'Precision', 'Recall', 'F1']
        estimator = ExtraTreesClassifier()

    try:
        list_X = list(X_n.columns)
        df_result = pd.DataFrame(columns=df_columns)

    except Exception as e:
        st.error(f"Error while initializing variables: {str(e)}")
        return pd.DataFrame

    display_opt = st.radio("", ["All", "Custom", "None"], index=2, key="display_opt", horizontal=True)

    if display_opt == 'None':
        return

    if display_opt == "Custom":
        list_X = st.multiselect("Select features to display", list_X)

    to_sort_df = df_result.copy()

    st.write('#')

    # st.info("")

    total_iterations = len(list_X)

    progress_bar = st.progress(0, text=':blue[Stage 1: Calculating scores for each feature]')

    for i in range(len(list_X)):

        scores = cross_validate(estimator, X_n[list_X[i]].values.reshape(-1, 1), Y_n, cv=kfold, scoring=scoring)
        progress_percentage = (i + 1) / total_iterations

        # Update the progress bar
        progress_bar.progress(progress_percentage,
                              text=':blue[Stage 1: Calculating scores for each feature]' + ('..') * (i % 3))
        try:
            to_sort_df.loc[list_X[i]] = [
                round(scores['test_' + score].mean() * (1 if problem_type == 'classification' else -1), 4) for score in
                scoring]
        except Exception as e:
            st.error(f"Error while adding data to result dataframe: {str(e)}")
            return pd.DataFrame
    progress_bar.progress(1.0, "Stage 1 completed!")

    if problem_type == 'classification':
        to_sort_df = to_sort_df.sort_values('F1', ascending=False)
    else:
        to_sort_df = to_sort_df.sort_values('RMSE')

    list_X = to_sort_df.index.tolist()
    k = list_X[0]

    df_all_result = df_result.copy()
    df_all_result_group = df_result.copy()
    df_result_group = df_result.copy()


    st.write('#')

    progress_bar = st.progress(0, ':blue[Stage 2: Dropping least important features]')

    mx_len = len(list_X)

    dropped_columns_group = df_result.copy()

    k = str(5)

    for i in range(0, len(list_X), 5):
        selected_column_data = X_n[list_X[:min(i + 5, mx_len)]]

        scores_all = cross_validate(estimator, selected_column_data, Y_n, cv=kfold, scoring=scoring)
        try:

            df_result_group.loc[str(i + 5)] = [
                round(scores_all['test_' + score].mean() * (1 if problem_type == 'classification' else -1), 4) for score
                in
                scoring]
            df_all_result_group.loc[str(i + 5)] = [
                round(scores_all['test_' + score].mean() * (1 if problem_type == 'classification' else -1), 4) for score
                in
                scoring]
        except Exception as e:
            st.error(f"Error while adding data to result dataframe: {str(e)}")
            return pd.DataFrame

        if i >= 1:
            if problem_type == 'regression':
                if df_result_group.loc[str(i + 5), 'RMSE'] >= df_result_group.loc[k, 'RMSE']:
                    dropped_columns_group.loc[str(i + 5)]=df_result_group.loc[str(i + 5)]  # add column name to list
                    df_result_group = df_result_group.drop(str(i + 5))
                else:
                    k = str(i + 5)
            else:
                if df_result_group.loc[str(i + 5), 'F1'] <= df_result_group.loc[k, 'F1']:
                    dropped_columns_group.loc[str(i + 5)] = df_result_group.loc[str(i + 5)]
                    df_result_group = df_result_group.drop(str(i + 5))
                else:
                    k = str(i + 5)

        progress_percentage = (min(i + 5, total_iterations)) / total_iterations

        # Update the progress bar
        progress_bar.progress(progress_percentage,
                              text=':blue[Stage 2: Dropping least important features]' + '..' * (i % 3))

    progress_bar.progress(1.0, "Feature selection process completed!")

    dropped_columns = df_result.copy()

    all_column_data_first = pd.DataFrame()
    selected_column_data = pd.DataFrame()

    k = list_X[0]

    for i in range(len(list_X)):
        all_column_data_first[list_X[i]] = X_n[list_X[i]]
        selected_column_data[list_X[i]] = X_n[list_X[i]]

        if len(dropped_columns) > 0:
            scores_all = cross_validate(estimator, all_column_data_first, Y_n, cv=kfold, scoring=scoring)
            scores_selected = cross_validate(estimator, selected_column_data, Y_n, cv=kfold, scoring=scoring)
            try:

                df_result.loc[list_X[i]] = [
                    round(scores_selected['test_' + score].mean() * (1 if problem_type == 'classification' else -1), 4)
                    for score
                    in
                    scoring]
                df_all_result.loc[list_X[i]] = [
                    round(scores_all['test_' + score].mean() * (1 if problem_type == 'classification' else -1), 4) for
                    score
                    in
                    scoring]
            except Exception as e:
                st.error(f"Error while adding data to result dataframe: {str(e)}")
                return pd.DataFrame
        else:
            scores_all = cross_validate(estimator, all_column_data_first, Y_n, cv=kfold, scoring=scoring)
            try:

                df_result.loc[list_X[i]] = [
                    round(scores_all['test_' + score].mean() * (1 if problem_type == 'classification' else -1), 4) for
                    score
                    in
                    scoring]
                df_all_result.loc[list_X[i]] = [
                    round(scores_all['test_' + score].mean() * (1 if problem_type == 'classification' else -1), 4) for
                    score
                    in
                    scoring]
            except Exception as e:
                st.error(f"Error while adding data to result dataframe: {str(e)}")
                return pd.DataFrame

        progress_percentage = (i + 1) / total_iterations

        # Update the progress bar
        progress_bar.progress(progress_percentage,
                              text=':blue[Stage 2: Dropping least important features]' + '..' * (i % 3))

        # create a list to store dropped columns
        if i >= 1:
            if problem_type == 'regression':
                if df_result.loc[list_X[i], 'RMSE'] >= df_result.loc[k, 'RMSE']:
                    dropped_columns.loc[list_X[i]]=df_result.loc[list_X[i]]  # add column name to list
                    df_result = df_result.drop(list_X[i])
                    selected_column_data = selected_column_data.drop(columns=list_X[i])
                else:
                    k = list_X[i]
            else:
                if df_result.loc[list_X[i], 'F1'] <= df_result.loc[k, 'F1']:
                    dropped_columns.loc[list_X[i]]=df_result.loc[list_X[i]]
                    df_result = df_result.drop(list_X[i])
                    selected_column_data = selected_column_data.drop(columns=list_X[i])
                else:
                    k = list_X[i]

    progress_bar.progress(1.0, "Feature selection process completed!")

    st.write('#')
    st.write('#')


    st.session_state.df_result = df_result
    st.session_state.df_all_result = df_all_result
    st.session_state.dropped_columns = dropped_columns

    feature_graph(df_result_group, df_all_result_group, problem_type, dropped_columns_group, 'group')
    feature_graph(df_result, df_all_result, problem_type, dropped_columns, 'single')


import plotly.graph_objects as go


def feature_graph(df_result, df_all_result, problem_type, dropped_columns, keys):
    # st.write(df_result)

    try:
        df_result = df_result.reset_index()
        df_all_result = df_all_result.reset_index()
        dropped_columns=dropped_columns.reset_index()
    except:
        st.error('Can\'t perform operation successfully')
        return

    if problem_type == "regression":
        # Define the chart and axis labels for regression
        matrices_to_display = st.multiselect('Select matrices to display', ['MAE', 'MAPE', 'MSE', 'RMSE'],
                                             default=['RMSE'], key=keys)
    else:
        # Define the chart and axis labels for classification
        matrices_to_display = st.multiselect('Select matrices to display', ['Accuracy', 'Precision', 'Recall', 'F1'],
                                             default=['Accuracy'], key=keys)


    df_result = df_result.rename(columns={'index': 'Features'})
    df_all_result = df_all_result.rename(columns={'index': 'Features'})

    dropped_columns=dropped_columns.rename(columns={'index':'Features'})


    merged_df = pd.merge(df_all_result, df_result, on='Features', how='outer', suffixes=('_Baseline', '_Improved'))

    if problem_type == "regression":
        try:
            merged_df = merged_df.sort_values('RMSE_Improved', ascending=False)
            dropped_columns=dropped_columns.sort_values('RMSE', ascending=False)
            merged_df = merged_df.reset_index()
        except:
            st.error('Can\'t perform operation successfully')
            return
    else:
        try:
            merged_df = merged_df.sort_values('F1_Improved')
            dropped_columns=dropped_columns.sort_values('F1', ascending=False)
            merged_df = merged_df.reset_index()
        except:
            st.error('Can\'t perform operation successfully')
            return

    dropped_columns=dropped_columns.reset_index()

    st.write('#')

    data = pd.concat([df_result[['Features']], df_result[matrices_to_display]], axis=1)
    dropped_columns = pd.concat([dropped_columns[['Features']], dropped_columns[matrices_to_display]], axis=1)

    fig = go.Figure()

    for matrix_name in matrices_to_display:
        # Mask the missing values
        mask = np.isfinite(merged_df[f'{matrix_name}_Improved'])

        # Add a line plot trace for the baseline
        fig.add_trace(go.Scatter(x=merged_df['Features'], y=merged_df[f'{matrix_name}_Baseline'],
                                 name='Baseline',
                                 mode='lines+markers',
                                 line=dict(color='#FF4949', dash='dot'),
                                 marker=dict(symbol='circle', size=4, color='#FF4949')
                                 ))

        # Add a line plot trace for the improved
        fig.add_trace(go.Scatter(x=merged_df['Features'][mask], y=merged_df[f'{matrix_name}_Improved'][mask],
                                 name='Improved', mode='lines+markers', line=dict(color='#19A7CE'),
                                 marker=dict(symbol='circle', size=5, color='#19A7CE')
                                 ))

    if keys == 'group':
        dc = dict(title='<b>Features</b>',
                  tickangle=0,
                  tickmode='linear',
                  tick0=0,
                  dtick=5,
                  tickvals=list(range(0, len(merged_df['Features']), 5)),
                  ticktext=[merged_df['Features'].iloc[i] for i in range(0, len(merged_df['Features']), 5)]
                  )
    else:
        dc = dict(title='<b>Features</b>',
                  tickangle=45 if keys != 'group' else 0,
                  )

    # Configure layout
    fig.update_layout(
        title='Comparison of Baseline and Improved',
        xaxis=dc,
        yaxis=dict(title=f'<b>Root-Mean-Square-Error ({matrices_to_display[0]})</b>'),
        autosize=True,  # Enables autosizing of the figure based on the container size
        hovermode='closest',  # Show data points on hover
        dragmode='zoom',  # Enable zooming and panning functionality
        width=1480,  # Set the width of the figure to 800 pixels
        height=920
        # showlegend=False  # Hide legend, as it may overlap with the plot
    )

    # Display the figure
    st.plotly_chart(fig)

    c0, col1, col2, c1 = st.columns([0.1, 4, 4, .1])

    with col1:
        st.subheader("Selected Features:")
        if keys == 'group':
            data = data.rename(columns={'Features': 'Features-k'})

        st.table(data)
        csv_data = data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="dropped_columns.csv",
            mime="text/csv"
        )
    with col2:
        if len(dropped_columns) > 0:
            st.subheader("Dropped Features:")
            try:
                if keys == 'group':
                    dropped_columns = dropped_columns.rename(columns={'Features': 'Features-k'})
                st.table(dropped_columns)
                csv_data = dropped_columns.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="dropped_columns.csv",
                    mime="text/csv"
                )
            except:
                pass
