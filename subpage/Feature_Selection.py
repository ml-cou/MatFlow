import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import cross_validate
import streamlit as st
import numpy as np


def feature_selection(dataset, table_name, target_var, problem_type):

    try:
        if "prev_table" not in st.session_state:
            st.session_state.prev_table = table_name
        elif st.session_state.prev_table == table_name:
            feature_graph(st.session_state.df_result, st.session_state.df_all_result, problem_type,
                          st.session_state.dropped_columns)

            return
        else:
            st.session_state.prev_table = table_name
            del st.session_state.df_result
            del st.session_state.df_all_result
            del st.session_state.dropped_columns
    except:
        pass

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

    to_sort_df=df_result

    st.write('#')

    # st.info("")

    total_iterations = len(list_X)

    progress_bar = st.progress(0,text=':blue[Stage 1: Calculating scores for each feature]')

    for i in range(len(list_X)):

        scores = cross_validate(estimator, X_n[list_X[i]].values.reshape(-1, 1), Y_n, cv=kfold, scoring=scoring)
        progress_percentage = (i + 1) / total_iterations

        # Update the progress bar
        progress_bar.progress(progress_percentage,text=':blue[Stage 1: Calculating scores for each feature]'+('..')*(i%3))
        try:
            to_sort_df.loc[list_X[i]] = [
                round(scores['test_' + score].mean() * (1 if problem_type == 'classification' else -1), 4) for score in
                scoring]
        except Exception as e:
            st.error(f"Error while adding data to result dataframe: {str(e)}")
            return pd.DataFrame
    progress_bar.progress(1.0,"Stage 1 completed!")

    if problem_type == 'classification':
        to_sort_df = to_sort_df.sort_values('F1',ascending=False)
    else:
        to_sort_df=to_sort_df.sort_values('RMSE')

    list_X = to_sort_df.index.tolist()
    k = list_X[0]

    df_all_result = df_result

    st.write('#')

    progress_bar = st.progress(0,':blue[Stage 2: Dropping least important features]')

    dropped_columns = []

    all_column_data_first=pd.DataFrame()
    selected_column_data=pd.DataFrame()

    for i in range(len(list_X)):
        all_column_data_first[list_X[i]] = X_n[list_X[i]]
        selected_column_data[list_X[i]] = X_n[list_X[i]]

        if len(dropped_columns)>0:
            scores_all = cross_validate(estimator, all_column_data_first, Y_n, cv=kfold, scoring=scoring)
            scores_selected = cross_validate(estimator, selected_column_data, Y_n, cv=kfold, scoring=scoring)
            try:

                df_result.loc[list_X[i]] = [
                    round(scores_selected['test_' + score].mean() * (1 if problem_type == 'classification' else -1), 4) for score
                    in
                    scoring]
                df_all_result.loc[list_X[i]] = [
                    round(scores_all['test_' + score].mean() * (1 if problem_type == 'classification' else -1), 4) for score
                    in
                    scoring]
            except Exception as e:
                st.error(f"Error while adding data to result dataframe: {str(e)}")
                return pd.DataFrame
        else:
            scores_all = cross_validate(estimator, all_column_data_first, Y_n, cv=kfold, scoring=scoring)
            try:

                df_result.loc[list_X[i]] = [
                    round(scores_all['test_' + score].mean() * (1 if problem_type == 'classification' else -1), 4) for score
                    in
                    scoring]
                df_all_result.loc[list_X[i]] = [
                    round(scores_all['test_' + score].mean() * (1 if problem_type == 'classification' else -1), 4) for score
                    in
                    scoring]
            except Exception as e:
                st.error(f"Error while adding data to result dataframe: {str(e)}")
                return pd.DataFrame


        progress_percentage = (i + 1) / total_iterations

        # Update the progress bar
        progress_bar.progress(progress_percentage,text=':blue[Stage 2: Dropping least important features]'+ '..'*(i%3))

        # create a list to store dropped columns
        if i >= 1:
            if problem_type == 'regression':
                if df_result.loc[list_X[i], 'RMSE'] >= df_result.loc[k, 'RMSE']:
                    dropped_columns.append(list_X[i])  # add column name to list
                    df_result = df_result.drop(list_X[i])
                    selected_column_data=selected_column_data.drop(columns=list_X[i])
                else:
                    k = list_X[i]
            else:
                if df_result.loc[list_X[i], 'F1'] <= df_result.loc[k, 'F1']:
                    dropped_columns.append(list_X[i])  # add column name to list
                    df_result = df_result.drop(list_X[i])
                    selected_column_data=selected_column_data.drop(columns=list_X[i])
                else:
                    k = list_X[i]

    progress_bar.progress(1.0,"Feature selection process completed!")

    st.write('#')
    st.write('#')


        # display dropped columns and reasons in Streamlit frontend

    st.session_state.df_result = df_result
    st.session_state.df_all_result = df_all_result
    st.session_state.dropped_columns = dropped_columns

    feature_graph(df_result, df_all_result, problem_type, dropped_columns)

import plotly.graph_objects as go

def feature_graph(df_result, df_all_result, problem_type, dropped_columns):

    # st.write(df_result)
    if problem_type == "regression":
        # Define the chart and axis labels for regression
        matrices_to_display = st.multiselect('Select matrices to display', ['MAE', 'MAPE', 'MSE', 'RMSE'],
                                             default=['RMSE'])
        try:
            df_result = df_result.reset_index()
            df_all_result = df_all_result.reset_index()
        except:
            st.error('Can\'t perform operation successfully')
            return
    else:
        # Define the chart and axis labels for classification
        matrices_to_display = st.multiselect('Select matrices to display', ['Accuracy', 'Precision', 'Recall', 'F1'],
                                             default=['Accuracy'])
        try:
            df_result = df_result.reset_index()
            df_all_result = df_all_result.reset_index()
        except:
            st.error('Can\'t perform operation successfully')
            return

    df_result = df_result.rename(columns={'index': 'Features'})
    df_all_result = df_all_result.rename(columns={'index': 'Features'})

    merged_df = pd.merge(df_all_result, df_result, on='Features', how='outer', suffixes=('_Baseline', '_Improved'))

    if problem_type == "regression":
        try:
            merged_df=merged_df.sort_values('RMSE_Improved',ascending=False)
            merged_df = merged_df.reset_index()
        except:
            st.error('Can\'t perform operation successfully')
            return
    else:
        try:
            merged_df=merged_df.sort_values('F1_Improved')
            merged_df = merged_df.reset_index()
        except:
            st.error('Can\'t perform operation successfully')
            return

    st.write('#')

    data = pd.concat([df_result[['Features']], df_result[matrices_to_display]], axis=1)

    fig = go.Figure()

    for matrix_name in matrices_to_display:
        # Mask the missing values
        mask = np.isfinite(merged_df[f'{matrix_name}_Improved'])

        # Add a line plot trace for the baseline
        fig.add_trace(go.Scatter(x=merged_df['Features'], y=merged_df[f'{matrix_name}_Baseline'],
                                 name='Baseline',
                                 mode='lines+markers',
                                line=dict(color='#FF4949',dash='dot'),
                                 marker=dict(symbol='circle', size=4, color='#FF4949')
                                 ))

        # Add a line plot trace for the improved
        fig.add_trace(go.Scatter(x=merged_df['Features'][mask], y=merged_df[f'{matrix_name}_Improved'][mask],
                                 name='Improved', mode='lines+markers', line=dict(color='#19A7CE'),
                                 marker=dict(symbol='circle', size=5, color='#19A7CE')
                                 ))

    # Configure layout
    fig.update_layout(
        title='Comparison of Baseline and Improved',
        xaxis=dict(title='Features', tickangle=45),
        yaxis=dict(title=matrices_to_display[0]),
        autosize=True,  # Enables autosizing of the figure based on the container size
        hovermode='closest',  # Show data points on hover
        dragmode='zoom',  # Enable zooming and panning functionality
        width=1280,  # Set the width of the figure to 800 pixels
        height=820
        # showlegend=False  # Hide legend, as it may overlap with the plot
    )

    # Display the figure
    st.plotly_chart(fig)

    c0, col1, col2, c1 = st.columns([0.1, 4, 2, .1])

    with col1:
        st.subheader("Selected Features:")
        st.table(data)

    with col2:
        if len(dropped_columns) > 0:
            st.subheader("Dropped Features:")
            st.table(dropped_columns)
