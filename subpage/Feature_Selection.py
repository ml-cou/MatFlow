import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import cross_validate
import streamlit as st


def feature_selection(dataset, table_name, target_var, problem_type):
    try:
        if "prev_table" not in st.session_state:
            st.session_state.prev_table = table_name
        elif st.session_state.prev_table == table_name:
            feature_graph(st.session_state.df_result, st.session_state.initial_df, problem_type,
                          st.session_state.dropped_columns)
            return
        else:
            st.session_state.prev_table = table_name
            del st.session_state.df_result
            del st.session_state.initial_df
    except:
        pass

    try:
        tab = dataset[table_name]
    except KeyError:
        st.error(f"The dataset '{table_name}' does not exist.")
        return pd.DataFrame

    try:
        X_n = tab.drop(columns=target_var)
        Y_n = tab[target_var]
    except Exception as e:
        st.error(f"Error while getting input and output data: {str(e)}")
        return pd.DataFrame

    kfold = 5

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
        k = list_X[0]
        df_result = pd.DataFrame(columns=df_columns)

    except Exception as e:
        st.error(f"Error while initializing variables: {str(e)}")
        return pd.DataFrame

    initial_df = df_result

    for i in range(len(list_X)):
        first_n_column_list = list_X[:i + 1]
        first_n_column = X_n[first_n_column_list]

        scores = cross_validate(estimator, first_n_column, Y_n, cv=kfold, scoring=scoring)
        try:
            initial_df.loc[list_X[i]] = [
                round(scores['test_' + score].mean() * (1 if problem_type == 'classification' else -1), 4) for score in
                scoring]
        except Exception as e:
            st.error(f"Error while adding data to result dataframe: {str(e)}")
            return pd.DataFrame

    dropped_columns = []
    for i in range(len(list_X)):
        first_n_column_list = list_X[:i + 1]
        first_n_column = X_n[first_n_column_list]

        scores = cross_validate(estimator, first_n_column, Y_n, cv=kfold, scoring=scoring)
        try:
            df_result.loc[list_X[i]] = [
                round(scores['test_' + score].mean() * (1 if problem_type == 'classification' else -1), 4) for score in
                scoring]
        except Exception as e:
            st.error(f"Error while adding data to result dataframe: {str(e)}")
            return pd.DataFrame
        # create a list to store dropped columns
        if i >= 1:
            if problem_type == 'regression':
                if df_result.loc[list_X[i], 'RMSE'] >= df_result.loc[k, 'RMSE']:
                    dropped_columns.append(list_X[i])  # add column name to list
                    df_result = df_result.drop(list_X[i])
                else:
                    k = list_X[i]
            else:
                if df_result.loc[list_X[i], 'F1'] <= df_result.loc[k, 'F1']:
                    dropped_columns.append(list_X[i])  # add column name to list
                    df_result = df_result.drop(list_X[i])
                else:
                    k = list_X[i]

        # display dropped columns and reasons in Streamlit frontend

    if df_result.isnull().values.any():
        st.header("There are None values in the DataFrame.")
        return pd.DataFrame


    if "df_result" not in st.session_state:
        st.session_state.df_result = df_result
        st.session_state.initial_df = initial_df
        st.session_state.dropping_col = dropped_columns

    feature_graph(df_result, initial_df, problem_type, dropped_columns)


def feature_graph(df_result, initial_df, problem_type, dropped_columns):
    if problem_type == "regression":
        # Define the chart and axis labels for regression
        matrices_to_display = st.multiselect('Select matrices to display', ['MAE', 'MAPE', 'MSE', 'RMSE'],
                                             default=['RMSE'])
        try:
            df_result = df_result.sort_values('RMSE', ascending=False).reset_index()
            initial_df = initial_df.sort_values('RMSE', ascending=False).reset_index()
        except:
            st.error('Can\'t perform operation successfully')
            return
    else:
        # Define the chart and axis labels for classification
        matrices_to_display = st.multiselect('Select matrices to display', ['Accuracy', 'Precision', 'Recall', 'F1'],
                                             default=['Accuracy'])
        try:
            df_result = df_result.sort_values('Accuracy', ascending=True).reset_index()
            initial_df = initial_df.sort_values('Accuracy', ascending=True).reset_index()
        except:
            st.error('Can\'t perform operation successfully')
            return

    df_result = df_result.rename(columns={'index': 'Features'})
    initial_df = initial_df.rename(columns={'index': 'Features'})


    merged_df = pd.merge(initial_df, df_result, on='Features', how='outer', suffixes=('_Baseline', '_Improved'))

    if problem_type == "regression":
        try:
            merged_df = merged_df.sort_values('RMSE_Baseline', ascending=False).reset_index()
        except:
            st.error('Can\'t perform operation successfully')
            return
    else:
        try:
            merged_df = merged_df.sort_values('Accuracy_Baseline', ascending=True).reset_index()
        except:
            st.error('Can\'t perform operation successfully')
            return

    st.write(df_result)


    st.write('#')

    data = pd.concat([df_result[['Features']], df_result[matrices_to_display]], axis=1)
    # data1 = pd.concat([initial_df[['Features']], initial_df[matrices_to_display]], axis=1)

    fig, axs = plt.subplots(len(matrices_to_display)+1, 1, figsize=(10, 7 * len(matrices_to_display)))

    plt.subplots_adjust(hspace=0.5)

    for i, matrix_name in enumerate(matrices_to_display):
        plot_matrix_comparison(merged_df, matrix_name, axs[i])
        axs[i].set_title(matrix_name)
    st.pyplot(fig)

    c0, col1, col2, c1 = st.columns([0.1, 4, 2, .1])

    with col1:
        st.subheader("Selected Features:")
        st.table(data)

    with col2:
        if len(dropped_columns) > 0:
            st.subheader("Dropped Features:")
            st.table(dropped_columns)


def plot_matrix_comparison(merged_df, matrix_name, ax):
    ax.plot(merged_df['Features'], merged_df[f'{matrix_name}_Baseline'], label='Baseline', alpha=0.7, marker='.')
    ax.plot(merged_df['Features'], merged_df[f'{matrix_name}_Improved'], label='Improved', alpha=0.7, marker='.')
    ax.set_xlabel('Features')
    ax.set_ylabel(matrix_name)
    plt.xticks(fontsize=10)
    ax.tick_params(axis='x', rotation=45)
    ax.legend()
