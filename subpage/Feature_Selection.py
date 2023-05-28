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
            feature_graph(st.session_state.df_result, st.session_state.initial_df, problem_type,
                          st.session_state.dropped_columns, st.session_state.selected_features)
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

    st.info("Stage 1: Calculating scores for each feature...")

    total_iterations = len(list_X)

    progress_bar = st.progress(0)

    for i in range(len(list_X)):

        scores = cross_validate(estimator, X_n[list_X[i]].values.reshape(-1, 1), Y_n, cv=kfold, scoring=scoring)
        progress_percentage = (i + 1) / total_iterations * 100

        # Update the progress bar
        progress_bar.progress(progress_percentage)
        try:
            to_sort_df.loc[list_X[i]] = [
                round(scores['test_' + score].mean() * (1 if problem_type == 'classification' else -1), 4) for score in
                scoring]
        except Exception as e:
            st.error(f"Error while adding data to result dataframe: {str(e)}")
            return pd.DataFrame
    st.success("Stage 1 completed!")

    if problem_type == 'classification':
        to_sort_df = to_sort_df.sort_values('F1',ascending=False)
    else:
        to_sort_df=to_sort_df.sort_values('RMSE')

    list_X = to_sort_df.index.tolist()
    k = list_X[0]

    initial_df = df_result

    st.info("Stage 2: Dropping least important features...")

    total_iterations = len(list_X)

    progress_bar = st.progress(0)

    dropped_columns = []

    all_column_data_first=pd.DataFrame
    selected_column_data=pd.DataFrame

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
                initial_df.loc[list_X[i]] = [
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
                initial_df.loc[list_X[i]] = [
                    round(scores_all['test_' + score].mean() * (1 if problem_type == 'classification' else -1), 4) for score
                    in
                    scoring]
            except Exception as e:
                st.error(f"Error while adding data to result dataframe: {str(e)}")
                return pd.DataFrame


        progress_percentage = (i + 1) / total_iterations * 100

        # Update the progress bar
        progress_bar.progress(progress_percentage)


        # create a list to store dropped columns
        if i >= 1:
            if problem_type == 'regression':
                if df_result.loc[list_X[i], 'RMSE'] >= df_result.loc[k, 'RMSE']:
                    dropped_columns.append(list_X[i])  # add column name to list
                    df_result = df_result.drop(list_X[i])
                    selected_column_data.drop(list_X[i])
                else:
                    k = list_X[i]
            else:
                if df_result.loc[list_X[i], 'F1'] <= df_result.loc[k, 'F1']:
                    dropped_columns.append(list_X[i])  # add column name to list
                    df_result = df_result.drop(list_X[i])
                    selected_column_data.drop(list_X[i])
                else:
                    k = list_X[i]

    st.success("Feature selection process completed!")


        # display dropped columns and reasons in Streamlit frontend

    if df_result.isnull().values.any():
        st.header("There are Null values in the DataFrame.")
        return pd.DataFrame


    if "df_result" not in st.session_state:
        st.session_state.df_result = df_result
        st.session_state.initial_df = initial_df
        st.session_state.dropping_col = dropped_columns
    feature_graph(df_result, initial_df, problem_type, dropped_columns, list_X)

def feature_graph(df_result, initial_df, problem_type, dropped_columns, list_X):
    if problem_type == "regression":
        # Define the chart and axis labels for regression
        matrices_to_display = st.multiselect('Select matrices to display', ['MAE', 'MAPE', 'MSE', 'RMSE'],
                                             default=['RMSE'])
        try:
            df_result = df_result.reset_index()
            initial_df = initial_df.reset_index()
        except:
            st.error('Can\'t perform operation successfully')
            return
    else:
        # Define the chart and axis labels for classification
        matrices_to_display = st.multiselect('Select matrices to display', ['Accuracy', 'Precision', 'Recall', 'F1'],
                                             default=['Accuracy'])
        try:
            df_result = df_result.reset_index()
            initial_df = initial_df.reset_index()
        except:
            st.error('Can\'t perform operation successfully')
            return

    #------------
    display_opt = st.radio(
        "",["All","Custom"],index=0,key="display_opt",
        horizontal=True
    )
    selected_features=[]
    if(display_opt=="All"):
        selected_features=''
    elif(display_opt=="Custom"):
        selected_features = st.multiselect("Select features to display", list_X)
        # if st.button('submit'):
        #     feature_graph(df_result, initial_df, problem_type, dropped_columns, selected_features)
    #---------------
    df_result = df_result.rename(columns={'index': 'Features'})
    initial_df = initial_df.rename(columns={'index': 'Features'})

    if selected_features:
        df_result = df_result[df_result['Features'].isin(selected_features)]
        initial_df = initial_df[initial_df['Features'].isin(selected_features)]

    merged_df = pd.merge(initial_df, df_result, on='Features', how='outer', suffixes=('_Baseline', '_Improved'))

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

    # data1 = pd.concat([initial_df[['Features']], initial_df[matrices_to_display]], axis=1)

    fig, axs = plt.subplots(len(matrices_to_display), 1, figsize=(10, 7 * len(matrices_to_display)))

    plt.subplots_adjust(hspace=0.5)

    for i, matrix_name in enumerate(matrices_to_display):
        ax=axs[i] if len(matrices_to_display) > 1 else axs
        plot_matrix_comparison(merged_df, matrix_name, ax)
        ax.set_title(matrix_name)
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
    # Mask the missing values
    mask = np.isfinite(merged_df[f'{matrix_name}_Improved'])

    # Plot the data
    ax.plot(merged_df['Features'], merged_df[f'{matrix_name}_Baseline'], label='Baseline', linestyle=':', marker='.',color="#FF4949")
    ax.plot(merged_df['Features'][mask], merged_df[f'{matrix_name}_Improved'][mask], label='Improved', marker='o',color="#19A7CE")

    ax.set_xlabel('Features',fontsize=12)
    ax.set_ylabel(matrix_name,fontsize=12)
    ax.tick_params(axis='x', rotation=45,labelsize=9)
    ax.legend()
