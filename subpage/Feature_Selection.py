import time

import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import cross_validate
import streamlit as st
def feature_selection(dataset, table_name, problem_type='regression'):
    try:
        if "prev_table" not in st.session_state:
            st.session_state.prev_table=table_name
        elif st.session_state.prev_table==table_name:
            return st.session_state.df_result
        else:
            st.session_state.prev_table=table_name
            del st.session_state.df_result

    except:
        pass


    try:
        tab = dataset[table_name]
    except KeyError:
        st.error(f"The dataset '{table_name}' does not exist.")
        return pd.DataFrame
    if (tab.dtypes == 'object').any():
        problem_type='cl'

    try:
        X_n = tab.iloc[:, :-1]
        Y_n = tab.iloc[:, -1]
    except Exception as e:
        st.error(f"Error while getting input and output data: {str(e)}")
        return pd.DataFrame

    kfold = 5

    if problem_type == 'regression':
        scoring = ['neg_mean_absolute_error', 'neg_mean_absolute_percentage_error',
                   'neg_mean_squared_error', 'neg_root_mean_squared_error']
        df_columns=['MAE', 'MAPE', 'MSE', 'RMSE']
        estimator = ExtraTreesRegressor()
    else:
        scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
        df_columns=['Accuracy', 'Precision', 'Recall', 'F1']
        estimator = ExtraTreesClassifier()


    try:
        list_X = list(X_n.columns)
        k = list_X[0]
        df_result = pd.DataFrame(columns=df_columns)

    except Exception as e:
        st.error(f"Error while initializing variables: {str(e)}")
        return pd.DataFrame



    for i in range(len(list_X)):
        first_n_column_list = list_X[:i + 1]
        first_n_column = X_n[first_n_column_list]

        scores = cross_validate(estimator, first_n_column, Y_n, cv=kfold, scoring=scoring)
        try:
            df_result.loc[list_X[i]] = [round(scores['test_'+score].mean()* (1 if problem_type=='cl' else -1),4) for score in scoring]
        except Exception as e:
            st.error(f"Error while adding data to result dataframe: {str(e)}")
            return pd.DataFrame

        if i >= 1:
            if problem_type == 'regression':
                if df_result.loc[list_X[i], 'RMSE'] >= df_result.loc[k, 'RMSE']:
                    df_result = df_result.drop(list_X[i])
                else:
                    k = list_X[i]
            else:
                if df_result.loc[list_X[i], 'F1'] <= df_result.loc[k, 'F1']:
                    df_result = df_result.drop(list_X[i])
                else:
                    k = list_X[i]
    if df_result.isnull().values.any():
        st.header("There are None values in the DataFrame.")
        return pd.DataFrame

    if "df_result" not in st.session_state:
        st.session_state.df_result=df_result


    return df_result

def feature_graph(df_result, problem_type='regression'):

    if list(df_result.columns)[-1] == 'F1' :
        problem_type='cl'


    if problem_type == "regression":
        # Define the chart and axis labels for regression
        chart = st.line_chart()
        chart.x_axis_label = 'Number of Features'
        chart.y_axis_label = 'Score'
        matrices_to_display = st.multiselect('Select matrices to display', ['MAE', 'MAPE', 'MSE', 'RMSE'], default=['RMSE'])
        df_result = df_result.sort_values('RMSE', ascending=False).reset_index()
    else:
        # Define the chart and axis labels for classification
        chart = st.line_chart()
        chart.x_axis_label = 'Number of Features'
        chart.y_axis_label = 'Accuracy'
        matrices_to_display = st.multiselect('Select matrices to display', ['Accuracy', 'Precision', 'Recall', 'F1'], default=['Accuracy'])
        df_result = df_result.sort_values('Accuracy', ascending=True).reset_index()
    for i in range(len(df_result)):
        try:
            # Select specific matrices to display in the line chart
            time.sleep(0.1)
            data = df_result[matrices_to_display]
            chart.add_rows(data.iloc[[i], :])
        except Exception as e:
            st.error(f"Error while adding rows to the chart: {str(e)}")
            return

    st.dataframe(df_result)


