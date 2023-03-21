import streamlit as st
from lazypredict.Supervised import LazyRegressor, LazyClassifier
from modules import utils
from sklearn.model_selection import train_test_split
from modules.classifier import decision_tree,knn,log_reg,random_forest,svm,perceptron
def model_evolution():
    # split data format : st.session_state.splitted_data = {'ytest': y_test, 'ytrain': y_train, 'Xtrain': X_train, 'Xtest': X_test,'target_var':target_var}
    try:
        train_data = st.session_state.splitted_data['X_train']
        target_var=st.session_state.splitted_data['target_var']
    except:
        st.header("Split Dataset First")
        return

    # Load data
    data = train_data
    st.write("ML type: ")
    col1, col2 = st.columns([4, 4])
    options = ["Regresion", "Classification"]

    if data[target_var].dtype == "float64" or data[target_var].dtype == "int64":
        st.write(f"{target_var} is numerical")
        default_option = "Regresion"
        with col1:
            selected_option = st.radio("", options, index=options.index(default_option))


        reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
        models, predictions = reg.fit(st.session_state.splitted_data['X_train'], st.session_state.splitted_data['X_test'], st.session_state.splitted_data['y_train'], st.session_state.splitted_data['y_test'])
        with col1:
            st.table(models)

    else:
        default_option = "Classification"
        with col1:
            selected_option = st.radio("", options, index=options.index(default_option))
            st.write(f"{target_var} is not numerical")

        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(st.session_state.splitted_data['X_train'], st.session_state.splitted_data['X_test'], st.session_state.splitted_data['y_train'], st.session_state.splitted_data['y_test'])
        results_df = models

        # Create a selectbox to choose a column from the dataframe
        # selected_column = st.selectbox('Select a column', options=results_df.columns)

        # Get the values of the selected column
        # selected_column_values = results_df[selected_column].tolist()

        # Display the values of the selected column
        # st.write(selected_column_values)
        idx = models.head()
        # st.write(idx.index.values)
        listCls = []
        # with col1:
        st.write(models.index)
        for i in models.index:
            if (i == "decision_tree"):
                listCls.insert(i, decision_tree())
            elif (i == "knn"):
                listCls.insert(i, knn())
            elif (i == "log_reg"):
                listCls.insert(i, log_reg())
            elif (i == "perceptron"):
                listCls.insert(i, perceptron())
            elif (i == "random_forest"):
                listCls.insert(i, random_forest())
            elif (i == "svm"):
                listCls.insert(i, svm())

        models.insert(loc=5, column="Model name", value=[i % 7 + 1 for i in range(len(models.index))])
        st.write(models)
