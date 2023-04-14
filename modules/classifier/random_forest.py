import time

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV


def hyperparameter_optimization(X_train, y_train):
    st.subheader("Hyperparameter Optimization Settings")
    n_iter = st.number_input("Number of iterations for hyperparameter search", min_value=1, value=10, step=1)
    cv = st.number_input("Number of cross-validation folds", min_value=2, value=5, step=1)
    random_state = st.number_input("Random state for hyperparameter search", min_value=0, value=0, step=1)

    st.write('#')

    if st.button('Run Optimization'):

        st.write('#')
        param_dist = {
            "n_estimators": [10, 50, 100],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_features": ["sqrt", "log2", None],
            "max_depth": [3, 5, 10, 15, 20, 25, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "n_jobs": [-1],
        }
        model = RandomForestClassifier()

        with st.spinner('Doing hyperparameter optimization...'):
            for i in range(100):
                time.sleep(0.1)
                st.spinner(f"Running iteration {i + 1} of {n_iter}...")
            clf = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, cv=cv,
                                     random_state=random_state)
            st.spinner("Fitting the model...")
            clf.fit(X_train, y_train)

        st.success("Hyperparameter optimization completed!")
        best_params = clf.best_params_

        st.write('Best estimator:')

        rows = []
        for param in best_params:
            rows.append([param, best_params[param]])
        st.table(pd.DataFrame(rows, columns=['Parameter', 'Value']))

        return clf.best_estimator_


def random_forest(X_train, y_train):
    do_hyperparameter_optimization = st.checkbox("Do Hyperparameter Optimization?")

    if do_hyperparameter_optimization:
        hyperparameter_optimization(X_train, y_train)

    st.subheader("Model Settings")

    max_depth = None
    col1, col2, col3 = st.columns(3)
    n_estimators = col1.number_input(
        "Number of Estimators",
        1, 100, 10,
        key="rf_n_estimators"
    )

    criterion = col2.selectbox(
        "Criterion",
        ["gini", "entropy", "log_loss"],
        0,
        key="rf_criterion"
    )

    min_samples_split = col3.number_input(
        "Min. Samples Split",
        2, 20, 2,
        key="rf_min_samples_split"
    )

    col1, col2, col3, col4 = st.columns([3.33, 2, 1.33, 3.33])
    min_samples_leaf = col1.number_input(
        "Min. Samples Leaf",
        1, 20, 1,
        key="rf_min_samples_leaf"
    )

    col3.markdown("#")
    auto_max_depth = col3.checkbox("Auto", True, key="rf_auto_max_depth")

    if auto_max_depth:
        max_depth = col2.text_input(
            "Max Depth",
            "None",
            key="rf_max_depth_none",
            disabled=True
        )
    else:
        max_depth = col2.number_input(
            "Max Depth",
            1, 20, 7,
            key="rf_max_depth"
        )

    random_state = col4.number_input(
        "Random State",
        0, 1000000, 0,
        key="rf_random_state"
    )

    max_depth = None if auto_max_depth else max_depth
    try:
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                       max_depth=max_depth, min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf, random_state=random_state)
    except Exception as e:
        print("An error occurred while initializing the RandomForestClassifier model:", e)
    # Handle the error in an appropriate way (e.g., log the error, display a message to the user, etc.)

    return model
