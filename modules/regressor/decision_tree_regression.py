import time

import pandas as pd
import streamlit as st
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor


def hyperparameter_optimization(X_train, y_train):
    st.subheader("Hyperparameter Optimization Settings")
    n_iter = st.number_input("Number of iterations for hyperparameter search", min_value=1, value=10, step=1)
    cv = st.number_input("Number of cross-validation folds", min_value=2, value=5, step=1)
    random_state = st.number_input("Random state for hyperparameter search", min_value=0, value=0, step=1)

    st.write('#')
    if "dtr_best_param" not in st.session_state:
        st.session_state.dtr_best_param = {
            "criterion": "mse",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": None,
            "random_state": 0
        }

    if st.button('Run Optimization'):
        st.write('#')
        param_dist = {
            "max_depth": [3, 5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "criterion": ["mse", "friedman_mse", "mae"],
            "random_state": [random_state],
        }
        model = DecisionTreeRegressor()

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
        for param, value in best_params.items():
            rows.append([param, value])
        st.table(pd.DataFrame(rows, columns=['Parameter', 'Value']))

        st.session_state.dtr_best_param = best_params

    return st.session_state.dtr_best_param


def decision_tree_regressor(X_train, y_train):
    best_params = hyperparameter_optimization(X_train, y_train)

    st.subheader("Model Settings")

    criterion = st.selectbox(
        "Criterion",
        ["mse", "friedman_mse", "mae"],
        index=["mse", "friedman_mse", "mae"].index(best_params['criterion'] if best_params else 0),
        key="dtr_criterion"
    )

    min_samples_split = st.number_input(
        "Min. Samples Split",
        2, 20,
        value=best_params['min_samples_split'] if best_params else 2,
        key="dtr_min_samples_split"
    )

    min_samples_leaf = st.number_input(
        "Min. Samples Leaf",
        1, 20,
        value=best_params['min_samples_leaf'] if best_params else 1,
        key="dtr_min_samples_leaf"
    )

    auto_max_depth = st.checkbox("None", value=best_params['max_depth'] is None, key="dtr_auto_max_depth")
    if auto_max_depth:
        max_depth = None
    else:
        max_depth = st.number_input(
            "Max Depth",
            1, 20,
            value=best_params['max_depth'] if best_params else 7,
            key="dtr_max_depth"
        )

    random_state = st.number_input(
        "Random State",
        0, 1000000,
        value=best_params['random_state'] if best_params else 0,
        key="dtr_random_state"
    )

    try:
        model = DecisionTreeRegressor(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
    except ValueError as e:
        print(f"Error creating DecisionTreeRegressor: {str(e)}")
        model = None
    except:
        print("Unexpected error creating DecisionTreeRegressor")
        model = None

    return model
