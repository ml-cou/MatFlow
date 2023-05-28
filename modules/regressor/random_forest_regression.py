import time

import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


def hyperparameter_optimization(X_train, y_train):
    st.subheader("Hyperparameter Optimization Settings")
    n_iter = st.number_input("Number of iterations for hyperparameter search", min_value=1, value=10, step=1)
    cv = st.number_input("Number of cross-validation folds", min_value=2, value=5, step=1)
    random_state = st.number_input("Random state for hyperparameter search", min_value=0, value=0, step=1)

    st.write('#')

    if "rfr_best_param" not in st.session_state:
        st.session_state.rfr_best_param = {
            "n_estimators": 10,
            "criterion": "friedman_mse",
            "max_features": "sqrt",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "n_jobs": -1
        }

    if st.button('Run Optimization'):
        st.write('#')
        param_dist = {
            "n_estimators": [10, 50, 100],
            "criterion": ["friedman_mse", "squared_error", "absolute_error", "poisson"],
            "max_features": ["sqrt", "log2", None],
            "max_depth": [3, 5, 10, 15, 20, 25, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "n_jobs": [-1],
        }
        model = RandomForestRegressor(random_state=0)
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
        st.session_state.rfr_best_param = best_params

    return st.session_state.rfr_best_param


def random_forest_regressor(X_train, y_train):
    best_params = hyperparameter_optimization(X_train, y_train)

    st.subheader("Model Settings")

    criterion = st.selectbox(
        "Criterion",
        ["friedman_mse", "squared_error", "absolute_error", "poisson"],
        index=["friedman_mse", "squared_error", "absolute_error", "poisson"].index(
            best_params["criterion"] if best_params else 0
        ),
        key="rfr_criterion"
    )

    max_features = st.selectbox(
        "Max Features",
        ["sqrt", "log2", None],
        index=["sqrt", "log2", None].index(best_params["max_features"] if best_params else "sqrt"),
        key="rfr_max_features"
    )

    min_samples_split = st.slider(
        "Min. Samples Split",
        2, 10, best_params["min_samples_split"] if best_params else 2,1,
        key="rfr_min_samples_split",
    )

    max_depth = st.slider(
        "Max Depth",
        1, 100, best_params["max_depth"] if best_params else 10, 1,
        key="rfr_max_depth",
    )

    min_samples_leaf = st.slider(
        "Min. Samples Leaf",
        1, 10, best_params["min_samples_leaf"] if best_params else 1 , 1 ,
        key="rfr_min_samples_leaf",

    )



    try:
        model = RandomForestRegressor(
            n_estimators=best_params["n_estimators"] if best_params else 10,
            criterion=criterion,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1
        )
    except Exception as e:
        print("Error during model instantiation:", e)
        # Handle the exception here

    return model
