import time

import pandas as pd
import streamlit as st
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV


def hyperparameter_optimization(X_train, y_train):
    st.subheader("Hyperparameter Optimization Settings")
    n_iter = st.number_input("Number of iterations for hyperparameter search", min_value=1, value=10, step=1)
    cv = st.number_input("Number of cross-validation folds", min_value=2, value=5, step=1)
    random_state = st.number_input("Random state for hyperparameter search", min_value=0, value=0, step=1)

    st.write('#')

    if "svr_best_param" not in st.session_state:
        st.session_state.svr_best_param = {
            "kernel": "rbf",
            "C": 1.0,
            "epsilon": 0.1
        }

    if st.button('Run Optimization'):
        st.write('#')
        param_dist = {
            "kernel": ["linear", "rbf", "poly", "sigmoid"],
            "C": [0.1, 1.0, 10.0],
            "epsilon": [0.1, 0.01, 0.001],
        }
        model = SVR()
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
        st.session_state.svr_best_param = best_params

    return st.session_state.svr_best_param


def support_vector_regressor(X_train, y_train):
    best_params = hyperparameter_optimization(X_train, y_train)

    st.subheader("Model Settings")

    kernel = st.selectbox(
        "Kernel",
        ["linear", "rbf", "poly", "sigmoid"],
        index=["linear", "rbf", "poly", "sigmoid"].index(best_params["kernel"] if best_params else "rbf"),
        key="svr_kernel"
    )

    C = st.number_input(
        "C",
        min_value=0.1,
        value=best_params["C"] if best_params else 1.0,
        step=0.1,
        key="svr_C",
    )

    epsilon = st.number_input(
        "Epsilon",
        min_value=0.01,
        value=best_params["epsilon"] if best_params else 0.1,
        step=0.01,
        key="svr_epsilon",
    )

    try:
        model = SVR(
            kernel=kernel,
            C=C,
            epsilon=epsilon
        )
    except Exception as e:
        print("Error during model instantiation:", e)
        # Handle the exception here

    return model
